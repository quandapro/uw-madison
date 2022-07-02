'''
    IMPORT LIBRARIES
'''
import albumentations as A
import argparse
import cv2
import gc
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import GroupKFold
import os

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from segmentation_models.metrics import *
import segmentation_models as sm
from keras_unet_collection.models import unet_3plus_2d

from metrics import *
from dataloader import DataLoader
from utils import *
from model.residual_unet2d import ResUnet2D
from model.unet2d import Unet2D

'''
    PARSE ARGUMENTS
'''
parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone")
parser.add_argument("--description", type=str, help="Model description", default="25d")
parser.add_argument("--batch", type=int, help="Batch size", default=32)
parser.add_argument("--datafolder", type=str, help="Data folder", default='preprocessed_384x384_2_25d')
parser.add_argument("--seed", type=int, help="Seed for random generator", default=2022)
parser.add_argument("--csv", type=str, help="Dataframe path", default='preprocessed_train.csv')
parser.add_argument("--trainsize", type=str, help="Training image size", default="224x224x5")
parser.add_argument("--validsize", type=str, help="Validation image size", default="384x384x5")
parser.add_argument("--fold", type=int, help="Number of folds", default=5)
parser.add_argument("--epoch", type=int, help="Number of epochs", default=100)
args = parser.parse_args()

'''
    CONFIGURATION
'''
RANDOM_SEED = args.seed
DATAFOLDER = args.datafolder
DF = pd.read_csv(args.csv)

MODEL_CHECKPOINTS_FOLDER = './model_checkpoint/'
MODEL_NAME = args.backbone
MODEL_DESC = args.description

TRAINING_SIZE = tuple([int(x) for x in args.trainsize.split("x")])
VALID_SIZE = tuple([int(x) for x in args.validsize.split("x")])
BATCH_SIZE = args.batch
KFOLD = args.fold
NUM_CLASSES = 3

TRANSFORM = A.Compose([
    A.RandomCrop(TRAINING_SIZE[0], TRAINING_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.GridDistortion(p=0.3),
    A.RandomGamma(p=0.3),
])

bad_cases = ["case7_day0", "case81_day30"]

initial_lr = 3e-4
min_lr = 1e-6
no_of_epochs = args.epoch

def augment(X, y):
    transformed = TRANSFORM(image=X, mask=y)
    X = transformed["image"]
    y = transformed["mask"]
    return X, y

def is_bad(x):
    for i in bad_cases:
        if i in x:
            return True
    return False

'''
    MAIN PROGRAM
'''
if __name__ == "__main__":
    sm.set_framework("tf.keras")

    # REPRODUCIBILITY
    seed_everything(RANDOM_SEED)

    print(f"Model name: {MODEL_NAME}. Description: {MODEL_DESC}")
    class_map = ["large_bowel", "small_bowel", "stomach"]

    if not os.path.exists(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}"):
        os.makedirs(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}")
        
    if not os.path.exists(f"plot/{MODEL_NAME}"):
        os.makedirs(f"plot/{MODEL_NAME}")

    # PREPROCESSING DATAFRAME
    DF = preprocess_dataframe(DF)

    # SPLIT DATA INTO KFOLD
    skf = GroupKFold(n_splits=KFOLD)
    for fold, (_, val_idx) in enumerate(skf.split(X=DF, groups =DF['case']), 1):
        DF.loc[val_idx, 'fold'] = fold

    hists = []

    # TRAINING FOR KFOLD
    for fold in range(1, KFOLD + 1):
        # Clear sessions and collect garbages
        K.clear_session()
        gc.collect()

        monitor = 'val_Dice_Coef'

        # Enable XLA for performance
        tf.config.optimizer.set_jit(True)

        train_id, test_id = DF["id"][DF["fold"] != fold].values, DF["id"][DF["fold"] == fold].values
        cleaned_train_id = [x for x in train_id if not is_bad(x)]
        cleaned_test_id = [x for x in test_id if not is_bad(x)]

        train_datagen = DataLoader(cleaned_train_id, TRAINING_SIZE, (*TRAINING_SIZE[:-1], NUM_CLASSES), DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)
        test_datagen = DataLoader(cleaned_test_id, VALID_SIZE, (*VALID_SIZE[:-1], NUM_CLASSES), DATAFOLDER, batch_size=16, shuffle=False, augment=None)
        
        if MODEL_NAME == "ResUnet2D_DS":
            model = ResUnet2D(input_shape=(None, None, TRAINING_SIZE[-1]), deep_supervision=True)()
            monitor = 'val_output_final_Dice_Coef'
        elif MODEL_NAME == "Unet2D_DS":
            model = Unet2D(input_shape=(None, None, TRAINING_SIZE[-1]), deep_supervision=True)()
            monitor = 'val_output_final_Dice_Coef'
        else:
            model = sm.Unet(MODEL_NAME, input_shape=(None, None, TRAINING_SIZE[-1]), classes=NUM_CLASSES, activation='sigmoid', encoder_weights=None)
        
        optimizer = Adam(learning_rate=initial_lr)

        model.compile(optimizer=optimizer, loss=bce_dice_loss(spartial_axis=(0, 1, 2)), metrics=[Dice_Coef(spartial_axis=(1, 2), ignore_empty=False)])
        
        callbacks = [
            ModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', verbose=1, save_best_only=True, monitor=monitor, mode='max'),
            LearningRateScheduler(schedule=poly_scheduler(initial_lr, no_of_epochs), verbose=1),
            CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
        ]
        hist = model.fit_generator(train_datagen, 
                                epochs=no_of_epochs, 
                                callbacks = callbacks,
                                validation_data=test_datagen,
                                verbose=2)
        hists.append(hist)
        break

    # PLOT TRAINING RESULTS
    val_Dice_Coef = []

    for i in range(1, KFOLD + 1):
        plot_training_result(hists[i-1], i, MODEL_NAME, MODEL_DESC)
        val_Dice_Coef.append(np.max(hists[i-1].history['val_Dice_Coef']))
        break

    print(val_Dice_Coef)
    print(f"{np.mean(val_Dice_Coef)} +- {np.std(val_Dice_Coef)}")
    print("Done!")