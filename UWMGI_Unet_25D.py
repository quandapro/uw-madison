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
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from segmentation_models.losses import *
import segmentation_models as sm
from keras_unet_collection import models


from metrics import *
from dataloader import DataLoader
from utils import *

'''
    PARSE ARGUMENTS
'''
parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone")
parser.add_argument("--description", type=str, help="Model description", default="25d")
parser.add_argument("--batch", type=int, help="Batch size", default=16)
parser.add_argument("--datafolder", type=str, help="Data folder", default='preprocessed_384x384_2_25d')
parser.add_argument("--seed", type=int, help="Seed for random generator", default=2022)
parser.add_argument("--csv", type=str, help="Dataframe path", default='preprocessed_train.csv')
parser.add_argument("--trainsize", type=str, help="Training image size", default="224x224x5")
parser.add_argument("--validsize", type=str, help="Training image size", default="384x384x5")
parser.add_argument("--fold", type=int, help="Number of folds", default=5)
parser.add_argument("--epoch", type=int, help="Number of epochs", default=50)
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
    A.Flip(),
    A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.),
    A.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.)
])

initial_lr = 1e-3
min_lr = 1e-6
no_of_epochs = args.epoch

def augment(X, y):
    transformed = TRANSFORM(image=X, mask=y)
    X = transformed["image"]
    y = transformed["mask"]
    return X, y

def plot_training_result(history, fold):
    plt.figure(figsize = (8,6))
    plt.plot(history.history['loss'], '-', label = 'train_loss', color = 'g')
    plt.plot(history.history['val_loss'], '--', label = 'valid_loss', color ='r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Fold {fold}: Loss')
    plt.legend()
    plt.savefig(f"plot/{MODEL_NAME}/{MODEL_DESC}_fold{fold}_loss.png")
    
    plt.figure(figsize = (8,6))
    plt.plot(history.history['Dice_Coef'], '-', label = 'train_Dice_coef', color = 'g')
    plt.plot(history.history['val_Dice_Coef'], '--', label = 'valid_Dice_coef', color ='r')
    plt.xlabel('epoch')
    plt.ylabel('Dice_Coef')
    plt.title(f'Fold {fold}: Dice Coef')
    plt.legend()
    plt.savefig(f"plot/{MODEL_NAME}/{MODEL_DESC}_fold{fold}_DC.png")

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

        # Enable XLA for performance
        tf.config.optimizer.set_jit(True)

        train_id, test_id = DF["id"][DF["fold"] != fold].values, DF["id"][DF["fold"] == fold].values
        train_datagen = DataLoader(train_id, TRAINING_SIZE, (*TRAINING_SIZE[:-1], NUM_CLASSES), DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)
        test_datagen = DataLoader(test_id, VALID_SIZE, (*VALID_SIZE[:-1], NUM_CLASSES), DATAFOLDER, batch_size=BATCH_SIZE, shuffle=False, augment=None)
        
        if MODEL_NAME == "transunet":
            model = models.transunet_2d(TRAINING_SIZE, [16, 64, 128, 256, 512], NUM_CLASSES, weights=None, batch_norm=True, freeze_backbone=False, freeze_batch_norm=False, output_activation='Sigmoid', backbone=None)
            model.summary()
            break
        elif MODEL_NAME == "swinunet":
            model = models.swin_unet_2d(input_size=TRAINING_SIZE, 
                                        filter_num_begin=TRAINING_SIZE[-1], 
                                        n_labels=NUM_CLASSES, 
                                        depth=4, 
                                        stack_num_down=2, 
                                        stack_num_up=2,
                                        patch_size=(4, 4), 
                                        num_heads=[3, 6, 12, 14],
                                        window_size=[7, 7, 7, 7], 
                                        num_mlp=96,
                                        output_activation="Sigmoid")
            model.summary()
            break
        else:
            model = sm.Unet(MODEL_NAME, input_shape=(None, None, TRAINING_SIZE[-1]), classes=NUM_CLASSES, activation='sigmoid', encoder_weights=None)
        
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[dice_large_bowel, dice_small_bowel, dice_stomach, Dice_Coef])
        
        callbacks = [
            ModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', verbose=1, save_best_only=True, monitor="val_Dice_Coef", mode='max'),
            LearningRateScheduler(schedule=cosine_scheduler(initial_lr, min_lr, no_of_epochs), verbose=1),
            CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
        ]
        hist = model.fit_generator(train_datagen, 
                                epochs=no_of_epochs, 
                                callbacks = callbacks,
                                validation_data=test_datagen,
                                verbose=2)
        hists.append(hist)

    # PLOT TRAINING RESULTS
    val_Dice_Coef = []

    for i in range(1, KFOLD + 1):
        plot_training_result(hists[i-1], i)
        val_Dice_Coef.append(np.max(hists[i-1].history['val_Dice_Coef']))

    print(val_Dice_Coef)
    print(f"{np.mean(val_Dice_Coef)} +- {np.std(val_Dice_Coef)}")
    print("Done!")