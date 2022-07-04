#!/usr/bin/env python
# coding: utf-8

'''
    Import Libraries
''' 
import argparse
import math
import numpy as np
import pandas as pd
import glob
import cv2
import os
import gc
import random
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
import tensorflow.keras.backend as K
from sklearn.model_selection import GroupKFold

from metrics import *
from model.unet3d import Unet3D
from model.residual_unet3d import ResUnet3D
from dataloader import DataLoader
from utils import seed_everything, poly_scheduler, cosine_scheduler, preprocess_dataframe, plot_training_result
from augment3d import Augment3D

'''
    PARSE ARGUMENTS
'''
parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone", default="unet3d")
parser.add_argument("--description", type=str, help="Model description", default="3d")
parser.add_argument("--batch", type=int, help="Batch size", default=16)
parser.add_argument("--datafolder", type=str, help="Data folder", default='preprocessed_3d')
parser.add_argument("--seed", type=int, help="Seed for random generator", default=2022)
parser.add_argument("--csv", type=str, help="Dataframe path", default='preprocessed_train.csv')
parser.add_argument("--trainsize", type=str, help="Training image size", default="64x224x224x1")
parser.add_argument("--unet", type=str, help="Unet conv settings", default="16x32x64x128x256")
parser.add_argument("--repeat", type=int, help="Unet repeat conv settings", default=2)
parser.add_argument("--ds", type=int, help="Enable deep supervision", default=0)
parser.add_argument("--fold", type=int, help="Number of folds", default=5)
parser.add_argument("--epoch", type=int, help="Number of epochs", default=300)
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
VALID_SIZE = TRAINING_SIZE
BATCH_SIZE = args.batch
KFOLD = args.fold
NUM_CLASSES = 3

augment = Augment3D(TRAINING_SIZE[0], TRAINING_SIZE[1], TRAINING_SIZE[2], NUM_CLASSES)

UNET_FILTERS = [int(x) for x in args.unet.split("x")]
REPEAT = [args.repeat] * len(UNET_FILTERS)
DEEP_SUPERVISION = args.ds != 0
initial_lr = 3e-4
min_lr = 1e-6
no_of_epochs = args.epoch
epochs_per_cycle = no_of_epochs

class_map = ["large_bowel", "small_bowel", "stomach"]

bad_cases = ["case7_day0", "case81_day30", "case138_day0", "case43_day18"]

'''
    MAIN PROGRAM
'''
if __name__ == "__main__":
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

    all_files = os.listdir(f"{DATAFOLDER}")
    all_files_wo_mask = [x for x in all_files if 'mask' not in x]
    cases_days = [os.path.splitext(x)[0] for x in all_files_wo_mask]

    hists = []

    # TRAINING FOR KFOLD
    for fold in range(1, KFOLD + 1):
        # Clear sessions and collect garbages
        K.clear_session()
        gc.collect()

        # Enable XLA for performance
        tf.config.optimizer.set_jit(True)

        train_case = [f"case{x}" for x in DF["case"][DF["fold"] != fold].values]
        test_case  = [f"case{x}" for x in DF["case"][DF["fold"] == fold].values]
        
        train_id = [x for x in cases_days if x.split("_")[0] in train_case]
        test_id = [x for x in cases_days if x.split("_")[0] in test_case]

        [train_id.remove(x) for x in bad_cases if x in train_id]
        [test_id.remove(x) for x in bad_cases if x in train_id]

        train_datagen = DataLoader(train_id, TRAINING_SIZE, (*TRAINING_SIZE[:-1], NUM_CLASSES), DATAFOLDER, batch_size=BATCH_SIZE, shuffle=True, augment=augment)
        test_datagen = DataLoader(test_id, VALID_SIZE, (*VALID_SIZE[:-1], NUM_CLASSES), DATAFOLDER, batch_size=1, shuffle=False, augment=None)
        
        model = Unet3D(conv_settings=UNET_FILTERS, deep_supervision=DEEP_SUPERVISION)()

        optimizer = Adam(learning_rate=initial_lr)
        
        model.compile(optimizer=optimizer, loss=bce_dice_loss(spartial_axis=(0, 1, 2, 3)), metrics=[Dice_Coef(spartial_axis=(2,3), ignore_empty=True)])
        
        callbacks = [
            CompetitionMetric(test_datagen, f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', period=10, deep_supervision=DEEP_SUPERVISION, window_size=TRAINING_SIZE[:3]),
            LearningRateScheduler(schedule=poly_scheduler(initial_lr, no_of_epochs), verbose=1),
            CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
        ]
        hist = model.fit_generator(train_datagen, 
                                epochs=no_of_epochs, 
                                callbacks = callbacks,
                                verbose=2)
        hists.append(callbacks[0].history)
        break

    # PLOT TRAINING RESULTS
    val_Dice_Coef = []

    for i in range(1, KFOLD + 1):
        val_Dice_Coef.append(np.max(hists[i-1]["val_score"]))
        break

    print(val_Dice_Coef)
    print(f"{np.mean(val_Dice_Coef)} +- {np.std(val_Dice_Coef)}")
    print("Done!")