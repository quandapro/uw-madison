#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import glob
import cv2
import os
import gc
import random
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *

import tensorflow.keras.backend as K
import math

from unet3d import Unet3D
from augment3d import Augment3D


# # Reproducibility
DEFAULT_RANDOM_SEED = 2022
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--description", type=str, help="model description", default="cascade")
parser.add_argument("--datafolder", type=str, help="Data folder", default="./preprocessed_144x384x384_16_cascade")
parser.add_argument("--prediction", type=str, help="Unet2D prediction folder", default="./prediction/resnet34/resnet34_tda")
parser.add_argument("--batch", type=int, help="Batch size", default=2)
args = parser.parse_args()


# # Configuration


'''
    CONFIGURATION
'''
TRAIN_DIR = './train/' # case/cases_day/scans
preprocessed_folder = args.datafolder
df = pd.read_csv("./preprocessed_train.csv")

MODEL_CHECKPOINTS_FOLDER = './model_checkpoint/'
MODEL_NAME = "UNET_3D"
MODEL_DESC = args.description

NUM_CLASSES = 3
IMAGE_SIZE = (144, 384, 384)
TRAINING_PATCH_SIZE = (16, 384, 384) # First channel = IMAGE + PREDICTED MASK FROM 2D MODEL. SO PATCH 1ST DIMENSION IS 32 * 32 * NUM_CLASSES
PREDICTION_FOLDER = args.prediction
AUGMENT = Augment3D(TRAINING_PATCH_SIZE[0])
BATCH_SIZE = args.batch
KFOLD = 5


UNET_FILTERS = [16, 32, 64, 128, 256]
initial_lr = 1e-3
min_lr = 1e-6
no_of_epochs = 50
epochs_per_cycle = 50

class_map = ["large_bowel", "small_bowel", "stomach"]

if not os.path.exists(f"{preprocessed_folder}"):
    os.makedirs(f"{preprocessed_folder}")

if not os.path.exists(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}"):
    os.makedirs(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}")
    
if not os.path.exists(f"plot/{MODEL_NAME}"):
    os.makedirs(f"plot/{MODEL_NAME}")


# # Preprocessing Dataframe

# RESTRUCTURE  DATAFRAME
df_train = pd.DataFrame({'id':df['id'][::3]})

df_train['large_bowel'] = df['segmentation'][::3].values
df_train['small_bowel'] = df['segmentation'][1::3].values
df_train['stomach'] = df['segmentation'][2::3].values

df_train['path'] = df['path'][::3].values
df_train['case'] = df['case'][::3].values
df_train['day'] = df['day'][::3].values
df_train['slice'] = df['slice'][::3].values

df_train.reset_index(inplace=True,drop=True)
df_train.fillna('',inplace=True); 
df_train['count'] = np.sum(df_train.iloc[:,1:4]!='',axis=1).values
df_train.head()


'''
    Preprocess and save to local disk
'''
# import tqdm

# i = 0
# while i < len(df_train):
#     case = df_train["case"][i]
#     day = df_train["day"][i]
#     # Determine number of slices ahead
#     num_slices = len(os.listdir( os.path.dirname(df_train["path"][i]) ))
            
#     volume = np.load(f"./preprocessed_144x384x384/case{case}_day{day}.npy")
#     volume = np.expand_dims(volume, axis=-1)
#     gt_mask_volume = np.load(f"./preprocessed_144x384x384/case{case}_day{day}_mask.npy")
    
#     pred_mask_volume = []
#     pred_mask_volume = np.empty((num_slices, 384, 384, 3), dtype='float32')
#     for j in range(num_slices): # Loop and preprocess each scan
#         scan_id = df_train["id"][i + j]
#         pred_mask_volume[j] = np.load(f"{PREDICTION_FOLDER}/{scan_id}_predict.npy")
#     pred_mask_volume = center_padding_3d([pred_mask_volume])[0]     
#     volume = np.concatenate([volume, pred_mask_volume], axis=-1)
        
#     d, w, h = TRAINING_PATCH_SIZE
    
#     for m in range(IMAGE_SIZE[0] // d): 
#         patch = volume[m * d:(m + 1) * d]
#         patch_mask = gt_mask_volume[m * d:(m + 1) * d]
#         np.save(f"{preprocessed_folder}/case{case}_day{day}_{m}.npy", patch)
#         np.save(f"{preprocessed_folder}/case{case}_day{day}_{m}_mask.npy", patch_mask)
    
#     i += num_slices

'''
    Data loader
'''
from tensorflow.keras.utils import *

class DataLoader(Sequence):
    def __init__(self, train_ids, batch_size = BATCH_SIZE, shuffle = False, augment = None ):
        self.train_ids = train_ids
        self.batch_size = batch_size 
        self.indices = np.arange(len(train_ids))
        self.augment = augment
        self.shuffle = shuffle
        
    def load_data(self, train_id):
        X = np.load(f"{preprocessed_folder}/{train_id}.npy")
        y = np.load(f"{preprocessed_folder}/{train_id}_mask.npy")
        return X, y

    def __len__(self):
        if len(self.indices) % self.batch_size == 0 or self.shuffle:
            return len(self.indices) // self.batch_size
        return len(self.indices) // self.batch_size + 1

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X = np.empty((len(indices), *TRAINING_PATCH_SIZE, 4), dtype='float32')
        y = np.empty((len(indices), *TRAINING_PATCH_SIZE, 3), dtype='float32')
        for i in range(len(indices)):
            train_id = self.train_ids[indices[i]]
            image, mask = self.load_data(train_id)
            if self.augment is not None:
                X[i], y[i] = self.augment(image, mask)
            else:
                X[i], y[i] = image, mask
        return X, y


from metrics import *
from sklearn.model_selection import GroupKFold

skf = GroupKFold(n_splits=KFOLD)
for fold, (_, val_idx) in enumerate(skf.split(X=df_train, groups =df_train['case']), 1):
    df_train.loc[val_idx, 'fold'] = fold
df_train.head()

def poly_scheduler(epoch, lr, exponent = 0.9):
    return initial_lr * (1 - epoch / no_of_epochs)**exponent

def scheduler(epoch, lr):
    return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2

all_files = os.listdir(f"{preprocessed_folder}")
all_files_wo_mask = [x for x in all_files if 'mask' not in x]
cases_days = [os.path.splitext(x)[0] for x in all_files_wo_mask]

hists = []
for fold in range(1, KFOLD + 1):
    K.clear_session()
    gc.collect()
    
    train_case = [f"case{x}" for x in df_train["case"][df_train["fold"] != fold].values]
    test_case  = [f"case{x}" for x in df_train["case"][df_train["fold"] == fold].values]
    
    train_id = [x for x in cases_days if x.split("_")[0] in train_case]
    test_id = [x for x in cases_days if x.split("_")[0] in test_case]
                 
    train_datagen = DataLoader(train_id, batch_size=BATCH_SIZE, shuffle=True, augment=AUGMENT)
    test_datagen = DataLoader(test_id, batch_size=BATCH_SIZE, shuffle=False, augment=None)
    
    model = Unet3D(conv_settings = UNET_FILTERS, input_shape = (None, None, None, 4), num_classes = 3).build_model()
    model.summary(line_length=150)
    model.compile(optimizer=Adam(), loss=bce_dice_loss, metrics=[dice_large_bowel, dice_small_bowel, dice_stomach, Dice_Coef])
    
    callbacks = [
        ModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', verbose=1, save_best_only=True, monitor="val_Dice_Coef", mode='max'),
        LearningRateScheduler(schedule=scheduler, verbose=1),
        CSVLogger(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.csv', separator=",", append=False)
    ]

    hist = model.fit_generator(train_datagen, 
                               epochs=no_of_epochs, 
                               callbacks = callbacks,
                               validation_data=test_datagen,
                               verbose=2)
    hists.append(hist)
    break

val_Dice_Coef = []
def plot_training_result(history, fold):
    plt.figure(figsize = (8,6))
    plt.plot(history.history['loss'], '-', label = 'train_loss', color = 'g')
    plt.plot(history.history['val_loss'], '--', label = 'valid_loss', color ='r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'Fold {fold}: Loss')
    plt.legend()
    plt.savefig(f"plot/{MODEL_NAME}/{MODEL_DESC}_fold{fold}_loss.png")
    plt.show()
    
    plt.figure(figsize = (8,6))
    plt.plot(history.history['Dice_Coef'], '-', label = 'train_Dice_coef', color = 'g')
    plt.plot(history.history['val_Dice_Coef'], '--', label = 'valid_Dice_coef', color ='r')
    plt.xlabel('epoch')
    plt.ylabel('Dice_Coef')
    plt.title(f'Fold {fold}: Dice Coef')
    plt.legend()
    plt.savefig(f"plot/{MODEL_NAME}/{MODEL_DESC}_fold{fold}_DC.png")
    plt.show()
    
    pass

for i in range(1, KFOLD):
    plot_training_result(hists[i-1], i)
    val_Dice_Coef.append(np.max(hists[i-1].history['val_Dice_Coef']))
    break

print(f"{np.mean(val_Dice_Coef)} +- {np.std(val_Dice_Coef)}")