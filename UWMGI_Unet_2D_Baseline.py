import albumentations as A
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import pandas as pd
import glob
import os
import random
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone")
parser.add_argument("--description", type=str, help="Model description")
parser.add_argument("--batch", type=int, help="Batch Size")
args = parser.parse_args()



'''
    CONFIGURATION
'''
DEFAULT_RANDOM_SEED = 2022
TRAIN_DIR = './train/' # case/cases_day/scans
preprocessed_folder = './preprocessed_384x384'
df = pd.read_csv("./preprocessed_train.csv")

MODEL_CHECKPOINTS_FOLDER = './model_checkpoint/'
MODEL_NAME = args.backbone
MODEL_DESC = args.description
MODEL_OUTPUT = f"./prediction/{MODEL_NAME}/{MODEL_DESC}"

print(f"Model name: {MODEL_NAME}. Description: {MODEL_DESC}")

IMAGE_SIZE = (384, 384)
BATCH_SIZE = 16
KFOLD = 5
NUM_CLASSES = 3

initial_lr = 1e-3
min_lr = 1e-6
no_of_epochs = 50
epochs_per_cycle = 50

class_map = ["large_bowel", "small_bowel", "stomach"]

if not os.path.exists(f"{MODEL_OUTPUT}"):
    os.makedirs(f"{MODEL_OUTPUT}")

if not os.path.exists(f"{preprocessed_folder}"):
    os.makedirs(f"{preprocessed_folder}")

if not os.path.exists(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}"):
    os.makedirs(f"{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}")
    
if not os.path.exists(f"plot/{MODEL_NAME}"):
    os.makedirs(f"plot/{MODEL_NAME}")

# REPRODUCIBILITY
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything()

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
df_train.sample(5)


transform = A.Compose([
    A.Flip(),
    A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.),
    A.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.)
])

valid_transform = A.Compose([
])

def augment(X, y):
    transformed = transform(image=X, mask=y)
    X = transformed["image"]
    y = transformed["mask"]
    return X, y

def valid_augment(X, y):
    transformed = valid_transform(image=X, mask=y)
    X = transformed["image"]
    y = transformed["mask"]
    return X, y


'''
    Metric and loss funtions
'''
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from segmentation_models.losses import *

import tensorflow.keras.backend as K
from monai.metrics.utils import get_mask_edges, get_surface_distance
import math

max_distance = np.sqrt(IMAGE_SIZE[0] ** 2 + IMAGE_SIZE[1] ** 2)

# Metrics
from metrics import *

from sklearn.model_selection import GroupKFold

skf = GroupKFold(n_splits=KFOLD)
for fold, (_, val_idx) in enumerate(skf.split(X=df_train, groups =df_train['case']), 1):
    df_train.loc[val_idx, 'fold'] = fold
df_train.head()

import segmentation_models as sm
sm.set_framework("tf.keras")
import gc

def poly_scheduler(epoch, lr, exponent = 0.9):
    return initial_lr * (1 - epoch / no_of_epochs)**exponent

def scheduler(epoch, lr):
    return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2

hists = []
for fold in range(1, KFOLD + 1):
    K.clear_session()
    gc.collect()
    train_id, test_id = df_train["id"][df_train["fold"] != fold].values, df_train["id"][df_train["fold"] == fold].values
    train_datagen = DataLoader(train_id, batch_size=BATCH_SIZE, shuffle=True, augment=augment)
    test_datagen = DataLoader(test_id, batch_size=BATCH_SIZE, shuffle=False, augment=None)
    
    model = sm.Unet(MODEL_NAME, input_shape=(None, None, 1), classes=3, activation='sigmoid', encoder_weights=None)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[dice_large_bowel, dice_small_bowel, dice_stomach, Dice_Coef])
    
    callbacks = [
        ModelCheckpoint(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5', verbose=1, save_best_only=True, monitor="val_Dice_Coef", mode='max'),
        ReduceLROnPlateau(patience=5, factor=0.1, monitor="val_Dice_Coef", mode='max', verbose=1, min_lr=min_lr),
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

import tqdm
datagen = DataLoader(df_train["id"], batch_size=BATCH_SIZE, shuffle=False, augment=None)
i = 0
model.load_weights(f'{MODEL_CHECKPOINTS_FOLDER}/{MODEL_NAME}/{MODEL_DESC}_fold{fold}.h5')
for X, _ in tqdm.tqdm(datagen):
    y = model.predict(X, verbose=0)
    for j in range(len(y)):
        scan_id = df_train["id"].values[i + j]
        np.save(f"{MODEL_OUTPUT}/{scan_id}_predict.npy", y[j])
    i += len(y)