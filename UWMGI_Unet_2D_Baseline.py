import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import glob
import cv2
import os
import random
import tensorflow as tf


DEFAULT_RANDOM_SEED = 2022
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything()

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", type=str, help="Unet backbone")
parser.add_argument("--description", type=str, help="model description")
args = parser.parse_args()



'''
    CONFIGURATION
'''
TRAIN_DIR = './train/' # case/cases_day/scans
preprocessed_folder = './preprocessed_384x384'
df = pd.read_csv("./preprocessed_train.csv")

MODEL_CHECKPOINTS_FOLDER = './model_checkpoint/'
MODEL_NAME = args.backbone
MODEL_DESC = args.description
MODEL_OUTPUT = f"./prediction/{MODEL_NAME}/{MODEL_DESC}"

print(f"model name: {MODEL_NAME}. Description: {MODEL_DESC}")

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

'''
    Data loader
'''
from tensorflow.keras.utils import *
import albumentations as A

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

class DataLoader(Sequence):
    def __init__(self, train_ids, batch_size = BATCH_SIZE, shuffle = False, augment = valid_augment ):
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
        X = np.empty((len(indices), *IMAGE_SIZE, 1), dtype='float32')
        y = np.empty((len(indices), *IMAGE_SIZE, 3), dtype='float32')
        for i in range(len(indices)):
            train_id = self.train_ids[indices[i]]
            image, mask = self.load_data(train_id)
            if self.augment is not None:
                X[i, :, : , 0], y[i] = self.augment(image, mask)
            else:
                X[i, :, : , 0], y[i] = image, mask
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
def dice_large_bowel(y_true, y_pred, smooth = 1e-7):
    y_true_f = K.flatten(y_true[..., 0])
    y_pred_f = tf.math.round(K.flatten(y_pred[..., 0]))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_small_bowel(y_true, y_pred, smooth = 1e-7):
    y_true_f = K.flatten(y_true[..., 1])
    y_pred_f = tf.math.round(K.flatten(y_pred[..., 1]))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_stomach(y_true, y_pred, smooth = 1e-7):
    y_true_f = K.flatten(y_true[..., 2])
    y_pred_f = tf.math.round(K.flatten(y_pred[..., 2]))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def Dice_Coef(y_true, y_pred):
    return (dice_large_bowel(y_true, y_pred) + dice_small_bowel(y_true, y_pred) + dice_stomach(y_true, y_pred)) / 3

def custom_bce(y_true, y_pred):
    bce = 0.
    for i in range(3):
        bce += binary_crossentropy(y_true[..., i], y_pred[..., i])
    return bce / 3

def compute_hausdorff_monai(pred, gt, max_dist = max_distance):
    if np.all(pred == gt):
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 1.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 0.0
    return 1 - dist / max_dist

def hausdorff(y_true, y_pred):
    y_true = np.round(y_true)
    y_pred = np.round(y_pred)
    hdos = []
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[3]):
            pred = y_pred[i,:,:,j]
            gt = y_true[i,:,:,j]
            hdos.append(compute_hausdorff_monai(pred, gt))
    return np.min(hdos) # Assume worst case scenario for a single case_day

def dice_coef_numpy(y_true, y_pred, smooth = 1e-6):
    dice_coef = []
    for i in range(y_true.shape[0]):
        for j in range(y_true.shape[-1]):
            y_true_f = np.round(np.ndarray.flatten(y_true[i,..., j]))
            y_pred_f = np.round(np.ndarray.flatten(y_pred[i,..., j]))

            denom = np.sum(y_true_f) + np.sum(y_pred_f)

            intersection = np.sum(np.dot(y_true_f, y_pred_f))
            dice_coef.append((2 * intersection + smooth) / (denom + smooth))
    return dice_coef / NUM_CLASSES

def competition_metric(y_true, y_pred):
    return 0.4 * dice_coef_numpy(y_true, y_pred) + 0.6 * hausdorff(y_true, y_pred)

class CompetitionMetric(Callback):
    def __init__(self, validation_data, model_checkpoint):
        super(Callback, self).__init__()
        
        self.validation_data = validation_data
        self.model_checkpoint = model_checkpoint
        self.best_validation_score = -np.inf
        
    def on_epoch_end(self, epoch, logs={}):
        dice_coef = []
        hausdorff_metric = []
        comp_metric = []
        for X, y_true in self.validation_data:
            y_pred = model.predict(X)
            dice_coef.append(dice_coef_numpy(y_true, y_pred))
            hausdorff_metric.append(hausdorff(y_true, y_pred))
            comp_metric.append(0.4 * dice_coef[-1] + 0.6 * hausdorff_metric[-1])
        mean_dice_coef = np.mean(dice_coef)
        mean_hausdorff_metric = np.mean(hausdorff_metric)
        result = np.mean(comp_metric)
        print(f"val_dice_coef: {mean_dice_coef}; val_hausdorff_metric: {mean_hausdorff_metric}; val_score: {result}")
        if result > self.best_validation_score:
            print(f"Validation score improved from {self.best_validation_score} to {result}. Saving model...")
            self.model.save_weights(self.model_checkpoint)
            self.best_validation_score = result

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
    y = (y > 0.5).astype('uint8')
    for j in range(len(y)):
        scan_id = df_train["id"].values[i + j]
        np.save(f"{MODEL_OUTPUT}/{scan_id}_predict.npy", y[j])
    i += len(y)