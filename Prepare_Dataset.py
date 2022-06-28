#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import cv2
import os
import random

'''
    CONFIGURATION
'''
TRAIN_DIR = './train/' # case/cases_day/scans
preprocessed_folder_2d = './preprocessed_2d'
preprocessed_folder_3d = './preprocessed_3d'
df = pd.read_csv("./train.csv")

IMAGE_SIZE_3D = (144, 384, 384)
NUM_CLASSES = 3
TOP_CLIP_PERCENT = 2
BOTTOM_CLIP_PERCENT = 2

class_map = ["large_bowel", "small_bowel", "stomach"]

if not os.path.exists(f"{preprocessed_folder_2d}"):
    os.makedirs(f"{preprocessed_folder_2d}")
    
if not os.path.exists(f"{preprocessed_folder_3d}"):
    os.makedirs(f"{preprocessed_folder_3d}")

df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

paths = []
for i in range(len(df)):
    case = df["case"][i]
    day = df["day"][i]
    slice_num = df["slice"][i]
    paths.append(glob.glob(f"{TRAIN_DIR}/case{case}/case{case}_day{day}/scans/slice_{slice_num}*")[0])

df["path"] = paths
df.to_csv("preprocessed_train.csv")


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
    Image helper function
''' 
def open_image(path):
    image = cv2.imread(path, -1)
    shape = image.shape
    return image, shape

def rle_encoding(mask):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def center_padding_3d(volumes, desired_shape):
    results = []
    for volume in volumes:
        result_volume = np.zeros(desired_shape, dtype=volume.dtype)
        if len(volume.shape) == 4:
            result_volume = np.zeros((*desired_shape, volume.shape[3]), dtype=volume.dtype)
            
        h, w, d = volume.shape[:3]
        desired_h, desired_w, desired_d = desired_shape
        padding_h = (desired_h - h) // 2
        padding_w = (desired_w - w) // 2
        padding_d = (desired_d - d) // 2
        result_volume[padding_h:padding_h + h, padding_w:padding_w + w, padding_d:padding_d + d] = volume
        results.append(result_volume)
    return results

'''
    Preprocessing function
'''
def clipping(volume, top = TOP_CLIP_PERCENT, bottom = BOTTOM_CLIP_PERCENT):
    top_percentile = np.percentile(volume.flatten(), 100 - top)
    bottom_percentile = np.percentile(volume.flatten(), bottom)
    volume[volume > top_percentile] = top_percentile
    volume[volume < bottom_percentile] = 0.
    return volume

def minmax_norm(volume):
    volume = volume.astype('float32')
    min_v = volume.min()
    max_v = volume.max()
    return (volume - min_v) / (max_v - min_v)

def zscore_norm(volume):
    volume = volume.astype('float32')
    non_zero = volume[volume != 0]
    mean = non_zero.mean()
    std = non_zero.std()
    volume[volume != 0] = (non_zero - mean) / std
    return volume

def preprocess(volume, top = TOP_CLIP_PERCENT, bottom = BOTTOM_CLIP_PERCENT):
    # clipped_volume = clipping(volume, top, bottom)
    return minmax_norm(volume)

'''
    Preprocess and save to disk
'''
import tqdm
index = 0
case_index = [0]

while index < len(df_train):
    index += len(os.listdir(os.path.dirname(df_train["path"][index])))
    case_index.append(index)

for i in tqdm.tnrange(0, len(case_index) - 1):
    current_index = case_index[i]
    num_scans = case_index[i + 1] - current_index
    image, size = open_image(df_train["path"][current_index])
    scan_volume = np.empty((num_scans, *size), dtype=image.dtype)
    mask_volume = np.empty((num_scans, *size, 3), dtype='uint8')
    
    case = df_train["case"][current_index]
    day = df_train["day"][current_index]

    # Ignore bad cases
    if (case == 7 and day == 0) or (case == 8 and day == 30):
        continue

    # Create scan and mask volume
    for j in range(num_scans):
        image, size = open_image(df_train["path"][current_index + j])
        mask = np.zeros((*size, 3), dtype='uint8')
        for k in range(3):
            rle = df_train[class_map[k]][current_index + j]      
            if (pd.isna(rle)):
                continue
            decoded = rle_decode(rle, size)
            mask[:,:,k] = decoded
        
        scan_volume[j] = image
        mask_volume[j] = mask

    scan_volume = preprocess(scan_volume)
    
    scan_volume, mask_volume = center_padding_3d([scan_volume, mask_volume], (num_scans, *IMAGE_SIZE_3D[1:]))
    for j in range(num_scans):
        scan_id = df_train["id"][current_index + j]
        np.save(f"{preprocessed_folder_2d}/{scan_id}.npy", scan_volume[j])
        np.save(f"{preprocessed_folder_2d}/{scan_id}_mask.npy", mask_volume[j])

    scan_volume, mask_volume = center_padding_3d([scan_volume, mask_volume], IMAGE_SIZE_3D)
    np.save(f"{preprocessed_folder_3d}/case{case}_day{day}.npy", scan_volume)
    np.save(f"{preprocessed_folder_3d}/case{case}_day{day}_mask.npy", mask_volume)





