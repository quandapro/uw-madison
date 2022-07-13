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

df = pd.read_csv("./train.csv")

IMAGE_SIZE_2D = (384, 384)
ADJACENT_SLICES = 2
STRIDE = 1
NUM_CLASSES = 3

preprocessed_folder_2d = f'./preprocessed_2d'
preprocessed_folder_25d = f'./preprocessed_25d'

class_map = ["large_bowel", "small_bowel", "stomach"]

if not os.path.exists(f"{preprocessed_folder_25d}"):
    os.makedirs(f"{preprocessed_folder_25d}")

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
    for j in range(0, num_scans):
        scan_id = df_train["id"][current_index + j]
        result_image = np.zeros((*IMAGE_SIZE_2D, ADJACENT_SLICES * 2 + 1), dtype='float32')
        result_mask = np.load(f"{preprocessed_folder_2d}/{scan_id}_mask.npy")

        for k in range(-ADJACENT_SLICES - STRIDE + 1, ADJACENT_SLICES + STRIDE, STRIDE): # -2 -1 0 1 2
            if j + k < 0 or j + k >= num_scans:
                continue
            scan_id = df_train["id"][current_index + j + k]
            result_image[..., k // STRIDE + ADJACENT_SLICES] = np.load(f"{preprocessed_folder_2d}/{scan_id}.npy")

        scan_id = df_train["id"][current_index + j]
        np.save(f"{preprocessed_folder_25d}/{scan_id}.npy", result_image)
        np.save(f"{preprocessed_folder_25d}/{scan_id}_mask.npy", result_mask)