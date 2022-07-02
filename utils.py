import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def poly_scheduler(initial_lr, no_of_epochs, exponent = 0.9):
    def scheduler(epoch, lr):
        return initial_lr * (1 - epoch / no_of_epochs)**exponent
    return scheduler

def cosine_scheduler(initial_lr, min_lr, epochs_per_cycle):
    def scheduler(epoch, lr):
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle)) / 2
    return scheduler

def preprocess_dataframe(df):
    result = pd.DataFrame({'id':df['id'][::3]})

    result['large_bowel'] = df['segmentation'][::3].values
    result['small_bowel'] = df['segmentation'][1::3].values
    result['stomach'] = df['segmentation'][2::3].values

    result['path'] = df['path'][::3].values
    result['case'] = df['case'][::3].values
    result['day'] = df['day'][::3].values
    result['slice'] = df['slice'][::3].values

    result.reset_index(inplace=True,drop=True)
    result.fillna('',inplace=True); 
    result['count'] = np.sum(result.iloc[:,1:4]!='',axis=1).values
    return result

def plot_training_result(history, fold, MODEL_NAME, MODEL_DESC):
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