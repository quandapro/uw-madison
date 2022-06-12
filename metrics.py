import tensorflow as tf
import math
import tensorflow.keras.backend as K

# Metrics
def dice_large_bowel(y_true, y_pred, smooth = 1e-6):
    y_true_f = K.flatten(y_true[..., 0])
    y_pred_f = tf.math.round(K.flatten(y_pred[..., 0]))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_small_bowel(y_true, y_pred, smooth = 1e-6):
    y_true_f = K.flatten(y_true[..., 1])
    y_pred_f = tf.math.round(K.flatten(y_pred[..., 1]))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def dice_stomach(y_true, y_pred, smooth = 1e-6):
    y_true_f = K.flatten(y_true[..., 2])
    y_pred_f = tf.math.round(K.flatten(y_pred[..., 2]))
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def Dice_Coef(y_true, y_pred):
    return (dice_large_bowel(y_true, y_pred) + dice_small_bowel(y_true, y_pred) + dice_stomach(y_true, y_pred)) / 3