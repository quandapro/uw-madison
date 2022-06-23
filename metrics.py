import math

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import Callback

import numpy as np
from monai.metrics.utils import get_mask_edges, get_surface_distance

'''
    METRICS AND LOSS FUNCTIONS
'''
def Dice_Coef(spartial_axis = (1,2), ignore_empty = True, smooth=1e-6):
    def Dice_Coef(y_true, y_pred):
        y_pred = tf.math.round(y_pred)
        tp = tf.math.reduce_sum(y_true * y_pred, axis=spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=spartial_axis) # calculate False Positive 
        numerator = 2 * tp
        denominator = 2 * tp + fn + fp
        if ignore_empty:
            non_empty = tf.math.count_nonzero(denominator, dtype=tf.float32)
            return tf.math.reduce_sum(numerator / (denominator + smooth)) / (non_empty + smooth)
        return tf.math.reduce_mean( (numerator + smooth) / (denominator + smooth) )
    return Dice_Coef

def dice_loss(spartial_axis=(1,2,3), smooth = 1e-6):
    def dice_loss(y_true, y_pred):
        tp = tf.math.reduce_sum(y_true * y_pred, axis=spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=spartial_axis) # calculate False Positive
        numerator = 2 * tp + smooth
        denominator = 2 * tp + fn + fp + smooth
        return tf.math.reduce_mean(1 - numerator / denominator) # Average over classes
    return dice_loss

def bce_dice_loss(spartial_axis=(1,2,3), smooth = 1e-6):
    dice_loss_func = dice_loss(spartial_axis, smooth)
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + dice_loss_func(y_true, y_pred)
    return loss

def compute_hausdorff_monai(y_true, y_pred, max_dist):
    if np.all(y_pred == y_true):
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(y_pred, y_true)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 1.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 0.0
    return 1 - dist / max_dist

def hausdorff(y_true, y_pred, max_dist):
    result = []
    for i in range(y_true.shape[-1]):
        result.append(compute_hausdorff_monai(y_true[..., i], y_pred[..., i], max_dist))
    return np.mean(result)

def dice_coef_numpy(y_true, y_pred, axis=(0,1,2), smooth = 1e-6):
    y_pred = np.round(y_pred)
    tp = np.sum(y_true * y_pred, axis=axis) # calculate True Positive
    fn = np.sum(y_true * (1 - y_pred), axis=axis) # calculate False Negative
    fp = np.sum((1 - y_true) * y_pred, axis=axis) # calculate False Positive
    numerator = 2 * tp + smooth
    denominator = 2 * tp + fn + fp + smooth
    return np.mean(numerator / denominator)

class CompetitionMetric(Callback):
    def __init__(self, validation_data, model_checkpoint, shape = (144, 384, 384)):
        super(Callback, self).__init__()
        
        self.validation_data = validation_data
        self.model_checkpoint = model_checkpoint
        self.best_validation_score = -np.inf
        self.max_dists = np.sqrt(np.sum([x ** 2 for x in shape]))
        
    def on_epoch_end(self, epoch, logs={}):
        dice_coef = []
        hausdorff_metric = []
        comp_metric = []
        for X, y_true in self.validation_data:
            y_pred = self.model.predict(X)
            dice_coef.append(dice_coef_numpy(y_true[0], y_pred[0]))
            hausdorff_metric.append(hausdorff(y_true[0], y_pred[0], self.max_dists))
            comp_metric.append(0.4 * dice_coef[-1] + 0.6 * hausdorff_metric[-1])
        mean_dice_coef = np.mean(dice_coef)
        mean_hausdorff_metric = np.mean(hausdorff_metric)
        result = np.mean(comp_metric)
        print(f"val_dice_coef: {mean_dice_coef}; val_hausdorff_metric: {mean_hausdorff_metric}; val_score: {result}")
        if result > self.best_validation_score:
            print(f"Validation score improved from {self.best_validation_score} to {result}. Saving model...")
            self.model.save_weights(self.model_checkpoint)
            self.best_validation_score = result
