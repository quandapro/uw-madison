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

def tversky_loss(alpha=0.5, spartial_axis=(1,2,3), smooth = 1e-6):
    '''
        Tversky loss
        High alpha -> Emphasize on false positive / Specificity
        Low alpha -> Emphasize on false negative / Sensitivity
    '''
    def tversky_loss(y_true, y_pred):
        tp = tf.math.reduce_sum(y_true * y_pred, axis=spartial_axis) # calculate True Positive
        fn = tf.math.reduce_sum(y_true * (1 - y_pred), axis=spartial_axis) # calculate False Negative
        fp = tf.math.reduce_sum((1 - y_true) * y_pred, axis=spartial_axis) # calculate False Positive
        numerator = tp + smooth
        denominator = tp + alpha * fp + (1 - alpha) * fn + smooth
        return tf.math.reduce_mean(1 - numerator / denominator) # Average over classes
    return tversky_loss

def bce_tversky_loss(alpha=0.5, spartial_axis=(1,2,3), smooth = 1e-6):
    tversky_loss_func = tversky_loss(alpha, spartial_axis, smooth)
    def loss(y_true, y_pred):
        return binary_crossentropy(y_true, y_pred) + tversky_loss_func(y_true, y_pred)
    return loss

def compute_directed_hausdorff(pred, gt, max_dist):
    if np.all(pred == gt):
        return 0.0
    if np.sum(pred) == 0:
        return 1.0
    if np.sum(gt) == 0:
        return 1.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()

    if dist > max_dist:
        return 1.0
    return dist / max_dist

def hausdorff(y_true, y_pred, max_dist):
    result = []
    for i in range(y_true.shape[-1]):
        result.append(1.0 - compute_directed_hausdorff(y_pred[..., i], y_true[..., i], max_dist))
    return np.mean(result)

def dice_coef_numpy(y_true, y_pred, axis=(1,2), smooth = 1e-6, ignore_empty=True):
    y_pred = np.round(y_pred)
    tp = np.sum(y_true * y_pred, axis=axis) # calculate True Positive
    fn = np.sum(y_true * (1 - y_pred), axis=axis) # calculate False Negative
    fp = np.sum((1 - y_true) * y_pred, axis=axis) # calculate False Positive
    numerator = 2 * tp
    denominator = 2 * tp + fn + fp
    if ignore_empty:
        non_empty = float(np.count_nonzero(denominator))
        return np.sum(numerator / (denominator + smooth)) / (non_empty + smooth)
    return np.mean((numerator + smooth) / (denominator + smooth))

from timeit import default_timer as timer

class CompetitionMetric(Callback):
    def __init__(self, validation_data, model_checkpoint, period = 10, deep_supervision = True, window_size = (64, 224, 224)):
        super(Callback, self).__init__()
        
        self.validation_data = validation_data
        self.model_checkpoint = model_checkpoint
        self.best_validation_score = -np.inf
        self.deep_supervision = deep_supervision
        self.period = period
        self.history = {
            "val_dice_coef" : [],
            "val_hausdorff" : [],
            "val_score"     : []
        } 
        self.window_size = window_size
        self.stride = tuple([x // 2 for x in window_size])
        
    def sliding_window_inference(self, volume):
        '''
            Sliding window inference
            --------
            volume : numpy.ndarray
                Input 3D volume with shape = (1, Depth, Height, Width, 1)
            model : tf.keras.Model
                Inference model output with shape = (1, Depth, Height, Width, 3)
    
            --------
            return : numpy.ndarray
                Output segmentation
        '''
        d, h, w = volume.shape[1:4]
        w_d, w_h, w_w = self.window_size
        s_d, s_h, s_w = self.stride
        result = np.zeros((*volume.shape[:4], 3), dtype='float32')
        overlap = np.zeros((*volume.shape[:4], 3), dtype='float32')
        starting_points = [(x, y, z) for x in set( list(range(0, d - w_d, s_d)) + [d - w_d] ) 
                                     for y in set( list(range(0, h - w_h, s_h)) + [h - w_h] ) 
                                     for z in set( list(range(0, w - w_w, s_w)) + [w - w_w] )]

        patches = np.empty((len(starting_points), *self.window_size, 1), dtype='float32')
        for i, (x, y, z) in enumerate(starting_points):
            patches[i] = volume[0, x:x + w_d, y:y + w_h, z:z + w_w, :]

        y_pred = self.model.predict(patches, batch_size = 4)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        
        for i in range(len(y_pred)):
            x, y, z = starting_points[i]
            result[:, x:x + w_d, y:y + w_h, z:z + w_w, :] += y_pred[i]
            overlap[:, x:x + w_d, y:y + w_h, z:z + w_w, :] += 1.

        assert np.sum(overlap == 0.) == 0, "Sliding window does not cover all volume"

        return result / overlap

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.period == 0:
            start = timer()

            dice_coef = []
            hausdorff_metric = []
            comp_metric = []
            for X, y_true in self.validation_data:
                y_pred = self.sliding_window_inference(X)
                # Thresholding
                y_pred = (y_pred > 0.5).astype('float32')
                dice_coef.append(dice_coef_numpy(y_true[0], y_pred[0]))
                max_dist = np.sqrt(np.sum([x ** 2 for x in y_pred.shape[1:4]]))
                hausdorff_metric.append(hausdorff(y_true[0], y_pred[0], max_dist))
                comp_metric.append(0.4 * dice_coef[-1] + 0.6 * hausdorff_metric[-1])

            mean_dice_coef = np.mean(dice_coef)
            mean_hausdorff_metric = np.mean(hausdorff_metric)
            result = np.mean(comp_metric)
            
            self.history["val_dice_coef"].append(mean_dice_coef)
            self.history["val_hausdorff"].append(mean_hausdorff_metric)
            self.history["val_score"].append(result)

            print(f"val_dice_coef: {mean_dice_coef}; val_hausdorff_metric: {mean_hausdorff_metric}; val_score: {result}")
            if result > self.best_validation_score:
                print(f"Validation score improved from {self.best_validation_score} to {result}. Saving model...")
                self.model.save_weights(self.model_checkpoint)
                self.best_validation_score = result
            else:
                print(f"Validation score does not improve from: {self.best_validation_score}")
            end = timer()
            print(f"Finished in {end - start}s")
