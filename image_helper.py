import cv2
import numpy as np

'''
    Image helper functions
''' 
def open_image(path):
    image = cv2.imread(path, -1)
    shape = image.shape
    return image, shape

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

def restore_original_3d(image, mask, original_shape):
    h, w, d = mask.shape
    original_h, original_w, original_d = original_shape
    padding_h = (h - original_h) // 2
    padding_w = (w - original_w) // 2
    padding_d = (d - original_d) // 2
    result_image = image[padding_h:padding_h + original_h, padding_w:padding_w + original_w, padding_d:padding_d + original_d]
    result_mask = mask[padding_h:padding_h + original_h, padding_w:padding_w + original_w, padding_d:padding_d + original_d]
    return result_image, result_mask