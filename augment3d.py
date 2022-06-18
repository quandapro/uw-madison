import numpy as np
import albumentations as A
import cv2

class Augment3D:
    def __init__(self, image_depth, image_width, image_height):
        keys = {f'image{i}' : 'image' for i in range(1, image_depth)}
        keys.update({f'mask{i}' : 'mask' for i in range(1, image_depth)})
        self.transform = A.Compose([
                                    A.HorizontalFlip(p=0.5),
                                    A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.),
                                    A.GridDistortion(p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0., mask_value=0.)
                                   ], additional_targets=keys)
        self.d = image_depth
        self.w = image_width
        self.h = image_height
    
    def __call__(self, image, mask):
        # Generate random patches 
        d, h, w = image.shape[:3]
        start_d = start_h = start_w = 0
        if d > self.d:
            start_d = np.random.randint(0, d - self.d)
        if h > self.h:
            start_h = np.random.randint(0, h - self.h)
        if w > self.w:
            start_w = np.random.randint(0, w - self.w)

        image = image[start_d:start_d + self.d, start_h:start_h + self.h, start_w:start_w + self.w]
        mask = mask[start_d:start_d + self.d, start_h:start_h + self.h, start_w:start_w + self.w]

        # Add channel axis if necessary
        if len(image.shape) < 4:
            image = np.expand_dims(image, axis=-1)

        data = {f'image{i}' : image[i] for i in range(1, image.shape[0])}
        data.update({f'mask{i}' : mask[i] for i in range(1, image.shape[0])})
        data["image"] = image[0]
        data["mask"] = mask[0]

        aug_data = self.transform(**data)

        aug_image = np.empty(image.shape, dtype='float32')
        aug_image[0] = aug_data["image"]
        aug_mask = np.empty(mask.shape, dtype='float32')
        aug_mask[0] = aug_data["mask"]
        for i in range(1, image.shape[0]):
            aug_image[i] = aug_data[f"image{i}"]
            aug_mask[i] = aug_data[f"mask{i}"]
        return aug_image, aug_mask