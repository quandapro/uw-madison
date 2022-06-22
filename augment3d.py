import numpy as np
import albumentations as A
import cv2

class Augment3D:
    '''
        Performs data augmentation on 3D data
    '''
    def __init__(self, image_depth, image_width, image_height, num_class):
        keys = {f'mask{i}' : 'mask' for i in range(num_class)}
        self.transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.ShiftScaleRotate(p=0.5),
                            A.GridDistortion(p=0.2),
                            A.RandomGamma(p=0.2)
                        ], additional_keys=keys)
        self.d = image_depth
        self.w = image_width
        self.h = image_height
        self.num_class = num_class
    
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

        # Augment on first two axis
        data = {
            'image' : image,
        }
        data.update({f"mask{i}" : mask[..., i] for i in range(self.num_class)})
        aug_data = self.transform(**data)
        image = aug_data["image"]
        for i in range(self.num_class):
            mask[..., i] = aug_data[f"mask{i}"]

        # Augment on the other axis
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1, 3))

        data = {
            'image' : image,
        }
        data.update({f"mask{i}" : mask[..., i] for i in range(self.num_class)})
        aug_data = self.transform(**data)

        image = aug_data["image"]
        for i in range(self.num_class):
            mask[..., i] = aug_data[f"mask{i}"]

        # Restore axis
        image = image.transpose((1, 2, 0))
        mask = mask.transpose((1, 2, 0, 3))

        # Add channel axis if necessary
        if len(image.shape) < 4:
            image = np.expand_dims(image, axis=-1)

        return image, mask