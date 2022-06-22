import numpy as np
from tensorflow.keras.utils import Sequence

class DataLoader(Sequence):
    def __init__(self, train_ids, image_size, mask_size, datafolder, batch_size, shuffle, augment):
        self.train_ids = train_ids
        self.image_size = image_size
        self.mask_size = mask_size
        self.datafolder = datafolder
        self.batch_size = batch_size 
        indices = np.arange(len(train_ids))
        if shuffle:
            np.random.shuffle(indices)
        self.indices = indices
        self.augment = augment
        self.shuffle = shuffle
        
    def load_data(self, train_id):
        X = np.load(f"{self.datafolder}/{train_id}.npy")
        y = np.load(f"{self.datafolder}/{train_id}_mask.npy")
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
        X = np.empty((len(indices), *self.image_size), dtype='float32')
        y = np.empty((len(indices), *self.mask_size), dtype='float32')
        for i in range(len(indices)):
            train_id = self.train_ids[indices[i]]
            image, mask = self.load_data(train_id)
            if self.augment is not None:
                X[i], y[i] = self.augment(image, mask)
            else:
                X[i], y[i] = image.reshape(X.shape[1:]), mask
        return X, y

class SmartDataLoader(Sequence):
    def __init__(self, train_ids, image_size, mask_size, datafolder, batch_size, shuffle, augment):
        self.train_ids = train_ids
        self.image_size = image_size
        self.mask_size = mask_size
        self.datafolder = datafolder
        self.batch_size = batch_size 
        indices = np.arange(len(train_ids))
        if shuffle:
            np.random.shuffle(indices)
        self.indices = indices
        self.augment = augment
        self.shuffle = shuffle

    def clipping(volume, top = 2, bottom = 2):
        top_percentile = np.percentile(volume.flatten(), 100 - top)
        bottom_percentile = np.percentile(volume.flatten(), bottom)
        volume[volume > top_percentile] = top_percentile
        volume[volume < bottom_percentile] = 0.
        return volume

    def make_divisible(self, num, divisible=16):
        mod = num % divisible
        if mod == 0:
            return num
        return num + (divisible - mod)

    def center_padding_3d(self, volumes, divisible=16):
        results = []
        for volume in volumes:      
            h, w, d = volume.shape[:3]
            desired_h, desired_w, desired_d = self.make_divisible(h, divisible), self.make_divisible(w, divisible), self.make_divisible(d, divisible)
            desired_shape = (desired_h, desired_w, desired_d)
            
            padding_h = (desired_h - h) // 2
            padding_w = (desired_w - w) // 2
            padding_d = (desired_d - d) // 2

            result_volume = np.full(desired_shape, fill_value=volume.min(), dtype=volume.dtype)
            if len(volume.shape) == 4:
                result_volume = np.zeros((*desired_shape, volume.shape[3]), dtype=volume.dtype)
            result_volume[padding_h:padding_h + h, padding_w:padding_w + w, padding_d:padding_d + d] = volume
            results.append(result_volume)
        return results
        
    def load_data(self, train_id):
        X = np.load(f"{self.datafolder}/{train_id}.npy")
        y = np.load(f"{self.datafolder}/{train_id}_mask.npy")

        # Preprocessing here
        X = X.astype('float32')
        X = ( X - X.mean() ) / X.std()

        # Padding if validation
        if not self.shuffle: 
            X, y = self.center_padding_3d([X, y])
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
        X = []
        y = []
        if self.shuffle:
            X = np.empty((len(indices), *self.image_size), dtype='float32')
            y = np.empty((len(indices), *self.mask_size), dtype='float32')

        for i in range(len(indices)):
            train_id = self.train_ids[indices[i]]
            image, mask = self.load_data(train_id)
            if self.augment is not None:
                X[i], y[i] = self.augment(image, mask)
            else:
                X.append(np.expand_dims(image, -1))
                y.append(mask)
        if self.shuffle:
            return X, y
        return np.array(X, dtype='float32'), np.array(y, dtype='float32')