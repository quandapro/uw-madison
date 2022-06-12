import tensorflow as tf
import random
import os
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_RANDOM_SEED = 2022
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
