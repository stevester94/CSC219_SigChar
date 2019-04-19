from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
# Import MNIST data
from deepsig_accessor import Deepsig_Accessor
import random
from time import sleep
import numpy as np

tf.enable_eager_execution()

DATASET_LEN_X = 2048
DATASET_LEN_Y = 24




def fucking_wrapper():
    modulation_targets = '32QAM', 'FM'

    snr_targets = [30]

    ds_accessor = Deepsig_Accessor(
        modulation_targets, snr_targets, 0.75, batch_size=200, throw_after_epoch=True, shuffle=True)
    
    ds_training_generator = ds_accessor.get_training_generator()

    return ds_training_generator

ds = tf.data.Dataset.from_generator(
    fucking_wrapper, (tf.float32, tf.int64), (tf.TensorShape([DATASET_LEN_X]), tf.TensorShape([DATASET_LEN_Y])))


for value in ds.take(1):
  print(value)
