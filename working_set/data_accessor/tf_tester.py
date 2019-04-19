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




# def fucking_wrapper():
#     modulation_targets = '32QAM', 'FM'

#     snr_targets = [30]

#     ds_accessor = Deepsig_Accessor(
#         modulation_targets, snr_targets, 0.75, batch_size=200, throw_after_epoch=True, shuffle=True)
    
#     ds_training_generator = ds_accessor.get_training_generator()

#     return ds_training_generator

modulation_targets = '32QAM', 'FM'

snr_targets = [30]

ds_accessor = Deepsig_Accessor(
    modulation_targets, snr_targets, 0.75, batch_size=200, throw_after_epoch=True, shuffle=True)


# Build the god damn thing
with tf.python_io.TFRecordWriter('fugg.tfrecord') as writer:

    for sample in ds_accessor.get_training_generator():

        train_x_list = tf.train.FloatList(value=sample[0])
        train_y_list = tf.train.Int64List(value=sample[1])


        X = tf.train.Feature(float_list=train_x_list)
        Y = tf.train.Feature(int64_list=train_y_list)

        train_dict = {
            'X': X,
            'Y': Y
        }
        features = tf.train.Features(feature=train_dict)

        example = tf.train.Example(features=features)

        writer.write(example.SerializeToString())

def transform_to_orig(proto):
    features = {
        'X':         tf.VarLenFeature(tf.float32),
        'Y':        tf.VarLenFeature(tf.int64)
    }

    parsed_features = tf.parse_single_example(proto, features)

    X = tf.sparse_tensor_to_dense(parsed_features['X'])
    Y = tf.sparse_tensor_to_dense(parsed_features['Y'])

    X = tf.reshape(X, [DATASET_LEN_X])
    Y = tf.reshape(Y, [DATASET_LEN_Y])

    return X,Y


ds = tf.data.TFRecordDataset("fugg.tfrecord").map(transform_to_orig)


total_len = 0
for i in ds:
    print(i)
    total_len += 1

print(total_len)



