from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
# Import MNIST data
from deepsig_accessor import Deepsig_Accessor
import random
from time import sleep
import numpy as np

from multiprocessing import Pool

tf.enable_eager_execution()

DATASET_LEN_X = 2048
DATASET_LEN_Y = 24

BASE_DIR = "/media/steven/1TB/CSC219_Data/"

all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]

limited_snr = [-20, -10, 0, 10, 20, 30]
low_snr = [-20, -10]
medium_snr = [0, 10]
high_snr = [20, 30]
thirty_snr = [30]

subset_targets = []

# subset_targets.append((subset_modulation_targets, [30]))
# subset_targets.append((easy_modulation_targets, [30]))
# subset_targets.append((all_modulation_targets, [30]))

# subset_targets.append( (subset_modulation_targets, limited_snr) )
# subset_targets.append((easy_modulation_targets, limited_snr))
# subset_targets.append((all_modulation_targets, limited_snr))

# subset_targets.append((subset_modulation_targets, high_snr))
# subset_targets.append((easy_modulation_targets, high_snr))
# subset_targets.append((all_modulation_targets, high_snr))

# subset_targets.append((subset_modulation_targets, all_snr_targets))
# subset_targets.append((easy_modulation_targets, all_snr_targets))
# subset_targets.append((all_modulation_targets, all_snr_targets))

# subset_targets.append((subset_modulation_targets, thirty_snr))

subset_targets.append((all_modulation_targets, low_snr))
subset_targets.append((all_modulation_targets, medium_snr))
subset_targets.append((all_modulation_targets, high_snr))




def convert_hdf5_to_dataset_file(target):
    mod_names = '_'.join(mod for mod in target[0])
    snr_names = '_'.join(str(snr) for snr in target[1])
    prelim_filename = BASE_DIR + mod_names + snr_names
    print(prelim_filename)

    # Build the god damn thing
    ds_accessor = Deepsig_Accessor(
        target[0], target[1], 0.9, batch_size=1, throw_after_epoch=True, shuffle=True)

    print("Train count: %d" % ds_accessor.get_total_num_training_samples())
    iteration_counter = 0
    with tf.python_io.TFRecordWriter(prelim_filename + '_train.tfrecord') as writer:

        for sample in ds_accessor.get_training_generator():
            if iteration_counter % 1000 == 0:
                print("At %d" % iteration_counter)
            iteration_counter += 1
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
    print("Test count: %d" % ds_accessor.get_total_num_testing_samples())
    iteration_counter = 0
    with tf.python_io.TFRecordWriter(prelim_filename + '_test.tfrecord') as writer:
        for sample in ds_accessor.get_testing_generator():
            if iteration_counter % 100 == 0:
                print("At %d" % iteration_counter)
            iteration_counter += 1
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


with Pool(4) as p:
    p.map(convert_hdf5_to_dataset_file, subset_targets)

# convert_hdf5_to_dataset_file(subset_targets[0])
