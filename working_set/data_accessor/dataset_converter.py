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

IMPORT_DIR =  "../../data_exploration/datasets/"
EXPORT_DIR = "../../data_exploration/offenders/"

all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]

limited_snr = [-20, -10, 0, 10, 20, 30]
high_snr = [24, 26, 28, 30]

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

    return X, Y


class_lookup = {
     '32PSK': 0,
     '16APSK': 1,
     '32QAM': 2,
     'FM': 3,
     'GMSK': 4,
     '32APSK': 5,
     'OQPSK': 6,
     '8ASK': 7,
     'BPSK': 8,
     '8PSK': 9,
     'AM-SSB-SC': 10,
     '4ASK': 11,
     '16PSK': 12,
     '64APSK': 13,
     '128QAM': 14,
     '128APSK': 15,
     'AM-DSB-SC': 16,
     'AM-SSB-WC': 17,
     '64QAM': 18,
     'QPSK': 19,
     '256QAM': 20,
     'AM-DSB-WC': 21,
     'OOK': 22,
     '16QAM': 23
}

target_indices = [class_lookup["64QAM"], class_lookup["AM-SSB-WC"]]
def predicate(X, Y):
    label = tf.argmax(Y, 0)

    if label in target_indices:
        return True
    else:
        return True


TRAIN_OUT_NAME = "AM-SSB-WC_64QAM0_2_4_6_8_10_12_14_16_18_20_22_24_26_28_30_-20_-18_-16_-14_-12_-10_-8_-6_-4_-2_train.tfrecord"
TEST_OUT_NAME  = "AM-SSB-WC_64QAM0_2_4_6_8_10_12_14_16_18_20_22_24_26_28_30_-20_-18_-16_-14_-12_-10_-8_-6_-4_-2_test.tfrecord"

def transform_dataset(target):
    mod_names = '_'.join(mod for mod in target[0])
    snr_names = '_'.join(str(snr) for snr in target[1])
    prelim_filename = IMPORT_DIR + mod_names + snr_names
    # print(prelim_filename)



    train_ds = tf.data.TFRecordDataset(prelim_filename+"_train.tfrecord").map(transform_to_orig)
    iter = train_ds.make_one_shot_iterator()

    counter = 0
    with tf.python_io.TFRecordWriter(prelim_filename + '_test.tfrecord') as writer:
        try:
            for i in iter:
                print(counter)
                if predicate(i[0], i[1]):

                counter += 1



    except StopIteration:
        pass

transform_dataset((all_modulation_targets, all_snr_targets))



