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
high_snr = [24, 26, 28, 30]

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
subset_targets.append((all_modulation_targets, all_snr_targets))


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


elem = None
def verify_dataset(target):
    global elem
    mod_names = '_'.join(mod for mod in target[0])
    snr_names = '_'.join(str(snr) for snr in target[1])
    prelim_filename = BASE_DIR + mod_names + snr_names
    # print(prelim_filename)

    # Build the god damn thing
    ds_accessor = Deepsig_Accessor(
        target[0], target[1], 0.9, batch_size=1, throw_after_epoch=True, shuffle=True)

    train_ds = tf.data.TFRecordDataset(prelim_filename+"_train.tfrecord").map(transform_to_orig)
    iter = train_ds.make_one_shot_iterator()
    
    tf_sum = None
    try:
        for i in iter:

            if tf_sum == None:
                tf_sum = i[1]
            else:
                tf_sum = tf.add(tf_sum, i[1])
    except StopIteration:
        pass
    
    if not np.array_equal(tf_sum.numpy(), ds_accessor.cksum_train_labels()):
        print(prelim_filename + "_train does not match")
    else:
        print("OK")
    


    #
    # Now do test
    # 
    test_ds = tf.data.TFRecordDataset(
        prelim_filename+"_test.tfrecord").map(transform_to_orig)
    iter = test_ds.make_one_shot_iterator()

    tf_sum = None
    try:
        for i in iter:

            if tf_sum == None:
                tf_sum = i[1]
            else:
                tf_sum = tf.add(tf_sum, i[1])
    except StopIteration:
        pass

    if not np.array_equal(tf_sum.numpy(),ds_accessor.cksum_test_labels()):
        print(prelim_filename + "_test does not match")
    else:
        print("OK")

with Pool(2) as p:
    p.map(verify_dataset, subset_targets)
