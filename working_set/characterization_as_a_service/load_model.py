from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import random
from time import sleep
import numpy as np
import json
from timeit import timeit
import sys

EXPORT_DIR = "/tmp/fuckin_save/"

# Some info about our data
DATASET_LEN_X = 2048
DATASET_LEN_Y = 24

BASE_DIR = "../../data_exploration/datasets/"

batch_size = 1

##########################
# Configure our datasets
##########################
def build_dataset_names(target):
    mod_names = '_'.join(mod for mod in target[0])
    snr_names = '_'.join(str(snr) for snr in target[1])
    prelim_filename = BASE_DIR + mod_names + snr_names

    print("Using %s" % prelim_filename)

    return prelim_filename+"_train.tfrecord", prelim_filename+"_test.tfrecord"

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


#############################
# Target dataset parameters
#############################
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
thirty_snr = [30]

target = (subset_modulation_targets, thirty_snr)

train_path = build_dataset_names(target)[0]
test_path = build_dataset_names(target)[1]

train_ds = tf.data.TFRecordDataset(train_path).map(transform_to_orig)
train_ds = train_ds.shuffle(batch_size*50)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(batch_size)

test_ds = tf.data.TFRecordDataset(test_path).map(transform_to_orig)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(batch_size)

train_iterator = train_ds.make_initializable_iterator()
train_iter_node = train_iterator.get_next()

test_iterator = test_ds.make_initializable_iterator()
test_iter_node = test_iterator.get_next()

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["fuck"], EXPORT_DIR)

    graph = tf.get_default_graph()
    # print(graph.get_operations())

    sess.run(test_iterator.initializer)

    val = sess.run([test_iter_node])

    x_test = val[0][0]
    y_test = val[0][1]

    # soft_predict = tf.get_default_graph().get_operation_by_name("soft_predict")
    soft_predict = tf.get_default_graph().get_tensor_by_name("soft_predict:0")
    predictions = tf.argmax(tf.nn.softmax(soft_predict), 1)

    print("Trying to run")
    print(sess.run(predictions, feed_dict={"x_placeholder:0": x_test}))
    print("Actual: %s" % str(y_test))
    print("Done")
