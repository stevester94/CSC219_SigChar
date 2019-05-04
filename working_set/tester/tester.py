from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import random
from time import sleep
import numpy as np
import json
from timeit import timeit
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
low_snr = [-20, -10]
medium_snr = [0, 10]
high_snr = [20, 30]
thirty_snr = [30]



def tier_test_model(target, model_dir):
    test_path = build_dataset_names(target)[1]

    test_ds = tf.data.TFRecordDataset(test_path).map(transform_to_orig)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(batch_size)

    test_iterator = test_ds.make_initializable_iterator()
    test_iter_node = test_iterator.get_next()

    # It's a little fruity, but we need to have local placeholders... for reasons...
    y = tf.placeholder(tf.int32, [None, DATASET_LEN_Y], name="tester_y_placeholder")
    total_confusion = tf.Variable(tf.zeros(
        [DATASET_LEN_Y, DATASET_LEN_Y],
        dtype=tf.dtypes.int32,
        name=None
    ))

    total_labels = tf.Variable(tf.zeros(
        [1, DATASET_LEN_Y],
        dtype=tf.dtypes.int32,
        name=None
    ))

    init_op = tf.global_variables_initializer()



    with tf.Session() as sess:
        sess.run(init_op)
        tf.saved_model.loader.load(sess, ["fuck"], model_dir)


        soft_predict = tf.get_default_graph().get_tensor_by_name("soft_predict:0")
        predictions = tf.argmax(soft_predict, 1)

        labels = tf.argmax(y, 1)

        acc, acc_op = tf.metrics.accuracy(labels=labels,
                                    predictions=predictions)

        batch_confusion = tf.math.confusion_matrix(
            labels,
            predictions,
            num_classes=DATASET_LEN_Y,
            dtype=tf.dtypes.int32,
            name=None,
            weights=None
        )

        add_labels = tf.add(total_labels, y)
        add_labels = tf.assign(total_labels, add_labels)

        cumulate_confusion = tf.add(batch_confusion, total_confusion)
        cumulate_confusion = tf.assign(total_confusion, cumulate_confusion)

        sess.run(test_iterator.initializer)

        sess.run(tf.local_variables_initializer())

        while True:
            try:
                val = sess.run([test_iter_node])

                x_test = val[0][0]
                y_test = val[0][1]

                sess.run([acc_op, cumulate_confusion, add_labels],
                        feed_dict={"x_placeholder:0": x_test, y: y_test})
            except tf.errors.OutOfRangeError:
                break

        print("Total labels: %s" % sess.run(total_labels))

        return {"confusion": sess.run(total_confusion), "accuracy": sess.run(acc)}


def plot_confusion_matrix(confusion, labels):
   # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # cax = ax.matshow(d, cmap='bone')
    cax = ax.matshow(confusion, cmap='Purples')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([""] + labels, rotation=90)
    ax.set_yticklabels([""] + labels)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # This is very hack-ish
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(which='minor')

    # Set labels
    ax.set_xlabel("Predicted Modulation")
    ax.set_ylabel("Actual Modulation")

    plt.show()

if __name__ == "__main__":
    EXPORT_DIR = "../models/"

    test_cases = []

    # case = {
    #     "label": "OTA 256 filters",
    #     "target": (all_modulation_targets, low_snr)
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256 filters",
    #     "target": (all_modulation_targets, medium_snr)
    # }
    # test_cases.append(case)

    # case = {
    #     "label": "OTA 256 filters",
    #     "target": (all_modulation_targets, high_snr)
    # }
    # test_cases.append(case)

    case = {
        "label": "OTA 256 filters",
        "target": (all_modulation_targets, high_snr)
    }
    test_cases.append(case)

    results = None
    for case in test_cases:
        print("Running the following: ")
        print(case)
        print(EXPORT_DIR+case["label"])
        results = tier_test_model(case["target"], EXPORT_DIR+case["label"])

        print("Accuracy: %s" % results["accuracy"])
        
        plot_confusion_matrix(results["confusion"], case["target"][0])
