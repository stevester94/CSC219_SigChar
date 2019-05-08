from __future__ import division, print_function, absolute_import

import tensorflow as tf
# Import MNIST data
import random
from time import sleep
import numpy as np
import struct
import socket
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

tf.enable_eager_execution()

all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

offenders = ["64QAM", "AM-SSB-WC"]

all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]
limited_snr = [-20, -10, 0, 10, 20, 30]
low_snr = [-20, -10]
medium_snr = [0, 10]
high_snr = [20, 30]

BUFFER_SIZE = 20


DATASET_LEN_X = 2048
DATASET_LEN_Y = 24


BASE_DIR = "../../data_exploration/datasets/"


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

    # Accuracy subplot
    # Calc the correct cases
    correct = 0
    total = 0
    for i in range(0, len(confusion[0])):
        correct += confusion[i][i]
    total = np.sum(confusion)
    ax.text(0, 30, "Accuracy: %f" % (correct/total))

    plt.show()

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

def call_CAAS(data):
    buf = struct.pack('%sf' % len(data), *data)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", 1337))

    # print("Sum of payload: %f" % sum(data))

    sock.sendall(buf)

    response = sock.recv(BUFFER_SIZE)
    sock.close()

    classification, confidence = struct.unpack("If", response)
    
    return {"classification":classification, "confidence":confidence}


target = (all_modulation_targets, limited_snr)

test_path = build_dataset_names(target)[1]
test_ds = tf.data.TFRecordDataset(test_path).map(transform_to_orig)
test_ds = test_ds.batch(1)
test_ds = test_ds.prefetch(10)

# Indexing is [actual][predicted]
confusion = np.zeros((DATASET_LEN_Y, DATASET_LEN_Y), np.int64)
total_samps = 0
correct = 0

print(confusion)

for samp in test_ds:
    IQ = samp[0][0].numpy()
    label = samp[1][0].numpy()
    
    results = call_CAAS(IQ)

    actual = np.argmax(label)

    confusion[actual][results["classification"]] += 1
    if actual == results["classification"]: correct += 1

    total_samps += 1

    print("Done with %s" % total_samps)


print("Accuracy: %s" % (correct/total_samps))
print("Confusion: %s" % confusion)

plot_confusion_matrix(confusion, all_modulation_targets)