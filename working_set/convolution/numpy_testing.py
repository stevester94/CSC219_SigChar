from __future__ import division, print_function, absolute_import

import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import deepsig_accessor as ds_accessor
import random
from time import sleep
import numpy as np



#################################
# Begin Build training set
#################################

# Seed our randomizer for reproducability
random.seed(1337)
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]


all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]


modulation_targets = subset_modulation_targets
snr_targets = [30]


dataset = ds_accessor.get_data_samples(modulation_targets, snr_targets)
random.shuffle(dataset)

train_x = []
train_y = []
test_x = []
test_y = []
set_split_point = int(len(dataset)*0.75)

for i in range(0, set_split_point):
    train_x.append(dataset[i][0])
    train_y.append(dataset[i][2])

for i in range(set_split_point, len(dataset)):
    test_x.append(dataset[i][0])
    test_y.append(dataset[i][2])

train_x = np.array(train_x)
train_y = np.array(train_x)
test_x = np.array(test_x)
test_y = np.array(test_y)

#############################
# End Build training set
#############################

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_x, y=train_y,
    batch_size=batch_size, num_epochs=None, shuffle=True)

