from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
# Import MNIST data
from deepsig_accessor import Deepsig_Accessor
import random
from time import sleep
import numpy as np

""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

tf.logging.set_verbosity(tf.logging.INFO)

#################################
# Begin Build training set
#################################

# Seed our randomizer for reproducability
random.seed(1337)

all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                          '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                           "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]

toy_targets = ['32QAM', 'FM']

all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]


modulation_targets = all_modulation_targets
snr_targets = [6]

batch_size = 100
train_test_ratio = 0.75

ds_accessor = Deepsig_Accessor(modulation_targets, snr_targets, batch_size, train_test_ratio, throw_after_epoch=True, shuffle=False)

print("Num training elements: %d" % ds_accessor.get_total_num_training_samples())

# Some info about our data
DATASET_LEN_X = 2048
DATASET_LEN_Y = 24

#############################
# End Build training set
#############################

######################
# Training Parameters
######################
learning_rate = 0.001
num_train_epochs = 20

#############################
# Define the neural network #
#############################
default_conv_settings = {"conv_num_filters": 64, "conv_kernel_size": 3, "conv_activation": tf.nn.relu, "max_pool_stride":2, "max_pool_kernel_size":2}
default_fc_settings = {"fc_num_nodes": 128, "fc_activation": tf.nn.relu}

network_conv_settings = []
for i in range(0,7): network_conv_settings.append(default_conv_settings)

network_fc_settings   = []
for i in range(0,2): network_fc_settings.append(default_fc_settings)


# Create the neural network
# There is a single implicit FC layer in addition to what is specified for fc_settings
# conv settings: [{"conv_num_filters", "conv_kernel_size", "conv_activation", max_pool_stride", "max_pool_kernel_size"}...]
#                                                           IE tf.nn.relu
# fc_settings: [{"fc_num_nodes", "fc_activation"}...]
def build_CNN_node(features, conv_settings, fc_settings, num_classes, reuse):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # TF Estimator input is a dict, in case of multiple inputs

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        layer = tf.reshape(features, shape=[-1, DATASET_LEN_X, 1])

        for conv in conv_settings:
            # # Convolution Layer with 32 filters and a kernel size of 5
            layer = tf.layers.conv1d(layer, conv["conv_num_filters"], conv["conv_kernel_size"], activation=conv["conv_activation"])
            # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            layer = tf.layers.max_pooling1d(layer, conv["max_pool_stride"], conv["max_pool_kernel_size"])

        # # Flatten the data to a 1-D vector for the fully connected layer
        layer = tf.contrib.layers.flatten(layer)

        # Output layer, class prediction
        for fc in fc_settings:
            layer = tf.layers.dense(
                layer, fc["fc_num_nodes"], activation=fc["fc_activation"])

        layer = tf.layers.dense(layer, num_classes)
    return layer


# Define the model function (following TF Estimator Template)
def build_model(features_placeholder, labels_placeholder, conv_settings, fc_settings, num_classes):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    
    logits_node = build_CNN_node(features_placeholder, conv_settings, fc_settings, num_classes, reuse=False)
    
    loss_node = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_node, labels=labels_placeholder)
    loss_node = tf.reduce_mean(loss_node)


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_node = optimizer.minimize(loss_node,
                                  global_step=tf.train.get_global_step())

    return logits_node, train_node, loss_node


x = tf.placeholder(tf.float32, [None, DATASET_LEN_X])
y = tf.placeholder(tf.float32, [None, DATASET_LEN_Y])

logits_node, train_node, loss_node = build_model(x, y, network_conv_settings, network_fc_settings, DATASET_LEN_Y)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()




with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    for epoch in range(num_train_epochs):
        try:
            while True:  # I realize now this is a bad anti-pattern, oh well
                train_batch = ds_accessor.get_next_train_batch()
                _, c = sess.run([train_node, loss_node],
                                feed_dict={x: train_batch[0], y: train_batch[1]})
        except StopIteration:
            pass
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))

    ##############
    # Test model
    ##############

    # Softmax the output
    pred = tf.nn.softmax(logits_node)

    # For each categorical output, see if they're equal and return a tensor
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Cast equality vector to float, take the mean of the whole thing
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Run the shit on our test sets
    test_batch = ds_accessor.get_next_test_batch()
    print("Accuracy:", accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1]}))
