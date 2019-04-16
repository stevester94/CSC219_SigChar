from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
# Import MNIST data
import deepsig_accessor as ds_accessor
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


all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
                   26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]


modulation_targets = subset_modulation_targets
snr_targets = [30]


dataset = ds_accessor.get_data_samples(modulation_targets, snr_targets)
# random.shuffle(dataset)

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
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

#############################
# End Build training set
#############################

# Training Parameters
learning_rate = 0.001
num_steps = 10
# batch_size = 128
batch_size = 1

# Network Parameters
num_classes = len(train_y[0])
dropout = 0.25  # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        # x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # # Convolution Layer with 32 filters and a kernel size of 5
        # conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # # Convolution Layer with 64 filters and a kernel size of 3
        # conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        # conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # # Flatten the data to a 1-D vector for the fully connected layer
        # fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        # fc1 = tf.layers.dense(x, 24) # Fucked with this one
        # Apply Dropout (if is_training is False, dropout is not applied)
        # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction

        print("Steven - num_classes: " + str(n_classes))
        print("Steven - X: %s" % str(x))
        out = tf.Print(x, [x], "Input: ")
        out = tf.layers.dense(out, 2048, activation=tf.nn.relu)
        out = tf.layers.dense(out, 2048, activation=tf.nn.relu)
        # out = tf.layers.dense(out, 2048, activation=tf.nn.relu)
        out = tf.layers.dense(out, n_classes)
        out = tf.Print(out, [out], "Output: ", summarize=100)
    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    # pred_classes = tf.argmax(logits_test, axis=1)
    pred_classes = logits_test
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))

    # print("Rank of logits: %s" % str(tf.shape(logits_train)))
    # print("Rank of labels: %s" % str(labels))
    # It's gotta be this shit

    # loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))

    printed_labels = tf.Print(labels, [labels], "Labels: ", summarize=100)

    loss_op = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_train, labels=printed_labels)
    loss_op = tf.Print(loss_op, [loss_op], "Cross Entropy: ")

    loss_op = tf.reduce_mean(loss_op)
    loss_op = tf.Print(loss_op, [loss_op], "Reduce Mean: ")



    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

def my_fucking_feeder():
    return tf.constant(train_x[0:1]), tf.constant(train_y[0:1])
    # return tf.constant([train_x[0]]), tf.constant([train_y[0]])
    # return train_x[0:100], tf.constant(1, shape=[100])
    # return {"kek": train_x[0:100]}, range(0,100)
    # return {'images': mnist.train.images[0:100]}, mnist.train.labels[0:100]

# Define the input function for training
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x=train_x, y=train_y,
#     batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model
print("Training!")
model.train(my_fucking_feeder, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'images': mnist.test.images}, y=mnist.test.labels,
#     batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
# print("Evaluating!")
# e = model.evaluate(my_fucking_feeder, steps=num_steps)

# print("Testing Accuracy:", e['accuracy'])

feeder_called = False
def another_feeder():
    global feeder_called
    if not feeder_called:
        feeder_called = True
        return train_x[0:100], train_y[0:100]
    else:
        raise StopIteration


output = tf.constant([[225.725769,
  -22.1992931,
  -29.8155403,
  -23.4150391,
  -24.1813965,
  -36.8599548,
  -38.964447,
  -38.1486969,
  -43.3182373,
  -32.6156807,
  -32.9504051,
  -39.4783707,
  -11.6777391,
  -24.5229359,
  -31.0119743,
  -10.3592606,
  -40.6214256,
  -36.9803429,
  -46.4612923,
  -38.5201645,
  -41.3779602,
  -18.8825436,
  -45.1395454,
  -57.8695526]])

labels = tf.constant([[1,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0,
                       0]])

with tf.Session() as sess:
    # Prints 0!
    sm = tf.nn.softmax_cross_entropy_with_logits(
        logits=output, labels=labels)
    sm = tf.Print(sm, [sm], "Test softmax: ", summarize=100)
    sess.run(sm)

# p = model.predict(another_feeder)

