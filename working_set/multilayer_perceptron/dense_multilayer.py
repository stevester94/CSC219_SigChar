#! /bin/python
# Using https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
import tensorflow as tf
import deepsig_accessor as ds_accessor
import random
from time import sleep

# Seed our randomizer for reproducability
random.seed(1337)

# Some info about our data
LEN_X = 2048
LEN_Y = 24

# Build our training and test data
TRAIN_TEST_RATIO = 0.75 # X% of the dataset will be for training data
train_x = [] # numpy array of IQ
train_y = [] # One hot
test_x  = []
test_y  = []


all_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
                      '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']
subset_modulation_targets = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK']

easy_modulation_targets = ["OOK", "4ASK", "BPSK", "QPSK",
                             "8PSK", "16QAM", "AM-SSB-SC", "AM-DSB-SC", "FM", "GMSK", "OQPSK"]


all_snr_targets = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
               26, 28, 30, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2]


modulation_targets = subset_modulation_targets
snr_targets = [30]


dataset = ds_accessor.get_data_samples(modulation_targets, snr_targets)
random.shuffle(dataset)

set_split_point = int(len(dataset)*0.75)

for i in range(0, set_split_point):
    train_x.append(dataset[i][0])
    train_y.append(dataset[i][2])

for i in range(set_split_point, len(dataset)):
    test_x.append(dataset[i][0])
    test_y.append(dataset[i][2])

# Python optimisation variables
learning_rate = 0.001 # Orignally 0.001 for Adam
epochs = 100

# Notes on identity function:
# 10 works well for 1 hidden, not for 3
NUM_HIDDEN_NODES = 2048


# Code to hijack the shit and just train an identity function
# if False:
#     print("Hijacking for identity testing!")
#     sleep(5)
#     LEN_X = 10
#     LEN_Y = LEN_X
#     train_x = []
#     train_y = []
#     for i in range(LEN_X):
#         row = [0] * LEN_X
#         row[i] = 1
#         train_x.append(row)
#         train_y.append(row)
#     test_x = train_x
#     test_y = train_y

# declare the training data placeholders
# Input 10 nodes, output 10 nodes
x = tf.placeholder(tf.float32, [None, LEN_X])
y = tf.placeholder(tf.float32, [None, LEN_Y])



print_op = tf.print("Shape of input: ", tf.shape(x))

hidden_out = tf.layers.dense(x, NUM_HIDDEN_NODES, activation=tf.nn.relu)
hidden_out = tf.layers.dense(hidden_out, NUM_HIDDEN_NODES, activation=tf.nn.relu)

with tf.control_dependencies([print_op]):
    out_layer = tf.layers.dense(hidden_out, LEN_Y)

# calculate the output of the hidden layer

#  Hidden layer: Multiply input by the weights, add bias, rectify
# hidden_out = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
# hidden_out = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))
# hidden_out = tf.nn.relu(tf.add(tf.matmul(hidden_out, W3), b3))
# out_layer = tf.add(tf.matmul(hidden_out, W4), b4)


# No clue why, but the magic comes from this, was originally using some crazy ass logarithm and clipping
print2_op = tf.print("Shape of output: ", tf.shape(out_layer))

with tf.control_dependencies([print2_op]):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=out_layer, labels=y))

# add an optimiser
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)





# finally setup the initialisation operator
init_op=tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    # Not doing batches, do the whole training data each epoch
    for epoch in range(epochs):
        _, c = sess.run([train_op, loss_op],
                feed_dict={x: train_x, y: train_y})

        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))

    ##############
    # Test model
    ##############

    # Softmax the output

    pred = tf.nn.softmax(out_layer)

    # For each categorical output, see if they're equal and return a tensor
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    # Cast equality vector to float, take the mean of the whole thing
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Run the shit on our test sets
    print("Accuracy:", accuracy.eval(feed_dict={x: test_x, y: test_y}))
