#! /bin/python
# Using https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
import tensorflow as tf
import deepsig_accessor as ds_accessor
import random

# Seed our randomizer for reproducability
random.seed(1337)

# Some info about our data
LEN_X = 2048
LEN_Y = 24

# Build our training and test data
TRAIN_TEST_RATIO = 0.75 # X% of the dataset will be for training data
train_x = []
train_y = []
test_x  = []
test_y  = []

dataset = ds_accessor.get_data_samples(["32PSK", "32QAM"], [30])
random.shuffle(dataset)

set_split_point = int(len(dataset)*0.75)

for i in range(0, set_split_point):
    train_x.append(dataset[i][0])
    train_y.append(dataset[i][2])

for i in range(set_split_point, len(dataset)):
    test_x.append(dataset[i][0])
    test_x.append(dataset[i][2])

# Python optimisation variables
learning_rate = 0.5
epochs = 1000

# declare the training data placeholders
# Input 10 nodes, output 10 nodes
x = tf.placeholder(tf.float32, [None, 2048])
y = tf.placeholder(tf.float32, [None, 24])

NUM_HIDDEN_NODES = 100

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([LEN_X, NUM_HIDDEN_NODES], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([NUM_HIDDEN_NODES]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([NUM_HIDDEN_NODES, LEN_Y], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([LEN_Y]), name='b2')

# calculate the output of the hidden layer

# Hidden layer: Multiply by the weights, add bias
hidden_out = tf.add(tf.matmul(x, W1), b1)

# Next, we finalise the hidden_out operation by applying a rectified linear unit activation function to the matrix multiplication plus bias.
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

#converting the output y_ to a clipped version, limited between 1e-10 to 0.999999.  
# This is to make sure that we never get a case were we have a log(0) operation occurring during training 
# – this would return NaN and break the training process.
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)


cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                              + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)


# finally setup the initialisation operator
init_op=tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction=tf.equal(
    tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy=tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)

   # Not doing batches, do the whole training data each epoch
   for epoch in range(epochs):
    _, c = sess.run([optimiser, cross_entropy],
                    feed_dict={x: train_x, y: train_y})
    
    print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))


   print(sess.run(accuracy, feed_dict={
         x: train_x, y: train_y}))

   print("Testing: " + str(train_x[0]))
   print(sess.run(y_, feed_dict={
       x: [train_x[0]] }))

   print(sess.run(W1))
