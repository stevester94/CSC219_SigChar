#! /bin/python
# Using https://adventuresinmachinelearning.com/python-tensorflow-tutorial/
import tensorflow as tf

# Lets try to train the identity function!

# Build our training and test data
train_x = []
train_y = []

for i in range(10):
    row = [0] * 10
    row[i] = 1
    train_x.append(row)
    train_y.append(row)

# Python optimisation variables
learning_rate = 0.5
epochs = 1000

# declare the training data placeholders
# Input 10 nodes, output 10 nodes
x = tf.placeholder(tf.float32, [None, 10])
y = tf.placeholder(tf.float32, [None, 10])

NUM_HIDDEN_NODES = 10

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([10, NUM_HIDDEN_NODES], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([NUM_HIDDEN_NODES]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([NUM_HIDDEN_NODES, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

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


# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# start the session
with tf.Session() as sess:
    # initialise the variables
    saver.restore(sess, "tf_vars/model_save.ckpt")
    print("Model restored.")

    print("Testing: " + str(train_x[0]))
    print(sess.run(y_clipped, feed_dict={
            x: [train_x[0]] }))


