# Experimenting with placeholders

import tensorflow as tf


with tf.Session() as sess:
    placeholder1 = tf.placeholder(tf.int32)  # Value is not filled
    placeholder2 = tf.placeholder(tf.int32)  # Value is not filled
    # Note how we're using a sort of shorthand here for the TF multiply operation
    multiply_op = placeholder1 * placeholder2 * 2

    # Feed dict says what to put into respective placeholders
    # Prints 20 == 2*5*2
    print(sess.run(multiply_op, feed_dict={placeholder1: 2, placeholder2: 5}))
