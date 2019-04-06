# Experimenting with placeholders

import tensorflow as tf



with tf.Session() as sess:
    placeholder1 = tf.placeholder(tf.int32) # Value is not filled
    placeholder2 = tf.placeholder(tf.int32) # Value is not filled
    multiply_op = placeholder1 * placeholder2 * 2 # Note how we're using a sort of shorthand here for the TF multiply operation

    # Feed dict says what to put into respective placeholders
    print(sess.run(multiply_op, feed_dict={placeholder1: 2, placeholder2:5})) # Prints 20 == 2*5*2