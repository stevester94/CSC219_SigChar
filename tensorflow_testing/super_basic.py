import tensorflow as tf

# Alright so we are building a graph of operations
# The graph will look like this

# (multiply) -> (add) -> (assign result)
with tf.Session() as sess:
    # Declare our variables and constants
    var = tf.Variable(420)
    result = tf.Variable(0)
    two_const  = tf.constant(2)
    five_const = tf.constant(5)

    # This node is multiplying by our 2 constant
    multiply_op = tf.multiply(var, two_const)

    # This is adding to the result of the multiply op. The multiply operation 'flows' into this one
    add_op = tf.add(five_const, multiply_op)

    # This node is assigning the output of the previous node to the variable blaze_var
    assign_op = tf.assign(result, add_op)

    # Variable initialization
    init_op = tf.global_variables_initializer()
    sess.run(init_op)


    # Now, if we run a node which is in the 'middle' of the chain...
    sess.run(add_op)

    # It will print 0, because we don't execute the entire chain, only that node and its dependent nodes
    print(sess.run(result)) # Prints 0, because the assign node was not executed


    # We run that node, which in turn runs all dependent nodes
    sess.run(assign_op)
    print(sess.run(result)) # prints 845 == 420*2+5
    print(sess.run(var)) # Prints 420, because we did not assign to it

    # Now, lets make the graph a little more complicated by back feeding the result of the add_op into var
    backfeed_assign_op = tf.assign(var, add_op)
    sess.run(backfeed_assign_op)
    print("Backfed var: %d" % sess.run(var)) # Prints 845, because we assigned the result to var
    # Notice how we don't use any intermediate variables. The nodes are flowing directly into each other

    # Now that we are backfeeding, if we run the backfeed assign op again a few times, it will keep getting updated
    for _ in range(5):
        sess.run(backfeed_assign_op)
    
    print("Repeated backfed var: %d" % sess.run(var)) # Prints a big number, because we backfed a few times