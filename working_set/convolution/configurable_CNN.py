from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import random
from time import sleep
import numpy as np
import json
from timeit import timeit
import sys

tf.logging.set_verbosity(tf.logging.INFO)

# Seed our randomizer for reproducability
random.seed(1337)


# Some info about our data
DATASET_LEN_X = 2048
DATASET_LEN_Y = 24

BASE_DIR = "../../data_exploration/datasets/"
EXPORT_DIR_BASE = "../models/"


def train_and_test(_sentinel_= None, learning_rate=None,
                   num_train_epochs=None,
                   batch_size=None,
                   target=None,
                   network_conv_settings=None,
                   network_fc_settings=None,
                   label=None):
    if _sentinel_ != None:
        print("Can only use kwargs!")
        return
    
    if learning_rate == None or num_train_epochs == None or batch_size == None or target == None or network_conv_settings == None or network_fc_settings == None:
        print("You're missing an arg!")
        return

    ##########################
    # Configure our datasets
    ##########################
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


    train_path = build_dataset_names(target)[0]
    test_path = build_dataset_names(target)[1]

    train_ds = tf.data.TFRecordDataset(train_path).map(transform_to_orig)
    train_ds = train_ds.shuffle(batch_size*50)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(batch_size)

    test_ds = tf.data.TFRecordDataset(test_path).map(transform_to_orig)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(batch_size)

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
                layer = tf.layers.conv1d(
                    layer, conv["conv_num_filters"], conv["conv_kernel_size"], activation=tf.nn.relu)
                # # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
                layer = tf.layers.max_pooling1d(layer, conv["max_pool_stride"], conv["max_pool_kernel_size"])

            # # Flatten the data to a 1-D vector for the fully connected layer
            layer = tf.contrib.layers.flatten(layer)

            # Output layer, class prediction
            for fc in fc_settings:
                layer = tf.layers.dense(
                    layer, fc["fc_num_nodes"], activation=tf.nn.relu)

            layer = tf.layers.dense(layer, num_classes, name="cnn_logits_out")
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


    x = tf.placeholder(tf.float32, [None, DATASET_LEN_X], name="x_placeholder")
    y = tf.placeholder(tf.float32, [None, DATASET_LEN_Y], name="y_placeholder")

    logits_node, train_node, loss_node = build_model(x, y, network_conv_settings, network_fc_settings, DATASET_LEN_Y)

    # This is a kludge becuase I don't know how to access the the logits directly
    # Used only  for transfer learning (IE, when you import this op is how you exercise the model)
    soft_prediction = tf.nn.softmax(logits_node, name="soft_predict")

    total_confusion = tf.Variable(tf.zeros(
        [DATASET_LEN_Y, DATASET_LEN_Y],
        dtype=tf.dtypes.int32,
        name=None
    ))

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()


    train_iterator = train_ds.make_initializable_iterator()
    train_iter_node = train_iterator.get_next()

    test_iterator = test_ds.make_initializable_iterator()
    test_iter_node = test_iterator.get_next()

    # We're gonna save the entire model for later use
    EXPORT_DIR = EXPORT_DIR_BASE + label
    print("Will export to %s" % EXPORT_DIR)
    builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)

    with tf.Session() as sess:
        # initialize the variables
        sess.run(init_op)

        for epoch in range(num_train_epochs):
            sess.run(train_iterator.initializer)
            while True:
                try:
                    val = sess.run([train_iter_node])

                    x_train = val[0][0]
                    y_train = val[0][1]
                    _, c = sess.run([train_node, loss_node],
                                    feed_dict={x: x_train, y: y_train})
                except tf.errors.OutOfRangeError:
                    break

            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(c))

        # Done training
        builder.add_meta_graph_and_variables(
            sess,
            ["fuck"], # Not sure on this one
            signature_def_map=None,
            assets_collection=None,
            legacy_init_op=None,
            clear_devices=False,
            main_op=None,
            strip_default_attrs=False,
            saver=None
        )

        builder.save()

        ##############
        # Test model
        ##############

        labels = tf.argmax(y, 1)
        predictions = tf.argmax(tf.nn.softmax(logits_node), 1)

        acc, acc_op = tf.metrics.accuracy(labels=labels,
                                        predictions=predictions)

        batch_confusion = tf.math.confusion_matrix(
            labels,
            predictions,
            num_classes=DATASET_LEN_Y,
            dtype=tf.dtypes.int32,
            name=None,
            weights=None
        )



        cumulate_confusion = tf.add(batch_confusion, total_confusion)
        cumulate_confusion = tf.assign(total_confusion, cumulate_confusion)

        sess.run(test_iterator.initializer)

        sess.run(tf.local_variables_initializer())

        while True:
            try:
                val = sess.run([test_iter_node])

                x_test = val[0][0]
                y_test = val[0][1]

                sess.run([acc_op, cumulate_confusion],
                                feed_dict={x: x_test, y: y_test})
            except tf.errors.OutOfRangeError:
                break

        print("Final accuracy: %f" % sess.run(acc))
        print(sess.run(total_confusion))
        return sess.run(acc)





#############################################################
# DO THE THING, this is ran from CLI now, via test_runner
#############################################################
def train_test_time(parameters_dict):
    accuracy = None

    def wrapper():
        nonlocal accuracy
        accuracy = train_and_test(
            learning_rate = parameters_dict["learning_rate"],
            num_train_epochs = parameters_dict["num_train_epochs"],
            batch_size = parameters_dict["batch_size"],
            target = parameters_dict["target"],
            network_conv_settings =  parameters_dict["network_conv_settings"],
            network_fc_settings = parameters_dict["network_fc_settings"],
            label=parameters_dict["label"])

    time = timeit(wrapper, number=1)

    parameters_dict["accuracy"] = str(accuracy)
    parameters_dict["time"] = str(time)

    return parameters_dict

if __name__ == "__main__":
    str_case = sys.argv[1]

    case = json.loads(str_case)

    print("Attempting to run %s" % case["label"])

    results = train_test_time(case)

    with open("test_results.json", "a") as f:
        f.write(json.dumps(results))
        f.write("\n\n")

