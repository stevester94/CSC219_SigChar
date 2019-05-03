from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import random
from time import sleep
import numpy as np
import json
from timeit import timeit
import sys
import socket
import struct

# Some info about our data
DATASET_LEN_X = 2048
DATASET_LEN_Y = 24

BASE_DIR = "../../data_exploration/datasets/"
MODEL_DIR = "../models/fuckin_save"

######################
# Server parameters 
######################
BIND_PORT = 1337
BIND_ADDRESS = "127.0.0.1"
BUFFER_SIZE = 1024
SIZE_OF_FLOAT = 4

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
recv_sock.bind((BIND_ADDRESS, BIND_PORT))
recv_sock.listen(1)

# Returns the parsed data


def receive_helper(recv_sock):
    data = b''
    conn,_ = recv_sock.accept()
    while len(data) < DATASET_LEN_X * SIZE_OF_FLOAT:
        rec = conn.recv(BUFFER_SIZE)
        if not rec:
            print("No data received")
            break
        else:
            data += rec
    try:
        ret_array = struct.unpack("%sf"%DATASET_LEN_X, data)
    except:
        print("Error unpacking, got %d bytes" % len(data))
        sys.exit(1)

    return ret_array, conn


def sender_helper(send_sock, classification, confidence):
    payload = struct.pack("If", classification, confidence)
    send_sock.send(payload)
    send_sock.close()



with tf.Session() as sess:
    tf.saved_model.loader.load(sess, ["fuck"], MODEL_DIR)

    graph = tf.get_default_graph()
    # print(graph.get_operations())

    print("CAAS ready for service")
    while True:
        X, recv_con = receive_helper(recv_sock)

        print(X)


        soft_predict = tf.get_default_graph().get_tensor_by_name("soft_predict:0")
        predictions = tf.argmax(tf.nn.softmax(soft_predict), 1)

        
        confidence, classification = sess.run([soft_predict, predictions], feed_dict={"x_placeholder:0": [X]})

        classification = classification[0]
        confidence = confidence[0][classification]

        print("Classification: %s, Confidence: %s" % (classification, confidence))

        sender_helper(recv_con, classification, confidence)
