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
GENERAL_MODEL_DIR = "../models/OTA 256 filters"
# SEPARATOR_MODEL_DIR = "../models/OTA 256 filters"
SEPARATOR_MODEL_DIR = "../models/SEPARATOR OTA 256 filters, 2+1 hidden of 128"

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

######################
# Dataset parameters
######################
class_map_onehot = {
    '32PSK':         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    '16APSK':        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    '32QAM':         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
    'FM':            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
    'GMSK':          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
    '32APSK':        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
    'OQPSK':         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
    '8ASK':          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
    'BPSK':          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
    '8PSK':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
    'AM-SSB-SC':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10
    '4ASK':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 11
    '16PSK':         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
    '64APSK':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
    '128QAM':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
    '128APSK':       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
    'AM-DSB-SC':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 16
    'AM-SSB-WC':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 17
    '64QAM':         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 18
    'QPSK':          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 19
    '256QAM':        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 20
    'AM-DSB-WC':     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 21
    'OOK':           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 22
    '16QAM':         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 23
}

onehot_class_lookup = {
    0: '32PSK',
    1: '16APSK',
    2: '32QAM',
    3: 'FM',
    4: 'GMSK',
    5: '32APSK',
    6: 'OQPSK',
    7: '8ASK',
    8: 'BPSK',
    9: '8PSK',
    10: 'AM-SSB-SC',
    11: '4ASK',
    12: '16PSK',
    13: '64APSK',
    14: '128QAM',
    15: '128APSK',
    16: 'AM-DSB-SC',
    17: 'AM-SSB-WC',
    18: '64QAM',
    19: 'QPSK',
    20: '256QAM',
    21: 'AM-DSB-WC',
    22: 'OOK',
    23: '16QAM'
}

##################
# Special cases
##################

# This is a bit of a one off, but these modulations have a special
# 'separator' model trained specifically for the offenders because they
# are often mis-predicted together
offender_list = ['64QAM', 'AM-SSB-WC']


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

    tf.saved_model.loader.load(sess, ["fuck"], GENERAL_MODEL_DIR, import_scope="GENERAL")
    tf.saved_model.loader.load(sess, ["fuck"], SEPARATOR_MODEL_DIR, import_scope="SEPARATOR")

    # print("Default ops")
    # print(tf.get_default_graph().get_operations())

    general_soft_predict = tf.get_default_graph().get_tensor_by_name("GENERAL/soft_predict:0")
    general_predictions = tf.argmax(tf.nn.softmax(general_soft_predict), 1)

    separator_soft_predict = tf.get_default_graph().get_tensor_by_name("SEPARATOR/soft_predict:0")
    separator_predictions = tf.argmax(tf.nn.softmax(separator_soft_predict), 1)

    print("CAAS ready for service")
    while True:
        X, recv_con = receive_helper(recv_sock)

        confidence, classification = sess.run([general_soft_predict, general_predictions], feed_dict={"GENERAL/x_placeholder:0": [X]})

        # We only handle one at a time, so we access the first and only element of the prediction
        classification = classification[0] 

        classification_str = onehot_class_lookup[classification]
        confidence = confidence[0][classification] # Index to softmax of our prediction

        # if classification_str in offender_list:
        #     print("Offender predicted (%s), using separator model" % classification_str)
        #     confidence, classification = sess.run([separator_soft_predict, separator_predictions], feed_dict={"SEPARATOR/x_placeholder:0": [X]})

        #     classification = classification[0]
        #     confidence = confidence[0][classification]  # Index to softmax of our prediction    


        print("Classification: %s, Confidence: %s" % (classification, confidence))

        sender_helper(recv_con, classification, confidence)
