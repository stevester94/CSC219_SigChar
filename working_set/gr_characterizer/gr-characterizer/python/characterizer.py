#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Copyright 2019 <+YOU OR YOUR COMPANY+>.
# 
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
# 
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
# 

import numpy
from gnuradio import gr
import socket
import struct

CAAS_IP = "127.0.0.1"
CAAS_PORT = 1337
BUFFER_SIZE = 20

class characterizer(gr.sync_block):
    """
    docstring for block characterizer
    """
    def __init__(self):
        gr.sync_block.__init__(self,
            name="characterizer",
            in_sig=[numpy.complex64],
            out_sig=None)
        self.buffer = []
        self.threshold = 0
        self.buffer_target_len = 1024

    def request_characterization(self, IQ):
        print("Sum of payload: %f" % sum(IQ))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((CAAS_IP, CAAS_PORT))

        buf = struct.pack('%sf' % len(IQ), *IQ)

        sock.sendall(buf)

        response = sock.recv(BUFFER_SIZE)
        sock.close()

        classification, confidence = struct.unpack("If", response)

        return classification, confidence

    def work(self, input_items, output_items):
        # Input items is an array of one element,
        #    the single element is an array of numpy complex

        input_iq = input_items[0]
        magnitudes = numpy.absolute(input_iq)

        for index, magnitude in enumerate(magnitudes):
            if magnitude > self.threshold:
                self.buffer.append(input_iq[index])

                if len(self.buffer) == self.buffer_target_len:
                    np_array = numpy.array(self.buffer)

                    classification, confidence = self.request_characterization(np_array.view(numpy.float32))

                    print("Classification: %d, confidence: %f" % (classification, confidence))

                    self.buffer = []
                    break
            else:
                print("Signal too low (%f), clearing buffer" % magnitude)
                self.buffer = []

        # I guess this is important
        return len(input_items[0])

