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

class basicsink_py_c(gr.sync_block):
    """
    docstring for block basicsink_py_c
    """
    def __init__(self ):
        gr.sync_block.__init__(self,
            name="basicsink_py_c",
            in_sig=[numpy.complex64],
            out_sig=None)


    def work(self, input_items, output_items):
        # Input items is an array of one element,
        #    the single element is an array of numpy complex
        
        input_iq = input_items[0]
        magnitudes = numpy.absolute(input_iq)

        print(magnitudes)


        # I guess this is important
        return len(input_items[0])

