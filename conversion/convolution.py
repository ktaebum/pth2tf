"""
convolution conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from converter.utils import tf_padding

def convolution(name):
    """ wrapper of various aten::convolution ops
    """
    def construct(*args):
        if name == "aten::_convolution":
            # convert to conv2d currently
            inputs = args[0]
            input_shape = inputs.shape

            filters = args[1]

            bias = args[2]

            strides = args[3]
            padding = args[4]
            dilations = args[5]
            groups = args[8]

            if groups != 1:
                assert input_shape[1] % groups == 0
                if input_shape[1] == groups:
                    # depthwise conv
                    padding = tf_padding(input_shape,
                                         [filters.shape[2], filters.shape[3]],
                                         strides,
                                         padding,
                                         dilations,
                                         data_format='NCHW')
                    inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                             [padding[0], padding[0]],
                                             [padding[1], padding[1]]])
                    filters = tf.transpose(filters, (2, 3, 0, 1))
                    strides = [1, 1, strides[0], strides[1]]
                    conv = tf.nn.depthwise_conv2d(inputs,
                                                  filters,
                                                  strides,
                                                  "VALID",
                                                  padding,
                                                  "NCHW",
                                                  dilations)
                else:
                    # grouped conv
                    # https://github.com/tensorflow/tensorflow/pull/10482/files
                    # TODO: more faster application?
                    filters = tf.transpose(filters, (2, 3, 1, 0))
                    padding = [[0, 0], [0, 0],
                               [padding[0], padding[0]],
                               [padding[1], padding[1]]]
                    input_slices = tf.split(inputs, groups, axis=1)
                    filters_slices = tf.split(filters, groups, axis=-1)
                    output_slices = [tf.nn.conv2d(
                        input_slice,
                        filters_slice,
                        strides,
                        padding,
                        'NCHW',
                        dilations) for input_slice, filters_slice in \
                                     zip(input_slices, filters_slices)]
                    conv = tf.concat(output_slices, axis=1)
            else:
                # normal conv
                filters = tf.transpose(filters, (2, 3, 1, 0))
                padding = [[0, 0], [0, 0],
                           [padding[0], padding[0]],
                           [padding[1], padding[1]]]
                conv = tf.nn.conv2d(inputs,
                                    filters,
                                    strides,
                                    padding,
                                    'NCHW',
                                    dilations)
            if bias is not None:
                conv = tf.nn.bias_add(conv, bias, data_format='NC...')
            return conv
        else:
            raise ValueError('Not support', name)

    return construct
