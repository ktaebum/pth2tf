"""
conversion rules for shape related operations
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys

import numpy as np

import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops

def shape_ops(name):
    """ wrapper of various shape related ops
    """
    def construct(*args):
        if name == "aten::view":
            return tf.reshape(*args)
        elif name == "aten::flatten":
            inputs = args[0]
            input_shape = inputs.shape
            if input_shape[1:].is_fully_defined():
                flattened_dim = tensor_shape.dimension_value(
                    np.prod(input_shape[1:], dtype=int))
                outputs = tf.reshape(inputs, (-1, flattened_dim))
            else:
                outputs = tf.reshape(
                    inputs, (tensor_shape.dimension_value(inputs.shape[0]) or
                             array_ops.shape(inputs)[0], -1))
            return outputs
        elif name == "aten::size":
            return tf.shape(args[0])[args[1]]
        elif name == "aten::t":
            return tf.transpose(*args)
        elif name == "aten::constant_pad_nd":
            inputs = args[0]
            padding = args[1]
            value = args[2]
            pad_left = padding[0]
            pad_right = padding[1]
            pad_up = padding[2]
            pad_down = padding[3]

            return tf.pad(inputs, [[0, 0], [0, 0],
                                   [pad_up, pad_down],
                                   [pad_left, pad_right]],
                          constant_values=value)
        elif name == "aten::slice":
            inputs = args[0]
            dim = args[1]
            start = args[2]
            end = args[3]
            step = args[4]

            slice_obj = [slice(None) if i != dim else slice(start, end, step) \
                         for i in range(len(inputs.shape))]

            return inputs[slice_obj]
        elif name == "aten::select":
            inputs = args[0]
            target_dim = args[1]
            index = args[2]
            
            slice_obj = [slice(None) for _ in range(target_dim - 1)]
            slice_obj.append(index)
            
            return inputs[slice_obj]
        elif name == "aten::chunk":
            inputs = args[0]
            splits = args[1]
            axis = args[2]
            return array_ops.split(
                inputs, num_or_size_splits=splits, axis=axis)
        else:
            raise ValueError("Not support", name)

    return construct

