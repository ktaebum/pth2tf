"""
conversion rules for activation operations
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.python.framework import constant_op

#TODO: merge activations.py with math.py?

def activation(name):
    """ wrapper of various activation ops
    """
    def construct(*args):
        if name == "aten::relu":
            return tf.nn.relu(*args)
        elif name == "aten::leaky_relu_":
            return tf.nn.leaky_relu(*args)
        elif name == "aten::tanh":
            return tf.tanh(*args)
        elif name == "aten::hardtanh_":
            inputs = args[0]
            min_value = args[1]
            max_value = args[2]

            if min_value == 0 and max_value == 6:
                # same as relu6
                return tf.nn.relu6(inputs)
            else:
                dtype = inputs.dtype
                min_value = constant_op.constant(min_value, dtype=dtype)
                max_value = constant_op.constant(max_value, dtype=dtype)
                return tf.minimum(tf.maximum(inputs, min_value), max_value)
        elif name == "aten::sigmoid":
            return tf.sigmoid(*args)
        elif name == "aten::softmax":
            return tf.nn.softmax(args[0], args[1])
        else:
            raise ValueError("Not support", name)

    return construct
