"""
conversion rules for basic math operations
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def math_ops(name):
    """ wrapper of various math ops
    """
    def construct(*args):
        if name == "aten::add":
            inputs = args[0]
            dtype = inputs.dtype

            operand = tf.cast(tf.multiply(args[1], args[2]), dtype)

            return tf.add(inputs, operand)
        elif name == "aten::mul":
            inputs = args[0]
            dtype = inputs.dtype

            operand = tf.cast(args[1], dtype)
            return tf.multiply(inputs, operand)
        elif name == "aten::div":
            inputs = args[0]
            dtype = inputs.dtype

            operand = tf.cast(args[1], dtype)
            return tf.divide(inputs, operand)
        elif name == "aten::floor":
            inputs = args[0]

            return tf.floor(inputs)
        elif name == "aten::ceil":
            inputs = args[0]

            return tf.math.ceil(inputs)
        elif name == "aten::sub":
            inputs = args[0]
            dtype = inputs.dtype

            operand = tf.cast(tf.multiply(args[1], args[2]), dtype)
            return tf.subtract(inputs, operand)
        elif name == "aten::log_softmax":
            inputs = args[0]
            if args[2] is not None:
                from converter.conversion import dtype_map
                dtype = dtype_map(args[2])
                inputs = tf.cast(inputs, dtype)

            return tf.nn.log_softmax(inputs, args[1])
        elif name == "aten::nll_loss":
            # TODO: support other options
            inputs = args[0]
            targets = args[1]
            weights = args[2]
            reduction = args[3]
            ignore_index = args[4]

            num_classes = inputs.shape[1]
            hard_prob = tf.one_hot(targets, num_classes)

            loss = inputs * hard_prob
            non_zero_idx = tf.where(loss != 0)  # prevent inf
            loss = tf.gather_nd(loss, non_zero_idx)

            if reduction == 1:
                # mean reduction
                loss = tf.reduce_mean(loss)
            elif reduction == 2:
                # sum reduction
                loss = tf.reduce_sum(loss)
            return loss
        else:
            raise ValueError("Not support", name)

    return construct
