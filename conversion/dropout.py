"""
dropout ops conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.python.framework import smart_cond

def dropout(name):
    """ wrapper for various dropouts """

    def construct(*args):
        if name == "aten::dropout":
            assert 'Placeholder' in args[-1].op.type, 'need training flag'
            inputs = args[0]
            prob = args[1]
            training = args[2]

            return smart_cond.smart_cond(args[-1],
                           lambda: tf.nn.dropout(inputs,
                                                 rate=prob),
                           lambda: tf.identity(inputs))
        else:
            raise ValueError("Not support", name)

    return construct
