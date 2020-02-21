"""
casting ops conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def cast(name):
    """ wrapper for various type and device placement cast """

    def construct(*args):
        if name == "aten::Int":
            return tf.cast(args[0], tf.int32)
        elif name == "aten::to":
            # pylint: disable=import-outside-toplevel
            from converter.conversion import dtype_map
            # pylint: enable=import-outside-toplevel
            if len(args) == 4:
                # args are
                # (Tensor, dtype, non_blocking, copy)
                inputs = args[0]
                dtype = args[1]
                return tf.cast(inputs, dtype_map(dtype))
            elif len(args) == 5:
                # args are
                # (Tensor, device, dtype, non_blocking, copy)
                # TODO: handle device placement option?
                inputs = args[0]
                dtype = args[2]
                return tf.cast(inputs, dtype_map(dtype))
            else:
                # return inputs
                return args[0]
        else:
            raise ValueError("Not support", name)

    return construct
