"""
general matrix multiplication (GEMM) conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def gemm(name):
    """ wrapper of various gemm ops
    current version just convert to dense gemm
    (Pytorch ATen supports sparse optim)
    """
    # TODO: support sparse optimization
    def construct(*args):
        if name == "aten::addmm":
            bias = args[0]

            inputs = args[1]

            kernel = args[2]

            beta = args[3]
            alpha = args[4]
            return alpha * tf.matmul(inputs, kernel) + beta * bias
        elif name == "aten::matmul":
            # nn.Linear without bias
            return tf.matmul(*args)
        else:
            raise ValueError('Not support', name)

    return construct
