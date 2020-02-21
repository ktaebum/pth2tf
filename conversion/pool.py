"""
pooling ops conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from converter.utils import tf_padding

def pool(name):
    """ wrapper of various pooling ops
    """
    def construct(*args):
        if name == "aten::max_pool2d":
            inputs = args[0]
            ksize = args[1]
            stride = args[2]
            if stride == []:
                # follow pytorch's default behavior
                stride = ksize
            padding = args[3]
            dilation = args[4]
            ceil_mode = args[5]
            inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                     [padding[0], padding[0]],
                                     [padding[1], padding[1]]])
            """
            padding_string = tf_padding(inputs.shape,
                                        ksize,
                                        stride,
                                        padding,
                                        dilation,
                                        ceil_mode)
            """

            return tf.nn.max_pool2d(inputs,
                                    ksize,
                                    stride,
                                    "VALID",
                                    # padding_string,
                                    "NCHW")
        elif name == "aten::adaptive_avg_pool2d":
            assert args[0].shape[2] % args[1][0] == 0 and \
                args[0].shape[3] % args[1][1] == 0, "Must be divided"
            inputs = args[0]
            ih = inputs.shape[2]
            iw = inputs.shape[3]

            output_size = args[1]
            oh = output_size[0]
            ow = output_size[1]

            kh = ih - (oh - 1) * (ih // oh)
            kw = iw - (ow - 1) * (iw // ow)

            sh = ih // oh
            sw = iw // ow

            return tf.nn.avg_pool2d(inputs,
                                    [kh, kw],
                                    [sh, sw],
                                    "VALID",
                                    "NCHW")
        elif name == "aten::upsample_nearest2d":
            inputs = args[0]
            output_size = args[1]

            # permute to follow NHWC
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

            inputs = tf.image.resize(inputs,
                                     output_size,
                                     method=tf.image.\
                                     ResizeMethod.NEAREST_NEIGHBOR)
            # make it to NCHW again
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
            return inputs
        else:
            raise ValueError('Not support', name)
    return construct
