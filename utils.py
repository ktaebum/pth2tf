""" modules for some helper functions """
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

def tf_padding(input_shape,
               ksize,
               stride,
               padding,
               dilation=(1, 1),
               ceil_mode=False,
               data_format="NHWC"):
    """ convert pytorch padding to tensorflow padding
    Actually, this function is not for convolution since tf.nn.conv2d
    supports padding input as list (https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)

    However, other ops like pooling do not support

    Assume input_shape is given as NHWC format

    use simplest method
    calculate pytorch's output shape as
    floor((input + 2 * padding - dilation * (kernel - 1) -1) / stride + 1)

    calculate tensorflow's output shape as
    'SAME'
        - ceil(input / stride)
    'VALID'
        - ceil((input - (kernel - 1) * dilation) / stride)

    got pytorch rule (general shape formula) from
    https://pytorch.org/docs/master/nn.html#conv2d

    got tensorflow's rule from
    https://www.tensorflow.org/api_docs/python/tf/nn/convolution
    """
    if isinstance(padding, int):
        padding = [padding, padding]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(stride, int):
        stride = [stride, stride]

    if data_format == "NHWC":
        h_idx = 1
        w_idx = 2
    elif data_format == "NCHW":
        h_idx = 2
        w_idx = 3
    else:
        raise ValueError("Invalid data format", data_format)

    def tf_same():
        out_h = math.ceil(input_shape[h_idx] / stride[0])
        out_w = math.ceil(input_shape[w_idx] / stride[1])
        return out_h, out_w

    def tf_valid():
        out_h = math.ceil((input_shape[h_idx] - (ksize[0] - 1) * dilation[0])\
                          / stride[0])
        out_w = math.ceil((input_shape[w_idx] - (ksize[1] - 1) * dilation[1])\
                          / stride[1])
        return out_h, out_w

    def torch_shape():
        truncate = math.ceil if ceil_mode else math.floor
        out_h = truncate((input_shape[h_idx] + 2 * padding[0] - dilation[0]\
                            * (ksize[0] - 1) - 1) / stride[0] + 1)
        out_w = truncate((input_shape[w_idx] + 2 * padding[1] - dilation[1]\
                            * (ksize[1] - 1) - 1) / stride[1] + 1)
        return out_h, out_w

    pth_shape = torch_shape()
    if pth_shape == tf_same():
        return "SAME"
    elif pth_shape == tf_valid():
        return "VALID"
    else:
        raise ValueError("Output shape does not match")
