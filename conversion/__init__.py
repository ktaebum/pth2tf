"""
convert Pytorch IR, dtype into Tensorflow Op, dtype
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

import tensorflow as tf

import numpy as np

from .convolution import convolution
from .gemm import gemm
from .normalization import batch_norm
from .math import math_ops
from .activations import activation
from .pool import pool
from .cast import cast
from .dropout import dropout
from .shape import shape_ops
from .recurrent import recurrent

_DTYPE_MAP = {
    # floating point
    torch.float16: tf.float16,
    torch.half: tf.float16,
    torch.float32: tf.float32,
    torch.float: tf.float32,
    torch.double: tf.float64,
    torch.float64: tf.float64,
    # integer
    torch.uint8: tf.uint8,
    torch.int8: tf.int8,
    torch.int16: tf.int16,
    torch.short: tf.int16,
    torch.int32: tf.int32,
    torch.int: tf.int32,
    torch.int64: tf.int64,
    torch.long: tf.int64,
    # boolean
    torch.bool: tf.bool,
    # ScalarType
    # from https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
    0: tf.uint8,
    1: tf.int8,
    2: tf.int16,
    3: tf.int32,
    4: tf.int64,
    5: tf.float16,
    6: tf.float32,
    7: tf.float64,
    11: tf.bool,
}

def dtype_map(dtype):
    if isinstance(dtype, np.dtype):
        return tf.as_dtype(dtype)
    return _DTYPE_MAP[dtype]

_OP_MAP = {
    # prim ops
    "prim::Constant": lambda x: x,
    "prim::NumToTensor": lambda x: x,  # is it realy need to convert to tensor?
    "prim::ListConstruct": lambda *x: list(x),
    "prim::ListUnpack": lambda x: x,
    "prim::TupleConstruct": lambda *x: tuple(x),
    "prim::TupleUnpack": lambda x: x,
    # basic math operations
    "aten::add": math_ops("aten::add"),
    "aten::add_": math_ops("aten::add"),
    "aten::mul": math_ops("aten::mul"),
    "aten::mul_": math_ops("aten::mul"),
    "aten::sub": math_ops("aten::sub"),
    "aten::div": math_ops("aten::div"),
    "aten::floor": math_ops("aten::floor"),
    "aten::ceil": math_ops("aten::ceil"),
    "aten::log_softmax": math_ops("aten::log_softmax"),
    "aten::nll_loss": math_ops("aten::nll_loss"),
    "aten::addmm": gemm("aten::addmm"),
    "aten::matmul": gemm("aten::matmul"),
    # activations
    "aten::relu": activation("aten::relu"),
    "aten::relu_": activation("aten::relu"),
    "aten::leaky_relu_": activation("aten::leaky_relu_"),
    "aten::tanh": activation("aten::tanh"),
    "aten::tanh_": activation("aten::tanh"),
    "aten::hardtanh_": activation("aten::hardtanh_"),
    "aten::sigmoid": activation("aten::sigmoid"),
    "aten::sigmoid_": activation("aten::sigmoid"),
    "aten::softmax": activation("aten::softmax"),
    # shape related
    "aten::view": shape_ops("aten::view"),
    "aten::flatten": shape_ops("aten::flatten"),
    "aten::size": shape_ops("aten::size"),
    "aten::t": shape_ops("aten::t"),
    "aten::constant_pad_nd": shape_ops("aten::constant_pad_nd"),
    "aten::slice": shape_ops("aten::slice"),
    "aten::select": shape_ops("aten::select"),
    "aten::chunk": shape_ops("aten::chunk"),
    # type, device placement casting
    "aten::Int": cast("aten::Int"),
    "aten::to": cast("aten::to"),
    # convolutions
    "aten::_convolution": convolution("aten::_convolution"),
    # recurrents
    "aten::lstm": recurrent("aten::lstm"),
    "aten::rnn_tanh": recurrent("aten::rnn_tanh"),
    "aten::rnn_relu": recurrent("aten::rnn_relu"),
    "aten::gru": recurrent("aten::gru"),
    # pooling & unpooling (upsample included)
    "aten::max_pool2d": pool("aten::max_pool2d"),
    "aten::adaptive_avg_pool2d": pool("aten::adaptive_avg_pool2d"),
    "aten::upsample_nearest2d": pool("aten::upsample_nearest2d"),
    # normalizations
    "aten::batch_norm": batch_norm("aten::batch_norm"),
    # dropouts
    "aten::dropout": dropout("aten::dropout"),
    "aten::feature_dropout": dropout("aten::dropout"),
    # embeddings
    "aten::embedding": lambda *x: tf.nn.embedding_lookup(x[0], x[1]),
    # misc torch ops
    "aten::detach": tf.stop_gradient,
    "aten::zeros": lambda *x: tf.zeros(x[0], dtype_map(x[1])),
    "aten::ones": lambda *x: tf.ones(x[0], dtype_map(x[1])),
}

def op_map(ir):
    assert isinstance(ir, torch.Node), "Must be torch.Node"
    kind = ir.kind()
    if kind in _OP_MAP:
        return _OP_MAP[kind]
    raise KeyError("%s is not in opmap" % kind)
