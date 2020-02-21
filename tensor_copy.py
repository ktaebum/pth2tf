"""
Module for copying torch tensor to tf tensor
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import tensorflow as tf

import numpy as np

from converter.conversion import dtype_map

class TensorInfo:
    """ Wrapped TorchTensor
    It does not contain actual tensor value, just shape, data type
    and dynamic axes (if exists)

    Supports torch.Tensor and numpy.ndarray
    """
    def __init__(self, tensor, dynamic_axes=[], name=''):
        assert isinstance(tensor, (torch.Tensor, np.ndarray)), "Must be torch.Tensor or np array"
        self._dtype = dtype_map(tensor.dtype)
        self._shape = list(tensor.shape)
        self._dynamic_axes = dynamic_axes
        self._name = name

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        shape = self._shape.copy()
        for dyn_axis in self._dynamic_axes:
            shape[dyn_axis] = None
        return shape

    @property
    def name(self):
        return self._name

def to_numpy(tensor):
    """ torch tensor to numpy tensor """
    if tensor.is_cuda:
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.cpu().numpy()
    else:
        if tensor.requires_grad:
            return tensor.detach().numpy()
        return tensor.numpy()

def copy_through_numpy(tensor, var_id):
    """ naive approach copy torch tensor to tf tensor """
    assert isinstance(tensor, torch.Tensor), "Must be torch Tensor"

    np_tensor = to_numpy(tensor)
    dtype = dtype_map(tensor.dtype)

    return tf.compat.v1.get_variable(
        'variable_%d' % var_id,
        initializer=np_tensor,
        dtype=dtype,
        trainable=isinstance(tensor, nn.Parameter))

def build_args(tensors):
    """ from given tensor informations, build input arguments """
    args = []

    def build_single_item(item):
        if isinstance(item, (tf.Tensor, tf.data.Dataset)):
            # just add to args directly
            return item

        # Not it must be a TensorInfo
        assert isinstance(item, TensorInfo)

        return tf.compat.v1.placeholder(item.dtype,
                                        item.shape,
                                        item.name)

    for tensor in tensors:
        if isinstance(tensor, tuple):
            tensor = tuple(map(build_single_item, tensor))
        elif isinstance(tensor, list):
            tensor = list(map(build_single_item, tensor))
        else:
            tensor = build_single_item(tensor)

        args.append(tensor)

    return args
