"""
normalization conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond

def _assign_moving_average(variable, value, momentum):
    """
    assign moving average
    Got from official implementation of tf.keras.layers.BatchNormalization
    """
    with tf.name_scope('AssignMoviginAverage') as scope:
        with ops.colocate_with(variable):
            decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
            if decay.dtype != variable.dtype.base_dtype:
                decay = math_ops.cast(decay, variable.dtype.base_dtype)
            update_delta = (
                variable - tf.cast(value, variable.dtype)) * decay
            return variable.assign_sub(update_delta, name=scope)

def _bn_layer(inputs, gamma, beta, mean, variance,
              training, eps, reduction_axes, broadcast_shape):
    """ non-fused batch-norm layer """
    if training:
        avg, var = tf.nn.moments(inputs, reduction_axes)
    else:
        avg, var = mean, variance

    ndims = len(inputs.shape)

    def broadcast(v):
        if (v is not None and len(v.shape) != ndims and
                reduction_axes != list(range(ndims - 1))):
            return tf.reshape(v, broadcast_shape)
        return v

    output = tf.nn.batch_normalization(inputs,
                                       broadcast(avg),
                                       broadcast(var),
                                       broadcast(beta),
                                       broadcast(gamma),
                                       eps)
    return output, avg, var

def _bn_fused(inputs, gamma, beta, mean,
              variance, training, eps):
    """ fused batch norm layer """
    if training:
        output, avg, var = nn.fused_batch_norm(
            inputs,
            gamma,
            beta,
            epsilon=eps,
            data_format='NCHW')
    else:
        output, avg, var = nn.fused_batch_norm(
            inputs,
            gamma,
            beta,
            mean=mean,
            variance=variance,
            is_training=False,
            data_format='NCHW',
            epsilon=eps)
    return output, avg, var

def batch_norm(name):
    """ wrapper of various batch norm
    """
    def construct(*args):
        if name == "aten::batch_norm":
            # assume fused batch norm
            assert 'Placeholder' in args[-1].op.type, 'need training flag'
            inputs = args[0]
            input_shape = inputs.shape

            ndims = len(input_shape)
            gamma = args[1]
            beta = args[2]
            mean = args[3]
            variance = args[4]
            momentum = args[6]
            eps = args[7]

            if len(input_shape) == 4:
                # use fused batch norm
                # TODO: any other condition to check fusable?
                # keras normalization use _fused_can_be_used if not use V2 Behavior
                output, avg, var = smart_cond.smart_cond(args[5],
                                           lambda: _bn_fused(inputs, gamma,
                                                             beta, mean,
                                                             variance, True,
                                                             eps),
                                           lambda: _bn_fused(inputs, gamma,
                                                             beta, mean,
                                                             variance, False,
                                                             eps))

                moving_variables = lambda: (avg, var)
                def update_variables_fused():
                    moving_mean = _assign_moving_average(mean, avg, momentum)
                    moving_var = _assign_moving_average(variance, var, momentum)
                    return moving_mean, moving_var

                update_means, update_vars = smart_cond.smart_cond(
                    args[5], update_variables_fused, moving_variables)

                ops.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                      update_means)
                ops.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                      update_vars)
                return output

            reduction_axes = [i for i in range(ndims) if i not in [1]]

            broadcast_shape = [1] * ndims
            broadcast_shape[1] = input_shape.dims[1].value

            output, avg, var = tf.cond(args[-1],
                                       lambda: _bn_layer(inputs, gamma, beta, mean,
                                                         variance, True,
                                                         eps, reduction_axes,
                                                         broadcast_shape),
                                       lambda: _bn_layer(inputs, gamma, beta, mean,
                                                         variance, False,
                                                         eps, reduction_axes,
                                                         broadcast_shape))

            moving_variables = lambda: (avg, var)
            def update_variables():
                moving_mean = _assign_moving_average(mean, avg, momentum)
                moving_var = _assign_moving_average(variance, var, momentum)
                return moving_mean, moving_var

            update_means, update_vars = tf.cond(
                args[-1], update_variables, moving_variables)

            ops.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                  update_means)
            ops.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS,
                                  update_vars)
            return output
        else:
            raise ValueError('Not support', name)

    return construct
