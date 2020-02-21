"""
recurrent network (RNN, LSTM, ...) conversion rules
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.util import nest

def _rnn_cell_forward(kernel, recurrent_kernel,
                      bias=None, recurrent_bias=None, nonlinearity="tanh"):
    def step(inputs, states):
        """ single step function for rnn (implementation == 2 in TF) """
        h, = states

        x = math_ops.matmul(inputs, kernel, transpose_b=True)

        if bias is not None:
            x = nn_ops.bias_add(x, bias)

        h = math_ops.matmul(h, recurrent_kernel, transpose_b=True)

        if recurrent_bias is not None:
            h = nn_ops.bias_add(h, recurrent_bias)

        if nonlinearity == 'tanh':
            h = gen_math_ops.tanh(x + h)
        else:
            h = gen_nn_ops.relu(x + h)

        return h, [h]
    return step

def _lstm_cell_forward_fused(kernel, recurrent_kernel,
                             bias=None, recurrent_bias=None):
    def step(inputs, states):
        h, c = states
        z = math_ops.matmul(inputs, kernel, transpose_b=True)
        z += math_ops.matmul(h, recurrent_kernel, transpose_b=True)

        if bias is not None:
            z = nn_ops.bias_add(z, bias)

        if recurrent_bias is not None:
            z = nn_ops.bias_add(z, recurrent_bias)

        z_i, z_f, z_c, z_o = array_ops.split(
            z, num_or_size_splits=4, axis=1)

        i = math_ops.sigmoid(z_i)
        f = math_ops.sigmoid(z_f)
        c = f * c + i * gen_math_ops.tanh(z_c)
        o = math_ops.sigmoid(z_o)

        h = o * gen_math_ops.tanh(c)

        return h, [h, c]
    return step

def _lstm_cell_forward(kernel, recurrent_kernel,
                       bias=None, recurrent_bias=None):
    def step(inputs, states):
        """ single step function for lstm """
        h, c = states

        k_i, k_f, k_c, k_o = array_ops.split(
            kernel, num_or_size_splits=4, axis=0)

        rk_i, rk_f, rk_c, rk_o = array_ops.split(
            recurrent_kernel, num_or_size_splits=4, axis=0)

        x_i = math_ops.matmul(inputs, k_i, transpose_b=True)
        x_f = math_ops.matmul(inputs, k_f, transpose_b=True)
        x_c = math_ops.matmul(inputs, k_c, transpose_b=True)
        x_o = math_ops.matmul(inputs, k_o, transpose_b=True)

        if bias is not None:
            b_i, b_f, b_c, b_o = array_ops.split(
                bias, num_or_size_splits=4, axis=0)
            x_i = nn_ops.bias_add(x_i, b_i)
            x_f = nn_ops.bias_add(x_f, b_f)
            x_c = nn_ops.bias_add(x_c, b_c)
            x_o = nn_ops.bias_add(x_o, b_o)

        h_i = math_ops.matmul(h, rk_i, transpose_b=True)
        h_f = math_ops.matmul(h, rk_f, transpose_b=True)
        h_c = math_ops.matmul(h, rk_c, transpose_b=True)
        h_o = math_ops.matmul(h, rk_o, transpose_b=True)

        if recurrent_bias is not None:
            rb_i, rb_f, rb_c, rb_o = array_ops.split(
                recurrent_bias, num_or_size_splits=4, axis=0)
            h_i = nn_ops.bias_add(h_i, rb_i)
            h_f = nn_ops.bias_add(h_f, rb_f)
            h_c = nn_ops.bias_add(h_c, rb_c)
            h_o = nn_ops.bias_add(h_o, rb_o)

        i = math_ops.sigmoid(x_i + h_i)
        f = math_ops.sigmoid(x_f + h_f)
        g = gen_math_ops.tanh(x_c + h_c)
        o = math_ops.sigmoid(x_o + h_o)

        c = f * c + i * g
        h = o * gen_math_ops.tanh(c)
        return h, [h, c]
    return step

def _gru_cell_forward(kernel, recurrent_kernel,
                      bias=None, recurrent_bias=None):
    def step(inputs, states):
        h, = states

        k_r, k_z, k_h = array_ops.split(
            kernel, num_or_size_splits=3, axis=0)

        x_r = math_ops.matmul(inputs, k_r, transpose_b=True)
        x_z = math_ops.matmul(inputs, k_z, transpose_b=True)
        x_h = math_ops.matmul(inputs, k_h, transpose_b=True)

        if bias is not None:
            b_r, b_z, b_h = array_ops.split(
                bias, num_or_size_splits=3, axis=0)
            x_r = nn_ops.bias_add(x_r, b_r)
            x_z = nn_ops.bias_add(x_z, b_z)
            x_h = nn_ops.bias_add(x_h, b_h)

        rk_r, rk_z, rk_h = array_ops.split(
            recurrent_kernel, num_or_size_splits=3, axis=0)

        r_r = math_ops.matmul(h, rk_r, transpose_b=True)
        r_z = math_ops.matmul(h, rk_z, transpose_b=True)
        r_h = math_ops.matmul(h, rk_h, transpose_b=True)

        if recurrent_bias is not None:
            rb_r, rb_z, rb_h = array_ops.split(
                recurrent_bias, num_or_size_splits=3, axis=0)
            r_r = nn_ops.bias_add(r_r, rb_r)
            r_z = nn_ops.bias_add(r_z, rb_z)
            r_h = nn_ops.bias_add(r_h, rb_h)

        r = math_ops.sigmoid(x_r + r_r)
        z = math_ops.sigmoid(x_z + r_z)
        n = gen_math_ops.tanh(x_h + r * r_h)
        h = (1 - z) * n + z * h

        return h, [h]
    return step

def _cudnn_rnn(inputs, init_h, params, time_major,
               dropout=0, direction='unidirectional', nonlinearity='tanh'):
    assert nonlinearity in {'tanh', 'relu'}

    if not time_major:
        inputs = array_ops.transpose(inputs, perm=(1, 0, 2))

    weights = []
    biases = []
    for i in range(0, len(params), 4):
        # collect params
        kernel, recurrent_kernel, bias, recurrent_bias = params[i:i+4]

        weights.append(kernel)
        weights.append(recurrent_kernel)

        biases.append(bias)
        biases.append(recurrent_bias)

    params = recurrent_v2._canonical_to_params(
        weights=weights,
        biases=biases,
        shape=constant_op.constant([-1]),
        transpose_weights=False)

    outputs, h, _, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs, input_h=init_h, input_c=0, params=params, is_training=True,
        rnn_mode='rnn_%s' % nonlinearity, dropout=dropout, direction=direction)

    if not time_major:
        outputs = array_ops.transpose(outputs, perm=(1, 0, 2))

    return outputs, h

def _cudnn_lstm(inputs, init_h, init_c, params, time_major,
                dropout=0, direction='unidirectional'):
    """ lstm forward using cudnn """
    if not time_major:
        inputs = array_ops.transpose(inputs, perm=(1, 0, 2))

    weights = []
    biases = []
    for i in range(0, len(params), 4):
        # collect params
        kernel, recurrent_kernel, bias, recurrent_bias = params[i:i+4]

        weights += array_ops.split(kernel, 4)
        weights += array_ops.split(recurrent_kernel, 4)

        biases += array_ops.split(bias, 4)
        biases += array_ops.split(recurrent_bias, 4)

    params = recurrent_v2._canonical_to_params(
        weights=weights,
        biases=biases,
        shape=constant_op.constant([-1]),
        transpose_weights=False)

    outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs, input_h=init_h, input_c=init_c, params=params, is_training=True,
        rnn_mode="lstm", direction=direction, dropout=dropout)

    if not time_major:
        outputs = array_ops.transpose(outputs, perm=(1, 0, 2))

    return outputs, h, c

def _cudnn_gru(inputs, init_h, params, time_major,
               dropout=0, direction='unidirectional'):
    """ gru forward using cudnn """
    if not time_major:
        inputs = array_ops.transpose(inputs, perm=(1, 0, 2))

    weights = []
    biases = []
    for i in range(0, len(params), 4):
        # collect params
        kernel, recurrent_kernel, bias, recurrent_bias = params[i:i+4]

        weights += array_ops.split(kernel, 3)
        weights += array_ops.split(recurrent_kernel, 3)

        biases += array_ops.split(bias, 3)
        biases += array_ops.split(recurrent_bias, 3)

    params = recurrent_v2._canonical_to_params(
        weights=weights,
        biases=biases,
        shape=constant_op.constant([-1]),
        transpose_weights=False)

    outputs, h, _, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs, input_h=init_h, input_c=0, params=params, is_training=True,
        rnn_mode='gru', dropout=dropout, direction=direction)

    if not time_major:
        outputs = array_ops.transpose(outputs, perm=(1, 0, 2))

    return outputs, h

def _rnn(step_function, inputs, initial_states, time_major, backward=False):
    """ general non-cudnn rnn
    step function contains cell information of rnn (rnn, lstm, gru...)

    got from keras.backend.rnn
    """
    if not time_major:
        inputs = array_ops.transpose(inputs, perm=(1, 0, 2))

    flatted_inputs = nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = array_ops.shape(flatted_inputs[0])[0]

    for input_ in flatted_inputs:
        input_.shape.with_rank_at_least(3)

    states = tuple(initial_states)
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))

    input_ta = tuple(
        ta.unstack(input_) if not backward \
        else ta.unstack(array_ops.reverse(input_, [0])) \
        for ta, input_ in zip(input_ta, flatted_inputs))

    input_time_zero = nest.pack_sequence_as(
        inputs, [inp[0]for inp in flatted_inputs])

    output_time_zero, _ = step_function(
        input_time_zero, states)

    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            element_shape=out.shape,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    while_loop_kwargs = {
        'cond': lambda time_, *_: time < time_steps_t,
        'maximum_iterations': time_steps,
        'parallel_iterations': 32,
        'swap_memory': True,
    }

    def _while_step(time, output_ta_t, *states):
        """ LSTM step function in loop """

        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        output, new_states = step_function(current_input, states)

        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
            if isinstance(new_state, ops.Tensor):
                new_state.set_shape(state.shape)

        flat_output = nest.flatten(output)
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)

        return (time + 1, output_ta_t) + tuple(new_states)

    final_outputs = control_flow_ops.while_loop(
        body=_while_step,
        loop_vars=(time, output_ta) + states,
        **while_loop_kwargs)

    new_states = final_outputs[2:]

    output_ta = final_outputs[1]

    outputs = tuple(o.stack() for o in output_ta)
    last_output = tuple(o[-1] for o in outputs)

    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

    def set_shape(output_):
        if isinstance(output_, ops.Tensor):
            shape = output_.shape.as_list()
            shape[0] = time_steps
            shape[1] = batch
            output_.set_shape(shape)
        return output_

    outputs = nest.map_structure(set_shape, outputs)

    if not time_major:
        outputs = array_ops.transpose(outputs, perm=(1, 0, 2))

    return last_output, outputs, new_states 

def recurrent_forward(inputs, layer, dropout_p, initial_states,
                      cell_type, params, time_major, train, bidirectional):
    """ common recurrent forward function (non-cudnn use case) """
    # apply dropout
    inputs = smart_cond.smart_cond(
        train and layer > 0 and dropout_p > 0.0,
        lambda: nn_ops.dropout(inputs, rate=dropout_p),
        lambda: array_ops.identity(inputs))

    if bidirectional:
        param_pivot = len(params) // 2

        forward_params = params[:param_pivot]
        backward_params = params[param_pivot:]

        forward_states = [initial_states[0][0]]
        backward_states = [initial_states[0][1]]
        if cell_type == 'rnn_tanh':
            forward_step = _rnn_cell_forward(*forward_params, nonlinearity='tanh')
            backward_step = _rnn_cell_forward(*backward_params, nonlinearity='tanh')
        elif cell_type == 'rnn_relu':
            forward_step = _rnn_cell_forward(*forward_params, nonlinearity='relu')
            backward_step = _rnn_cell_forward(*backward_params, nonlinearity='relu')
        elif cell_type == 'lstm':
            forward_step = _lstm_cell_forward(*forward_params)
            backward_step = _lstm_cell_forward(*backward_params)
            forward_states.append(initial_states[1][0])
            backward_states.append(initial_states[1][1])
        elif cell_type == 'gru':
            forward_step = _gru_cell_forward(*forward_params)
            backward_step = _gru_cell_forward(*backward_params)
        else:
            raise ValueError('Invalid cell type', cell_type)

        _, f_outputs, f_new_states = _rnn(
            forward_step, inputs, forward_states, time_major)

        _, b_outputs, b_new_states = _rnn(
            backward_step, inputs, backward_states, time_major, True)

        outputs = (f_outputs, b_outputs)

        return outputs, f_new_states, b_new_states

    forward_states = [initial_states[0][0]]

    if cell_type == 'rnn_tanh':
        step_function = _rnn_cell_forward(*params, nonlinearity='tanh')
    elif cell_type == 'rnn_relu':
        step_function = _rnn_cell_forward(*params, nonlinearity='relu')
    elif cell_type == 'lstm':
        step_function = _lstm_cell_forward(*params)
        forward_states.append(initial_states[1][0])
    elif cell_type == 'gru':
        step_function = _gru_cell_forward(*params)
    else:
        raise ValueError('Invalid cell type', cell_type)

    _, outputs, new_states = _rnn(
        step_function, inputs, forward_states, time_major)

    return outputs, new_states, None

def recurrent(name):
    """ wrapper of various recurrent nn ops
    """
    def construct(*args):
        if name == "aten::lstm":
            inputs = args[0]
            hidden_states, cell_states = args[1]
            params = args[2]
            has_bias = args[3]
            num_layers = args[4]
            dropout_p = args[5]
            train = args[6]
            bidirectional = args[7]
            batch_first = args[8]

            """
            original criteria:
                1. activation == 'tanh'
                2. recurrent activation == 'sigmoid'
                3. dropout == 0
                4. not unroll
                5. use_bias
                6. reset_after
                7. ops.executing_eagerly_outside_functions()
            PyTorch uses activation as tanh and recurrent activation as sigmoid
            Moreover, PyTorch apply dropout between layers (no recurrent dropout)
            """
            could_use_cudnn = has_bias

            if could_use_cudnn:
                outputs, new_h, new_c = _cudnn_lstm(
                    inputs, hidden_states, cell_states,
                    params, not batch_first, dropout=dropout_p,
                    direction='bidirectional' if bidirectional else 'unidirectional'
                )
                return outputs, new_h, new_c

            layer_step = 2
            state_slice = 1
            if has_bias:
                layer_step = 4

            if bidirectional:
                layer_step *= 2
                state_slice = 2

            hiddens = []
            cells = []
            for i in range(num_layers):
                start = i * layer_step
                end = start + layer_step
                layer_params = params[start:end]

                outputs, f_states, b_states = recurrent_forward(
                    inputs, i, dropout_p,
                    (hidden_states[i:i+state_slice], cell_states[i:i+state_slice]),
                    'lstm', layer_params, not batch_first, train, bidirectional)


                f_h, f_c = f_states
                hiddens.append(f_h)
                cells.append(f_c)

                if bidirectional:
                    b_h, b_c = b_states
                    hiddens.append(b_h)
                    cells.append(b_c)

                    if i < num_layers - 1:
                        # reverse again (to be input)
                        outputs = (outputs[0],
                                   array_ops.reverse(outputs[1], [int(batch_first)]))
                    outputs = array_ops.concat(outputs, -1)

                inputs = outputs

            return outputs, array_ops.stack(hiddens), array_ops.stack(cells)
        elif name == "aten::rnn_tanh":
            inputs = args[0]
            hidden_states = args[1]
            params = args[2]
            has_bias = args[3]
            num_layers = args[4]
            dropout_p = args[5]
            train = args[6]
            bidirectional = args[7]
            batch_first = args[8]

            could_use_cudnn = has_bias

            if could_use_cudnn:
                outputs, new_h = _cudnn_rnn(
                    inputs, hidden_states,
                    params, not batch_first, dropout=dropout_p,
                    direction='bidirectional' if bidirectional else 'unidirectional',
                    nonlinearity='tanh')

                return outputs, new_h

            layer_step = 2
            state_slice = 1
            if has_bias:
                layer_step = 4

            if bidirectional:
                layer_step *= 2
                state_slice = 2

            hiddens = []
            for i in range(num_layers):
                start = i * layer_step
                end = start + layer_step
                layer_params = params[start:end]

                outputs, f_states, b_states = recurrent_forward(
                    inputs, i, dropout_p, (hidden_states[i:i+state_slice],),
                    'rnn_tanh', layer_params, not batch_first, train, bidirectional)

                f_h, = f_states
                hiddens.append(f_h)

                if bidirectional:
                    b_h, = b_states
                    hiddens.append(b_h)

                    if i < num_layers - 1:
                        # reverse again (to be input)
                        outputs = (outputs[0],
                                   array_ops.reverse(outputs[1], [int(batch_first)]))
                    outputs = array_ops.concat(outputs, -1)

                inputs = outputs

            return outputs, array_ops.stack(hiddens)
        elif name == "aten::rnn_relu":
            inputs = args[0]
            hidden_states = args[1]
            params = args[2]
            has_bias = args[3]
            num_layers = args[4]
            dropout_p = args[5]
            train = args[6]
            bidirectional = args[7]
            batch_first = args[8]

            could_use_cudnn = has_bias

            if could_use_cudnn:
                outputs, new_h = _cudnn_rnn(
                    inputs, hidden_states,
                    params, not batch_first, dropout=dropout_p,
                    direction='bidirectional' if bidirectional else 'unidirectional',
                    nonlinearity='relu')

                return outputs, new_h

            layer_step = 2
            state_slice = 1
            if has_bias:
                layer_step = 4

            if bidirectional:
                layer_step *= 2
                state_slice = 2

            hiddens = []
            for i in range(num_layers):
                start = i * layer_step
                end = start + layer_step
                layer_params = params[start:end]

                outputs, f_states, b_states = recurrent_forward(
                    inputs, i, dropout_p, (hidden_states[i:i+state_slice],),
                    'rnn_relu', layer_params, not batch_first, train, bidirectional)

                f_h, = f_states
                hiddens.append(f_h)

                if bidirectional:
                    b_h, = b_states
                    hiddens.append(b_h)

                    if i < num_layers - 1:
                        # reverse again (to be input)
                        outputs = (outputs[0],
                                   array_ops.reverse(outputs[1], [int(batch_first)]))
                    outputs = array_ops.concat(outputs, -1)

                inputs = outputs

            return outputs, array_ops.stack(hiddens)
        elif name == "aten::gru":
            inputs = args[0]
            hidden_states = args[1]
            params = args[2]
            has_bias = args[3]
            num_layers = args[4]
            dropout_p = args[5]
            train = args[6]
            bidirectional = args[7]
            batch_first = args[8]

            could_use_cudnn = has_bias

            if could_use_cudnn:
                outputs, new_h = _cudnn_gru(
                    inputs, hidden_states,
                    params, not batch_first, dropout=dropout_p,
                    direction='bidirectional' if bidirectional else 'unidirectional')

                return outputs, new_h

            layer_step = 2
            state_slice = 1
            if has_bias:
                layer_step = 4

            if bidirectional:
                layer_step *= 2
                state_slice = 2

            hiddens = []
            for i in range(num_layers):
                start = i * layer_step
                end = start + layer_step
                layer_params = params[start:end]

                outputs, f_states, b_states = recurrent_forward(
                    inputs, i, dropout_p, (hidden_states[i:i+state_slice],),
                    'gru', layer_params, not batch_first, train, bidirectional)

                f_h, = f_states
                hiddens.append(f_h)

                if bidirectional:
                    b_h, = b_states
                    hiddens.append(b_h)

                    if i < num_layers - 1:
                        # reverse again (to be input)
                        outputs = (outputs[0],
                                   array_ops.reverse(outputs[1], [int(batch_first)]))
                    outputs = array_ops.concat(outputs, -1)

                inputs = outputs

            return outputs, array_ops.stack(hiddens)
        else:
            raise ValueError("Not support", name)

    return construct
