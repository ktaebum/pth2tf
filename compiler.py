"""
compile Pytorch IR Graph into Tensorflow Graph
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re

import torch

import tensorflow as tf

from converter.conversion import op_map
from converter.tensor_copy import copy_through_numpy
from converter.tensor_copy import build_args

# operation sets whose behavior differs btw train / inference
_NEED_TRAINING = {
    "aten::batch_norm",
    "aten::feature_dropout",
    "aten::dropout",
}

_TRAINING_HOLDER_NAME = "is_training"
_TRAINING_HOLDERS = {}

_SCOPE_VARIABLE_ID = {}

def get_training_placeholder(graph):
    """ get training flag placeholder of graph """
    return _TRAINING_HOLDERS[graph]

def _parse_dtype(node):
    """ get data type from node repr """
    # TODO get dtype from Node directly?
    match = re.findall(r': (.+?) =', repr(node))
    dtype = match[0]
    return dtype

def _safe_next(iterable):
    """ check target iterable object has next object
    return if it has next object
    return None if it does not have
    """
    try:
        return next(iterable)
    except StopIteration:
        return None

def convert(model, args):
    """ convert pytorch graph to tensorflow graph """
    tf.compat.v1.disable_eager_execution()

    graph, params = model.forward._lowered_graph()

    lookup_table = {}  # variable lookup

    tf_graph = tf.compat.v1.get_default_graph()

    with tf_graph.as_default():
        args = build_args(args)
        training_placeholder = _TRAINING_HOLDERS.setdefault(
            tf_graph,
            tf.compat.v1.placeholder_with_default(True, (),
                                                  _TRAINING_HOLDER_NAME))

        graph_inputs = graph.inputs()

        # store input shapes
        inputs = []
        for arg in args:
            inp = _safe_next(graph_inputs)
            if inp is None:
                raise ValueError
            inp_name = inp.debugName()
            lookup_table[inp_name] = arg
            inputs.append(arg)

        # store parameters
        for param in params:
            inp = _safe_next(graph_inputs)
            if inp is None:
                raise ValueError
            inp_name = inp.debugName()
            lookup_table[inp_name] = param

        assert _safe_next(graph_inputs) is None, "Didn't consume iterator all"

        # traverse graph nodes
        for node in graph.nodes():
            kind = node.kind()
            scope_name = re.sub(r"\[[^)]*\]", "", node.scopeName())
            extra_node_inputs = []
            if kind in _NEED_TRAINING:
                extra_node_inputs.append(training_placeholder)

            dtype = _parse_dtype(node)
            op_func = op_map(node)

            node_inputs = []
            for inp in node.inputs():
                inp_name = inp.debugName()
                inp_value = lookup_table[inp_name]

                if isinstance(inp_value, torch.Tensor):
                    # initialize tf variable in current scope
                    with tf.compat.v1.variable_scope(
                            scope_name,
                            reuse=tf.compat.v1.AUTO_REUSE):
                        var_id = _SCOPE_VARIABLE_ID.setdefault(scope_name, 0)
                        inp_value = copy_through_numpy(inp_value, var_id)
                        _SCOPE_VARIABLE_ID[scope_name] += 1

                    # update lookup_table with tf_variable
                    lookup_table[inp_name] = inp_value

                node_inputs.append(inp_value)

            if kind == "prim::Constant":
                for outp in node.outputs():
                    outp_name = outp.debugName()
                    ivalue = outp.toIValue()
                    if isinstance(ivalue, torch.Tensor):
                        lookup_table[outp_name] = ivalue.item()
                    else:
                        lookup_table[outp_name] = ivalue
                continue

            try:
                node_inputs = node_inputs + extra_node_inputs
                outputs = op_func(*node_inputs)
            except Exception as e:
                print("error in ", kind, node_inputs, extra_node_inputs)
                raise e

            if kind == "prim::ListUnpack":
                outputs = tuple(outputs)
            elif kind == "prim::TupleConstruct":
                outputs = (outputs,)

            if isinstance(outputs, tuple):
                for outp, tf_outp in zip(node.outputs(), outputs):
                    outp_name = outp.debugName()
                    lookup_table[outp_name] = tf_outp
            else:
                outp = next(node.outputs())
                outp_name = outp.debugName()
                lookup_table[outp_name] = outputs

        outputs = []
        for outp in graph.outputs():
            outp_name = outp.debugName()
            outp_var = lookup_table[outp_name]
            if isinstance(outp_var, tuple):
                for item in outp_var:
                    outputs.append(item)
            else:
                outputs.append(outp_var)

    return tf_graph, inputs, outputs
