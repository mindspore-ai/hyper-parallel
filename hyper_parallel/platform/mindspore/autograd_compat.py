"""MindSpore backward-style autograd compatibility helpers."""
# pylint: disable=protected-access,import-outside-toplevel

from __future__ import annotations

import warnings

from mindspore import ops
from mindspore._c_expression import TensorPy, pyboost_detach, run_backward
from mindspore._c_expression import typing
from mindspore.graph.api import _pynative_executor

_BACKWARD_COMPAT_ENABLED = False


@property
def requires_grad(self):
    """Return whether the tensor requires gradient."""
    return self._requires_grad


@requires_grad.setter
def requires_grad(self, value=True):
    if not isinstance(value, bool):
        raise TypeError("The argument `requires_grad` must be bool type")
    self._requires_grad = value


@property
def grad(self):
    """Return the current accumulated gradient."""
    if not self.is_leaf and self.requires_grad:
        warnings.warn(
            "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. "
            "Its .grad attribute won't be populated during autograd.backward(). "
            "If you indeed want the .grad field to be populated for a non-leaf Tensor, "
            "use .retain_grad() on the non-leaf Tensor.",
            stacklevel=2,
        )
    dtensor_grad = getattr(self, "_dtensor_grad", None)
    if dtensor_grad is not None:
        return dtensor_grad
    return self._grad


@grad.setter
def grad(self, value):
    try:
        from hyper_parallel.core.dtensor.dtensor import DTensor
    except ImportError:
        DTensor = ()

    if value is None:
        self._dtensor_grad = None
        self._grad = None
        return

    if DTensor and isinstance(value, DTensor):
        self._dtensor_grad = value
        self._grad = value._local_tensor
        return

    self._dtensor_grad = None
    self._grad = value


@property
def is_leaf(self):
    """Return whether the tensor is a leaf."""
    return self._is_leaf


@property
def retains_grad(self):
    """Return whether the tensor retains gradient."""
    return self._retains_grad


@property
def grad_fn(self):
    if self._grad_node and self._grad_node.is_leaf():
        return None
    return self._grad_node


@property
def output_nr(self):
    return self._output_index


def retain_grad(self):
    """Set the tensor retains gradient."""
    return self._retain_grad()


def detach(self):
    """Detach the tensor."""
    detached = pyboost_detach(self)
    detached._dtensor_grad = None
    return detached


def _is_same_size(output, grad_tensor):
    return tuple(output.shape) == tuple(grad_tensor.shape)


def _calculate_shape(output, grad_tensor):
    return output.shape, grad_tensor.shape


def _tensor_or_tensors_to_tuple(tensors, length):
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, TensorPy):
        return (tensors,)
    return tuple(tensors)


def _make_grads(outputs, grads):
    """Validate backward gradients and materialize implicit scalar grads."""
    new_grads = []
    for index, (out, grad_tensor) in enumerate(zip(outputs, grads)):
        if isinstance(grad_tensor, TensorPy):
            if not _is_same_size(out, grad_tensor):
                out_shape, grad_shape = _calculate_shape(out, grad_tensor)
                raise RuntimeError(
                    "Mismatch in shape: grad_output["
                    + str(index)
                    + "] has a shape of "
                    + str(grad_shape)
                    + " and output["
                    + str(index)
                    + "] has a shape of "
                    + str(out_shape)
                    + "."
                )
            if out.dtype.is_complex != grad_tensor.dtype.is_complex:
                raise RuntimeError(
                    "For complex Tensors, both grad_output and output"
                    " are required to have the same dtype."
                    " Mismatch in dtype: grad_output["
                    + str(index)
                    + "] has a dtype of "
                    + str(grad_tensor.dtype)
                    + " and output["
                    + str(index)
                    + "] has a dtype of "
                    + str(out.dtype)
                    + "."
                )
            new_grads.append(grad_tensor)
        elif grad_tensor is None:
            if out.numel() != 1:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
            if not isinstance(out.dtype, (typing.Float, typing.BFloat)):
                raise RuntimeError(
                    f"grad can be implicitly created only for real scalar outputs but got {out.dtype}"
                )
            new_grads.append(ops.ones_like(out))
        else:
            raise TypeError(
                "gradients can be either Tensors or None, but got " + type(grad_tensor).__name__
            )
    return tuple(new_grads)


def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    """Run torch-style backward on a MindSpore tensor."""
    outputs = (self,)
    has_explicit_inputs = inputs is not None
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    elif isinstance(inputs, TensorPy):
        inputs = (inputs,)
    elif inputs is None:
        inputs = ()
    else:
        inputs = tuple(inputs)
    if has_explicit_inputs and len(inputs) == 0:
        raise RuntimeError("'inputs' argument to backward() cannot be empty.")

    grad_tensors = _tensor_or_tensors_to_tuple(gradient, len(outputs))
    grad_tensors = _make_grads(outputs, grad_tensors)
    if retain_graph is None:
        retain_graph = create_graph

    return run_backward(
        outputs,
        grad_tensors,
        retain_graph,
        create_graph,
        inputs,
        allow_unreachable=True,
        accumulate_grad=True,
    )


def enable_mindspore_backward_compat() -> None:
    """Enable torch-like ``Tensor.backward()`` semantics for MindSpore PyNative."""
    global _BACKWARD_COMPAT_ENABLED
    if _BACKWARD_COMPAT_ENABLED:
        return

    _pynative_executor.set_grad_flag(True)
    TensorPy.requires_grad = requires_grad
    TensorPy.grad = grad
    TensorPy.backward = backward
    TensorPy.is_leaf = is_leaf
    TensorPy.retains_grad = retains_grad
    TensorPy.retain_grad = retain_grad
    TensorPy.grad_fn = grad_fn
    TensorPy.output_nr = output_nr
    TensorPy.detach = detach
    _BACKWARD_COMPAT_ENABLED = True
