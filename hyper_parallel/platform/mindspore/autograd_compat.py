"""MindSpore backward-style autograd compatibility helpers."""
# pylint: disable=protected-access,import-outside-toplevel

from __future__ import annotations

from mindspore._c_expression import TensorPy, pyboost_detach, run_backward
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
    return self._grad_node


@property
def output_nr(self):
    return self._output_index


def retain_grad(self):
    """Set the tensor retains gradient."""
    return self._retain_grad()


def detach(self):
    """Detach the tensor."""
    return pyboost_detach(self)


def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    if isinstance(inputs, list):
        inputs = tuple(inputs)
    if isinstance(gradient, list):
        gradient = tuple(gradient)
    return run_backward(
        (self,),
        gradient,
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
