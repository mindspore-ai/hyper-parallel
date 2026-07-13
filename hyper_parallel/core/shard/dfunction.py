# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Custom distributed autograd function base class."""
from __future__ import annotations

from hyper_parallel.platform import get_platform
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER

platform = get_platform()


class _LocalCallable:
    """Named callable wrapper that exposes the op name to both platform dispatchers.

    PyTorch's ``get_op_name`` inspects ``__name__``; MindSpore's inspects ``.name``.
    Setting both attributes here lets ``_OP_DISPATCHER`` look up the correct
    ``DistributedOp`` without modifying either platform implementation.

    Args:
        fn: The underlying callable to invoke.
        op_name: Canonical op name matching the registered ``DistributedOp``.
    """

    def __init__(self, fn: callable, op_name: str) -> None:
        self._fn = fn
        self.__name__ = op_name
        self.name = op_name

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class DFunction(platform.Function):
    """Base class for user-defined distributed autograd functions.

    Subclass this class and implement ``forward`` and ``backward`` as
    ``@staticmethod`` methods that operate on **local** tensors.  To enable
    multi-device support, also set ``_op_name`` to a string that matches the
    ``op_name`` of a registered ``DistributedOp`` subclass.

    Dispatch behaviour:

    * **No DTensor inputs** — calls ``super().apply()`` directly, going straight
      into the platform autograd mechanism (single-device path).
    * **DTensor inputs + ``_op_name`` set** — delegates to
      ``_OP_DISPATCHER.dispatch``.  The dispatcher extracts local tensors, calls
      ``DistributedOp.infer_layout`` to derive the output layout, invokes the
      local callable (which re-enters ``apply`` with plain tensors, triggering
      the single-device path), and wraps the result as a ``DTensor``.

    Note on non-tensor positional arguments:
        Non-tensor positional arguments are preserved via the three-phase dispatch
        flow (``preprocess()`` → ``infer_layout(cache_values)`` →
        ``get_expand_impl``).  Implement ``preprocess()`` in your ``DistributedOp``
        to collect non-tensor values into ``cache_values``.

    Example::

        class MyAddDistOp(DistributedOp):
            def __init__(self):
                super().__init__("MyAdd")

            def infer_layout(self, layouts, extra_args=None):
                return layouts[0]

        MyAddDistOp()  # instantiation triggers registration

        class MyAdd(DFunction):
            _op_name = "MyAdd"

            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x + y

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        result = MyAdd.apply(x, y)             # plain tensors → single-device
        result = MyAdd.apply(dtensor_x, dtensor_y)  # DTensors → distributed
    """

    _op_name: str = None

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Override in subclass to define the forward computation on local tensors."""
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Override in subclass to define the backward computation on local tensors."""
        raise NotImplementedError("Subclasses must implement backward()")

    @classmethod
    def apply(cls, *args, **kwargs):
        """Execute the function, routing to distributed dispatch when DTensor inputs are present.

        Args:
            *args: Positional arguments forwarded to ``forward``.  May mix
                ``DTensor`` and plain ``Tensor``.
            **kwargs: Keyword arguments forwarded to ``forward``.

        Returns:
            Output of the forward computation.  Wrapped as ``DTensor`` when the
            distributed dispatch path is taken and the ``DistributedOp`` infers a
            valid output layout.

        Raises:
            ValueError: If ``DTensor`` inputs are detected but ``_op_name`` is not
                set on the subclass.
        """
        has_dtensor = any(isinstance(a, DTensor) for a in args)
        if has_dtensor:
            if cls._op_name is None:
                raise ValueError(
                    f"{cls.__name__} received DTensor inputs but '_op_name' is not set. "
                    "Set '_op_name' on the subclass and register a matching DistributedOp."
                )
            return _OP_DISPATCHER.dispatch(cls._get_local_callable(), args, kwargs)
        return super().apply(*args, **kwargs)

    @classmethod
    def _get_local_callable(cls) -> _LocalCallable:
        """Return (and lazily create) the named local callable for this subclass.

        The callable re-enters ``cls.apply`` with plain tensors so the single-device
        autograd path is taken without recursion.  Cached per subclass in
        ``cls.__dict__`` to avoid repeated object creation.

        Returns:
            A ``_LocalCallable`` whose ``__name__`` and ``.name`` equal
            ``cls._op_name``, enabling ``platform.get_op_name`` to resolve the
            correct ``DistributedOp``.
        """
        if '_local_callable' not in cls.__dict__:
            def _local_fn(*a, **kw):
                # Called by _OP_DISPATCHER with extracted local (non-DTensor) tensors.
                # has_dtensor will be False here, so super().apply() is taken directly.
                return cls.apply(*a, **kw)

            cls._local_callable = _LocalCallable(_local_fn, cls._op_name)
        return cls._local_callable
