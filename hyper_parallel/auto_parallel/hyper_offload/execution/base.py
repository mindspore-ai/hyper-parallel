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
"""Base executor definition."""

from __future__ import annotations

import logging
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)


class OpaqueRegionStart(torch.autograd.Function):
    """Autograd function to mark the start of an opaque region."""

    @staticmethod
    def forward(ctx: Any, executor: BaseExecutor, func_name: str, dummy: torch.Tensor, *inputs: Any) -> Any:
        """Forward pass for the start boundary."""
        ctx.executor = executor
        ctx.func_name = func_name
        return (dummy,) + inputs

    @staticmethod
    def backward(ctx: Any, grad_dummy: Any, *grad_inputs: Any) -> Any:  # pylint: disable=unused-argument
        """Backward pass for the start boundary."""
        executor = ctx.executor
        executor.exit_opaque_region()

        # Reconstruct the backward op using on_op_end.
        # Outputs of the backward op are grad_inputs.
        # func/args/kwargs were cached by on_op_begin in OpaqueRegionEnd.backward.
        # on_op_end already shadows via apply_shadows and caches the bindings,
        # so we can use its return value directly.
        grad_inputs = executor.on_op_end(grad_inputs)

        return (None, None, None) + tuple(grad_inputs)


class OpaqueRegionEnd(torch.autograd.Function):
    """Autograd function to mark the end of an opaque region."""

    @staticmethod
    def forward(ctx: Any, executor: BaseExecutor, func_name: str, dummy: torch.Tensor, *outputs: Any) -> Any:  # pylint: disable=unused-argument
        """Forward pass for the end boundary.

        Wraps raw tensors into :class:`ShadowTensor` **inside** the
        autograd boundary.  This is required for the autograd engine to
        correctly link tensor-subclass outputs into the computation graph
        (subclass fixup).  The wrapping happens here instead of in
        :meth:`execute_opaque_op` so that :meth:`on_op_end` always
        receives raw tensors regardless of execution path.
        """
        ctx.executor = executor
        ctx.func_name = func_name
        # 'outputs' is a flat tuple of tensors passed via .apply(*flat_res).
        # on_op_end has already been called and cached output_bindings
        # via apply_shadows; use them to avoid a redundant traversal.
        # pylint: disable=protected-access
        return tuple(executor.apply_shadows(outputs, executor._last_output_bindings))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Backward pass for the end boundary."""
        executor = ctx.executor

        def bwd_dummy(*_args: Any, **_kwargs: Any) -> Any:
            """Dummy backward function that records the op boundary."""

        bwd_dummy.__name__ = ctx.func_name + "_bwd"

        executor.on_op_begin(bwd_dummy, grad_outputs, {})

        executor.enter_opaque_region()
        return (None, None, torch.zeros(1)) + grad_outputs


class BaseExecutor(ABC):
    """Abstract base class for execution phases (warmup or replay).

    Executors implement phase-specific lifecycle callbacks. Raw PyTorch
    dispatch mechanics are handled by :class:`ActivationDispatchMode`.
    """

    def __init__(
        self,
        residency_manager: ResidencyManager,
    ) -> None:
        self.residency_manager = residency_manager
        #: sid -> WeakSet of alive shadows (used only for ``retained_sids``
        #: computation at the end of warmup.
        self._alive_shadows: dict[int, weakref.WeakSet[ShadowTensor]] = defaultdict(weakref.WeakSet)
        self.op_idx: int = -1
        self._opaque_depth: int = 0

        # Cached by on_op_begin for use in on_op_end.
        self._last_func = None
        self._last_args = None
        self._last_kwargs = None
        # Cached by apply_shadows for use in autograd boundaries.
        self._last_output_bindings: dict[int, int] | None = None

    @property
    def in_opaque_region(self) -> bool:
        """Return True if the executor is currently inside an opaque region."""
        return self._opaque_depth > 0

    def enter_opaque_region(self) -> None:
        """Enter an opaque region where fine-grained tracing is suspended."""
        self._opaque_depth += 1

    def exit_opaque_region(self) -> None:
        """Exit an opaque region."""
        self._opaque_depth -= 1

    def execute_opaque_op(self, func_name: str, fn: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute a function as a single virtual op.

        Wraps the function execution into a single "virtual op" in the
        execution trace, while suspending fine-grained tracing for
        internal operations.

        The lifecycle hook order is:

        1. :meth:`on_op_begin` — pre-actions.
        2. Opaque region (inner ops bypass lifecycle hooks).
        3. :meth:`on_op_end` — trace recording with **raw** tensors
           (consistent with :meth:`dispatch`).
        4. :meth:`OpaqueRegionEnd` — autograd boundary that wraps
           outputs into :class:`ShadowTensor` **inside** the autograd
           function (required for correct subclass graph linkage).

        Steps 3 and 4 happen **inside** the opaque region so that any
        incidental dispatch triggered by the autograd engine during
        :meth:`OpaqueRegionEnd.apply` does not invoke lifecycle hooks.
        """
        if self.in_opaque_region:
            return fn(*args, **kwargs)

        def fwd_dummy(*_a: Any, **_kw: Any) -> Any:
            """Dummy forward function that records the op boundary."""

        fwd_dummy.__name__ = func_name + "_fwd"

        self.enter_opaque_region()
        try:
            self.on_op_begin(fwd_dummy, args, kwargs)

            # Inject a dummy tensor to ensure backward graph continuity
            dummy = torch.zeros(1, requires_grad=True)
            flat_args, spec_args = tree_flatten((args, kwargs))

            # Boundary 1: Wrap inputs to delay backward virtual step exit
            out_start = OpaqueRegionStart.apply(self, func_name, dummy, *flat_args)
            dummy_out = out_start[0]
            flat_args_out = out_start[1:]

            args_out, kwargs_out = tree_unflatten(flat_args_out, spec_args)

            result = fn(*args_out, **kwargs_out)

            # Lifecycle: After op — record trace with RAW tensors.
            self.on_op_end(result)

            # Boundary 2: Wrap outputs inside autograd (shadowing must
            # happen inside the autograd Function for the engine to
            # correctly link ShadowTensor subclasses into the graph).
            flat_res, spec_res = tree_flatten(result)
            out_end = OpaqueRegionEnd.apply(self, func_name, dummy_out, *flat_res)
            result_unflat = tree_unflatten(out_end, spec_res)
        finally:
            self.exit_opaque_region()

        return result_unflat

    # ------------------------------------------------------------------
    # Dispatch (template method)
    # ------------------------------------------------------------------

    def dispatch(self, func, args, kwargs):
        """Dispatch *func* with *args*/*kwargs*.

        When the executor is inside an opaque region (e.g. executing the
        internals of a virtual op), *func* is called directly,
        **skipping** the lifecycle hooks
        (:meth:`on_op_begin` / :meth:`on_op_end`) since those
        boundaries are managed by ``OpaqueRegionStart`` / ``OpaqueRegionEnd``.

        Otherwise the standard slow-path is taken:

        1. :meth:`on_op_begin` — pre-actions (prefetch, etc.).
        2. ``func(*args, **kwargs)`` — raw execution.
        3. :meth:`on_op_end` — trace recording, post-actions **and**
           :class:`ShadowTensor` wrapping (single pass).
        """
        if self.in_opaque_region:
            return func(*args, **kwargs)

        self.on_op_begin(func, args, kwargs)
        result = func(*args, **kwargs)
        return self.on_op_end(result)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_op_begin(self, func, args, kwargs) -> None:
        """Lifecycle callback before the operator is executed.

        Caches *func*, *args*, *kwargs* so that :meth:`on_op_end`
        can access them via ``self._last_func`` etc.  Subclasses that
        override this method **must** call ``super().on_op_begin(...)``
        to maintain the cache.
        """
        self._last_func = func
        self._last_args = args
        self._last_kwargs = kwargs
        self.op_idx += 1

    @abstractmethod
    def on_op_end(self, result) -> Any:
        """Lifecycle callback after the operator is executed.

        Responsible for trace recording, phase-specific post-actions,
        **and** shadow wrapping.  Must return the (possibly shadowed)
        result tree so that :meth:`dispatch` can return it directly.

        The op's function and arguments are available via
        ``self._last_func``, ``self._last_args``, ``self._last_kwargs``
        (cached by :meth:`on_op_begin`).

        Subclasses should call :meth:`apply_shadows` with the bindings
        they have already computed (e.g. from the tracker or the guide)
        to avoid redundant traversal.
        """

    def apply_shadows(self, result: Any, bindings: dict[int, int]) -> Any:
        """Replace result tensors with ShadowTensor instances per an explicit bindings map.

        Subclasses that have already computed the ``leaf_index → sid``
        mapping (e.g. during trace recording) pass it here to avoid a
        redundant second traversal and SID re-resolution.

        Caches *bindings* in ``_last_output_bindings`` for use by autograd
        boundaries (:class:`OpaqueRegionEnd`) that need to shadow outputs
        after :meth:`on_op_end` has already returned.

        Args:
            result: The raw output pytree.
            bindings: Mapping ``leaf_index → storage_id``.  Only leaves
                whose index appears in the map are shadowed.

        Returns:
            A pytree of the same structure as *result* with eligible
            tensors replaced by :class:`ShadowTensor`.

        """
        leaves, tree_spec = tree_flatten(result)
        shadowed = list(leaves)
        for idx, leaf in enumerate(leaves):
            if idx in bindings and isinstance(leaf, torch.Tensor):
                shadowed[idx] = self.make_shadow(bindings[idx], leaf)
        result = tree_unflatten(shadowed, tree_spec)
        self._last_output_bindings = bindings
        return result

    # ------------------------------------------------------------------
    # Orchestration helper (output wrapping)
    # ------------------------------------------------------------------

    @property
    def retained_sids(self) -> set[int]:
        """Return the set of storage IDs that still have alive shadows."""
        return {sid for sid, shadows in self._alive_shadows.items() if shadows}

    def make_shadow(self, storage_id: int, tensor: torch.Tensor) -> Any:
        """Register physical storage, create shadow, and track logically.

        Composes:
        #. :meth:`ResidencyManager.bind` — physical registration.
        #. :class:`ShadowTensor` construction.
        #. Tracking in ``_alive_shadows`` for ``retained_sids``.

        Args:
            storage_id: Physical storage ID.
            tensor: The device-resident tensor to shadow.

        Returns:
            The original *tensor* replaced by a :class:`ShadowTensor`,
            or the same shadow updated in-place for mutation.

        """
        with torch.utils._python_dispatch._disable_current_modes():  # pylint: disable=protected-access
            if isinstance(tensor, ShadowTensor):
                return tensor

            buffer = self.residency_manager.bind(storage_id, tensor)
            shadow = ShadowTensor(tensor, buffer, storage_id)
            self._alive_shadows[storage_id].add(shadow)
            return shadow

    def reset(self) -> None:
        """Reset per-cycle state before a new pass."""
        self._opaque_depth = 0
        self.op_idx = -1
        self._alive_shadows.clear()
        self.residency_manager.clear_runtime()
