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
"""Activation checkpointing related interfaces"""
import contextlib
import enum
from functools import partial
from typing import Callable, Optional, Tuple

from hyper_parallel.platform import get_platform
plat = get_platform()


class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    This enum extends PyTorch's selective activation checkpointing policies
    by introducing a SWAP-based strategy, which allows activation tensors
    to be offloaded during the forward pass and loaded back before backward
    computation.

    For PyTorch native policies (SAVE / RECOMPUTE semantics and MUST vs PREFER),
    see: https://docs.pytorch.org/docs/2.6/checkpoint.html#torch.utils.checkpoint.CheckpointPolicy

    Additional policy:

    - ``MUST_SWAP``: The operation's output is offloaded to host memory during the
      forward pass and loaded back asynchronously before backward computation. The backward
      pass reuses the loaded activations without recomputation.

      This policy must be used together with :class:`SwapManager` to coordinate
      asynchronous offload/load and stream synchronization.

    .. note::
        ``MUST_SWAP`` is typically applied to operations that are either
        computationally expensive or have large memory footprints. Note that
        swapping very small outputs may introduce additional overhead and
        reduce the effectiveness of asynchronous copy.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3

    # Offload during forward, reload before backward. Requires SwapManager.
    MUST_SWAP = 4


class _StackedCtx:
    """Compose multiple context managers as one — enter in order, exit reversed."""

    def __init__(self, ctxs) -> None:
        self._ctxs = list(ctxs)
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        for ctx in self._ctxs:
            self._stack.enter_context(ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._stack.__exit__(exc_type, exc_val, exc_tb)


def _compose_context_fns(
    factories: Tuple[Callable[[], Tuple[object, object]], ...],
) -> Callable[[], Tuple[_StackedCtx, _StackedCtx]]:
    """Combine ``(forward_ctx, recompute_ctx)`` factories into one factory.

    ``ms.recompute`` / ``torch.utils.checkpoint(use_reentrant=False)`` call
    ``context_fn()`` once per invocation and unpack the result as
    ``(forward_ctx, recompute_ctx)``.  This helper calls each input factory
    once, then stacks all forward contexts and all recompute contexts into
    two :class:`_StackedCtx` instances so the composite respects the
    single-call contract.
    """
    def factory() -> Tuple[_StackedCtx, _StackedCtx]:
        pairs = [fn() for fn in factories]
        fwd_ctxs = [pair[0] for pair in pairs]
        rec_ctxs = [pair[1] for pair in pairs]
        return _StackedCtx(fwd_ctxs), _StackedCtx(rec_ctxs)

    return factory


def checkpoint(
    function,
    *args,
    swap_inputs: bool = False,
    policy_fn: Optional[Callable] = None,
    context_fn: Optional[Callable[[], Tuple[object, object]]] = None,
    group_swap: bool = False,
    **kwargs,
):
    """
    Apply activation checkpointing to a function with optional input swapping.

    Args:
        function: The function to apply checkpointing to.
        *args: Arguments to pass to the function.
        swap_inputs (bool): Whether to enable input swapping using async_save_on_cpu context.
        policy_fn (callable, optional): Function that determines checkpoint policy for operations.
        context_fn (callable, optional): A no-arg factory returning a
            ``(forward_ctx, recompute_ctx)`` pair, matching the
            ``context_fn`` contract of ``ms.recompute(use_reentrant=False)``
            and ``torch.utils.checkpoint(use_reentrant=False)``.  Use this
            to bracket the backward-time forward re-run with custom logic.
            When ``policy_fn``, ``group_swap`` and ``context_fn`` are
            supplied together, the resulting factories are composed: their
            forward and recompute contexts are stacked so all enter in
            order and exit in reverse.
        group_swap (bool, optional): Whether MUST_SWAP tensors participate in group copy fusion.
            Only effective when ``policy_fn`` is provided. Default: ``False``.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The result of applying the function with checkpointing.
    """
    factories: list = []
    if policy_fn is not None:
        factories.append(partial(plat.create_selective_checkpoint_contexts, policy_fn, group_swap=group_swap))
    if context_fn is not None:
        factories.append(context_fn)

    if not factories:
        composed_context_fn = plat.noop_context_fn
    elif len(factories) == 1:
        composed_context_fn = factories[0]
    else:
        composed_context_fn = _compose_context_fns(tuple(factories))

    context = partial(plat.async_save_on_cpu, group_swap=group_swap) if swap_inputs else contextlib.nullcontext
    with context():
        return plat.checkpoint(
            function, *args, context_fn=composed_context_fn, use_reentrant=False, **kwargs
        )


def swap(function, *args, policy_fn=None, group_swap=False, **kwargs):
    """Apply activation swap to a function call.

    Offloads intermediate activations saved by the autograd engine to CPU
    during the forward pass and loads them back before the backward pass,
    trading device memory for host memory bandwidth.  Unlike
    :func:`checkpoint`, no recomputation is performed.

    Args:
        function (callable): The function whose activations should be swapped.
        *args: Positional arguments forwarded to *function*.
        policy_fn (callable, optional): Per-tensor swap policy.  Receives
            a tensor and returns a :class:`CheckpointPolicy` value.  Tensors
            that return ``CheckpointPolicy.MUST_SAVE`` are kept on device;
            all other eligible tensors are offloaded.  When ``None``, all
            eligible tensors are offloaded.
        group_swap (bool, optional): Whether swapped tensors participate in
            group copy fusion.  Default: ``False``.
        **kwargs: Keyword arguments forwarded to *function*.

    Returns:
        The return value of ``function(*args, **kwargs)``.

    Example:
        >>> output = swap(layer, x, policy_fn=lambda t: CheckpointPolicy.MUST_SAVE)
    """
    with plat.async_save_on_cpu(policy_fn=policy_fn, group_swap=group_swap):
        return function(*args, **kwargs)


checkpoint_wrapper = plat.checkpoint_wrapper
swap_wrapper = plat.swap_wrapper
swap_tensor_wrapper = plat.swap_tensor_wrapper
