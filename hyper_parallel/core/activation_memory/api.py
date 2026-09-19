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
from functools import partial
from typing import Any, Callable, Optional, Tuple

import torch

from .policy import CheckpointPolicy
from .recompute_state import create_recompute_contexts

__all__ = [
    "CheckpointPolicy",
    "checkpoint",
    "swap",
    "checkpoint_wrapper",
    "checkpoint_exclude_wrapper",
    "swap_wrapper",
    "swap_tensor_wrapper",
    "recompute_handle_collector_ctx",
    "recompute_handle",
    "recompute_session_ctx",
    "clear_recompute_session",
    "ignore_sac_ops",
    "create_selective_checkpoint_contexts",
    "create_native_selective_checkpoint_contexts",
    "async_save_on_cpu",
    "get_class_activation_wrapper",
    "noop_context_fn",
    "is_compiling",
]


class _StackedCtx:
    """Compose multiple context managers as one — enter in order, exit reversed."""

    def __init__(self, ctxs) -> None:
        self._ctxs = list(ctxs)
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        try:
            for ctx in self._ctxs:
                self._stack.enter_context(ctx)
        except BaseException as exc:
            self._stack.__exit__(type(exc), exc, exc.__traceback__)
            raise
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


def is_compiling() -> bool:
    """Return whether the caller runs inside a ``torch.compile`` capture."""
    return torch.compiler.is_compiling()


def noop_context_fn():
    """The Torch-native no-op ``context_fn``, for callers that need an explicit one."""
    # pylint: disable=C0415
    from torch.utils.checkpoint import noop_context_fn as _torch_noop_context_fn
    return _torch_noop_context_fn


def checkpoint_exclude_wrapper(module: Any, *, save_output: bool = True) -> Any:
    """Wrap a callable whose region is excluded from activation recomputation.

    Args:
        module: The module or callable to exclude from recomputation.
        save_output: Whether to retain the region output for checkpoint replay.
            Set this to ``False`` only when the output is passed directly as one
            argument to another excluded region. Default: ``True``.

    Returns:
        The checkpoint exclusion wrapper.
    """
    # pylint: disable=C0415
    from .wrapper import checkpoint_exclude_wrapper as _wrapper
    return _wrapper(module, save_output=save_output)


def checkpoint_wrapper(module, **checkpoint_kwargs):
    """Wrap *module* with HyperParallel activation checkpointing."""
    # pylint: disable=C0415
    from .wrapper import ckpt_wrapper
    return ckpt_wrapper(module, **checkpoint_kwargs)


def swap_wrapper(module, policy_fn=None, group_swap=False, cpu_pool=None):
    """Wrap *module* so its activations are offloaded to host memory.

    Args:
        module: Module or callable to wrap.
        policy_fn: Optional per-tensor swap policy.
        group_swap: Whether tensors participate in group copy fusion.
        cpu_pool: Optional pinned host memory pool for the swapped tensors.
            Only supported by the Torch backend, so it is forwarded only when set.

    Returns:
        The configured swap wrapper.
    """
    # pylint: disable=C0415
    from .wrapper import swap_wrapper as _swap_wrapper
    kwargs = {}
    if cpu_pool is not None:
        kwargs["cpu_pool"] = cpu_pool
    return _swap_wrapper(module, policy_fn=policy_fn, group_swap=group_swap, **kwargs)


def swap_tensor_wrapper(target, tag=None, group_swap=False, cpu_pool=None):
    """Wrap a single tensor or callable for activation swap.

    Args:
        target: Tensor or nested tensor structure to register.
        tag: Optional debug tag.
        group_swap: Whether tensors participate in group copy fusion.
        cpu_pool: Optional pinned host memory pool for the swapped tensors.
            Only supported by the Torch backend, so it is forwarded only when set.

    Returns:
        The registered tensor structure.
    """
    # pylint: disable=C0415
    from .wrapper import swap_tensor_wrapper as _swap_tensor_wrapper
    kwargs = {}
    if cpu_pool is not None:
        kwargs["cpu_pool"] = cpu_pool
    return _swap_tensor_wrapper(target, tag=tag, group_swap=group_swap, **kwargs)


def get_class_activation_wrapper():
    """Return the :class:`ActivationWrapper` base class of this backend."""
    # pylint: disable=C0415
    from .wrapper import ActivationWrapper
    return ActivationWrapper


def ignore_sac_ops(ignore_ops: list) -> None:
    """Exclude backend operators from selective-AC replay accounting."""
    # pylint: disable=C0415
    from .sac import ignore_sac_ops as _ignore_sac_ops
    _ignore_sac_ops(ignore_ops)


def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False,
                                         group_swap=False, cpu_pool=None):
    """Create HyperParallel selective-checkpoint ``(forward_ctx, recompute_ctx)`` contexts."""
    # pylint: disable=C0415
    from .sac import create_selective_checkpoint_contexts as _create_contexts
    return _create_contexts(
        policy_fn_or_list,
        allow_cache_entry_mutation=allow_cache_entry_mutation,
        group_swap=group_swap,
        cpu_pool=cpu_pool,
    )


def create_native_selective_checkpoint_contexts(policy_fn):
    """Create Torch-native selective-checkpoint contexts for compile capture."""
    # pylint: disable=C0415
    from .compile_adapter import create_native_selective_checkpoint_contexts as _create_contexts
    return _create_contexts(policy_fn)


def async_save_on_cpu(policy_fn=None, group_swap: bool = False, cpu_pool=None):
    """Return the async host-offload context used by :func:`swap` and ``swap_inputs``."""
    # pylint: disable=C0415
    from .wrapper import AsyncSaveOnCpu
    return AsyncSaveOnCpu(policy_fn=policy_fn, group_swap=group_swap, cpu_pool=cpu_pool)


def recompute_handle_collector_ctx():
    """Return the context that collects recompute handles for a checkpoint region."""
    # pylint: disable=C0415
    from .checkpoint import recompute_handle_collector_ctx as _collector_ctx
    return _collector_ctx()


def recompute_handle(handle, session_id):
    """Recompute the region behind *handle* under *session_id*."""
    # pylint: disable=C0415
    from .checkpoint import recompute_handle as _recompute_handle
    return _recompute_handle(handle, session_id)


def recompute_session_ctx(session_id, retain_on_unpack=False):
    """Open a recompute session so replay can unpack saved activations."""
    if session_id is None:
        raise ValueError("session_id must not be None.")
    # pylint: disable=C0415
    from .checkpoint import recompute_session_ctx as _session_ctx
    return _session_ctx(session_id=session_id, retain_on_unpack=retain_on_unpack)


def clear_recompute_session(session_id):
    """Release all state held for *session_id*."""
    # pylint: disable=C0415
    from .checkpoint import clear_recompute_session as _clear
    return _clear(session_id)


def _validate_compile_checkpoint_options(
    swap_inputs: bool,
    group_swap: bool,
    cpu_pool: Any,
    context_fn: Optional[Callable],
    use_reentrant: bool,
) -> None:
    """Reject eager-only checkpoint options before compile capture."""
    unsupported = []
    if swap_inputs:
        unsupported.append("swap_inputs")
    if group_swap:
        unsupported.append("group_swap")
    if cpu_pool is not None:
        unsupported.append("cpu_pool")
    if context_fn is not None:
        unsupported.append("custom context_fn")
    if use_reentrant:
        unsupported.append("use_reentrant=True")
    if unsupported:
        raise ValueError(
            "HyperParallel checkpoint compile mode does not support: "
            + ", ".join(unsupported)
            + ". Use Torch-native non-reentrant checkpointing with optional "
            "SAVE/RECOMPUTE selective policies."
        )


def _make_checkpoint_context_fn(
    policy_fn: Optional[Callable],
    context_fn: Optional[Callable],
    group_swap: bool,
    cpu_pool: Any,
) -> Callable:
    """Compose eager recompute, selective, and caller-provided contexts."""
    factories: list = [create_recompute_contexts]
    if policy_fn is not None:
        selective_kwargs = {"group_swap": group_swap}
        if cpu_pool is not None:
            selective_kwargs["cpu_pool"] = cpu_pool
        factories.append(partial(create_selective_checkpoint_contexts, policy_fn, **selective_kwargs))
    if context_fn is not None:
        factories.append(context_fn)
    if len(factories) == 1:
        return factories[0]
    return _compose_context_fns(tuple(factories))


def _checkpoint_input_context(swap_inputs: bool, group_swap: bool, cpu_pool: Any) -> Any:
    """Create the optional context that offloads checkpoint inputs."""
    if not swap_inputs:
        return contextlib.nullcontext()
    async_kwargs = {"group_swap": group_swap}
    if cpu_pool is not None:
        async_kwargs["cpu_pool"] = cpu_pool
    return async_save_on_cpu(**async_kwargs)


def checkpoint(
    function,
    *args,
    swap_inputs: bool = False,
    policy_fn: Optional[Callable] = None,
    context_fn: Optional[Callable[[], Tuple[object, object]]] = None,
    group_swap: bool = False,
    cpu_pool=None,
    early_stop: bool = True,
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
        cpu_pool (PinnedMemoryPool, optional): Explicit pinned host memory pool used by tensors
            selected for swapping. Currently supported only by the Torch backend. Default: ``None``.
        early_stop (bool, optional): Whether recomputation stops after all tensors needed by
            backward have been produced. This per-call keyword is the only supported way to
            configure early stop. Default: ``True``.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The result of applying the function with checkpointing.
    """
    if not isinstance(early_stop, bool):
        raise ValueError(f"early_stop must be bool, but got {type(early_stop).__name__}.")

    if is_compiling():
        _validate_compile_checkpoint_options(
            swap_inputs, group_swap, cpu_pool, context_fn, kwargs.get("use_reentrant", False)
        )
        composed_context_fn = (
            partial(create_native_selective_checkpoint_contexts, policy_fn)
            if policy_fn is not None
            else None
        )
    else:
        composed_context_fn = _make_checkpoint_context_fn(policy_fn, context_fn, group_swap, cpu_pool)

    with _checkpoint_input_context(swap_inputs, group_swap, cpu_pool):
        checkpoint_kwargs = {**kwargs, "use_reentrant": False, "early_stop": early_stop}
        if composed_context_fn is not None:
            checkpoint_kwargs["context_fn"] = composed_context_fn
        # pylint: disable=C0415
        from .checkpoint import checkpoint as hyper_checkpoint
        return hyper_checkpoint(function, *args, **checkpoint_kwargs)


def swap(function, *args, policy_fn=None, group_swap=False, cpu_pool=None, **kwargs):
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
        cpu_pool (PinnedMemoryPool, optional): Explicit pinned host memory pool used by swapped
            tensors. Currently supported only by the Torch backend. Default: ``None``.
        **kwargs: Keyword arguments forwarded to *function*.

    Returns:
        The return value of ``function(*args, **kwargs)``.

    Example:
        >>> output = swap(layer, x, policy_fn=lambda t: CheckpointPolicy.MUST_SAVE)
    """
    if is_compiling():
        raise ValueError(
            "HyperParallel activation swap is not supported in compile mode. "
            "Use Torch-native non-reentrant checkpointing with SAVE/RECOMPUTE policies."
        )
    async_kwargs = {"policy_fn": policy_fn, "group_swap": group_swap}
    if cpu_pool is not None:
        async_kwargs["cpu_pool"] = cpu_pool
    with async_save_on_cpu(**async_kwargs):
        return function(*args, **kwargs)
