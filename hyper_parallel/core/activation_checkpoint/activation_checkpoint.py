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
from typing import Any

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


def checkpoint(function, *args, swap_inputs=False, policy_fn=None, group_swap=False, **kwargs):
    """
    Apply activation checkpointing to a function with optional input swapping.

    Args:
        function: The function to apply checkpointing to.
        *args: Arguments to pass to the function.
        swap_inputs (bool): Whether to enable input swapping using async_save_on_cpu context.
        policy_fn (callable, optional): Function that determines checkpoint policy for operations.
        group_swap (bool, optional): Whether MUST_SWAP tensors participate in group copy fusion. Default: ``False``.
        **kwargs: Additional keyword arguments to pass to the function.

    Returns:
        The result of applying the function with checkpointing.
    """
    context_fn = (
        partial(plat.create_selective_checkpoint_contexts, policy_fn, group_swap=group_swap)
        if policy_fn or group_swap else plat.noop_context_fn
    )
    context = plat.async_save_on_cpu if swap_inputs else contextlib.nullcontext
    with context():
        return plat.checkpoint(function, *args, context_fn=context_fn, use_reentrant=False, **kwargs)


def swap(function, *args, policy_fn=None, **kwargs):
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
        **kwargs: Keyword arguments forwarded to *function*.

    Returns:
        The return value of ``function(*args, **kwargs)``.

    Example:
        >>> output = swap(layer, x, policy_fn=lambda t: CheckpointPolicy.MUST_SAVE)
    """
    with plat.async_save_on_cpu(policy_fn=policy_fn):
        return function(*args, **kwargs)


class CheckpointWrapper(plat.get_class_activation_wrapper()):
    """
    Wrap a module with activation checkpointing.

    On each forward call the wrapped module is executed under
    :func:`~hyper_parallel.core.activation_checkpoint.checkpoint`, which
    applies the platform-native recompute primitive
    (``torch.utils.checkpoint.checkpoint`` for PyTorch).

    Args:
        mod (Any): The module or callable to wrap.
        group_swap (bool, optional): Whether ``MUST_SWAP`` tensors participate in
            group copy fusion.  Defaults to ``False``.
        **checkpoint_kwargs: Extra keyword arguments forwarded to
            :func:`~hyper_parallel.core.activation_checkpoint.checkpoint`
            on every forward call (e.g. ``policy_fn``).

    Example:
        >>> from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
        >>> wrapped = checkpoint_wrapper(my_module, group_swap=True)
        >>> output = wrapped(inputs)
    """

    def __init__(
        self,
        mod: Any,
        group_swap: bool = False,
        **checkpoint_kwargs: Any,
    ):
        super().__init__(mod)
        self.group_swap = group_swap
        self.checkpoint_kwargs = checkpoint_kwargs

    def _do_checkpoint(self, wrapped_module: Any, *args: Any, **kwargs: Any) -> Any:
        # Checkpoint may save inputs before the wrapped cell's pre-hook runs.
        group_name = getattr(self, "_swap_group_name", None)
        if group_name is not None:
            from hyper_parallel.core.activation_checkpoint.swap import SwapManager  # pylint: disable=C0415
            SwapManager().set_current_group_name(group_name)
        return checkpoint(
            wrapped_module,
            *args,
            group_swap=self.group_swap,
            **self.checkpoint_kwargs,
            **kwargs,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._do_checkpoint(self._wrapped_module, *args, **kwargs)

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        return self._do_checkpoint(self._wrapped_module, *args, **kwargs)


def ckpt_wrapper(module, group_swap: bool = False, **checkpoint_kwargs):
    """Wrap *module* with activation checkpointing.

    Args:
        module (Any): The module or callable to wrap.
        group_swap (bool, optional): Whether ``MUST_SWAP`` tensors participate in group
            copy fusion.  Forwarded to
            :func:`~hyper_parallel.core.activation_checkpoint.checkpoint`
            on every forward call. Default: ``False``.
        **checkpoint_kwargs: Extra keyword arguments forwarded to
            :func:`~hyper_parallel.core.activation_checkpoint.checkpoint`
            on every forward call (e.g. ``policy_fn``).

    Returns:
        A platform-specific ``CheckpointWrapper`` instance.

    Example:
        >>> from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper
        >>> model.layers[i] = checkpoint_wrapper(model.layers[i])
        >>> wrapped_fn = checkpoint_wrapper(lambda x: x * 2, group_swap=True)
    """
    return CheckpointWrapper(module, group_swap=group_swap, **checkpoint_kwargs)


checkpoint_wrapper = ckpt_wrapper
swap_wrapper = plat.swap_wrapper
swap_tensor_wrapper = plat.swap_tensor_wrapper
