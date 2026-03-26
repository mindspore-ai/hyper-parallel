# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py
# adapted for MindSpore Cell API.
# ============================================================================
"""Activation Checkpoint Wrapper implementation for MindSpore."""
# pylint: disable=W0613
from typing import Optional, Callable

from mindspore.nn import Cell

from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper


class CheckpointWrapper(ActivationWrapper):
    """
    Wrap a MindSpore :class:`~mindspore.nn.Cell` with activation recomputation
    (gradient checkpointing).

    On construction the wrapped cell is marked for recomputation via
    :meth:`Cell.recompute`, which is effective in semi-auto and
    auto-parallel graph-mode training on Ascend/GPU.

    When *checkpoint_fn* is supplied it is called in :meth:`construct`
    instead, which allows callers to inject a custom recompute strategy
    (e.g. selective activation checkpoint).  Any extra keyword arguments
    passed to the constructor are forwarded to *checkpoint_fn* at every
    forward call.

    Args:
        mod (Cell): The cell to wrap.
        checkpoint_fn (callable, optional): Custom checkpoint/recompute
            function with signature
            ``checkpoint_fn(cell, *args, **checkpoint_fn_kwargs, **kwargs)``.
            When ``None``, MindSpore's native :meth:`Cell.recompute` is used.
        **checkpoint_fn_kwargs: Extra keyword arguments forwarded to
            *checkpoint_fn* at every forward call.

    Example:
        >>> from hyper_parallel.platform.mindspore.activation_checkpoint import checkpoint_wrapper
        >>> wrapped = checkpoint_wrapper(my_cell)
        >>> output = wrapped(inputs)
    """

    def __init__(
        self,
        mod: Cell,
        checkpoint_fn: Optional[Callable] = None,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.checkpoint_fn = checkpoint_fn
        self.checkpoint_fn_kwargs = checkpoint_fn_kwargs
        if checkpoint_fn is None:
            # Use MindSpore's native recompute mechanism (effective in graph /
            # semi-auto parallel mode).
            self._ckpt_wrapped_module.recompute()

    def construct(self, *args, **kwargs):
        if self.checkpoint_fn is not None:
            return self.checkpoint_fn(
                self._ckpt_wrapped_module,
                *args,
                **self.checkpoint_fn_kwargs,
                **kwargs,
            )
        return self._ckpt_wrapped_module(*args, **kwargs)


def checkpoint_wrapper(
    module: Cell,
    checkpoint_fn: Optional[Callable] = None,
    **checkpoint_fn_kwargs,
) -> CheckpointWrapper:
    """
    Wrap *module* with activation recomputation (gradient checkpointing).

    This is the MindSpore counterpart of
    ``torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper``.

    Args:
        module (Cell): The cell to wrap.
        checkpoint_fn (callable, optional): Custom recompute function.  When
            ``None`` (default), MindSpore's native :meth:`Cell.recompute` is
            used.
        **checkpoint_fn_kwargs: Extra keyword arguments forwarded to
            *checkpoint_fn* on every forward call.

    Returns:
        CheckpointWrapper: The wrapped cell with activation recomputation
        enabled.

    Example:
        >>> from hyper_parallel.platform.mindspore.activation_checkpoint import checkpoint_wrapper
        >>> model.layers[i] = checkpoint_wrapper(model.layers[i])
    """
    return CheckpointWrapper(module, checkpoint_fn=checkpoint_fn, **checkpoint_fn_kwargs)
