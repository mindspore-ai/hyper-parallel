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


def checkpoint(function, *args, swap_inputs=False, policy_fn=None, **kwargs):
    """
    Apply activation checkpointing to a function with optional input swapping.
    
    Args:
        function: The function to apply checkpointing to.
        *args: Arguments to pass to the function.
        swap_inputs (bool): Whether to enable input swapping using async_save_on_cpu context.
        policy_fn (callable, optional): Function that determines checkpoint policy for operations.
        **kwargs: Additional keyword arguments to pass to the function.
        
    Returns:
        The result of applying the function with checkpointing.
    """
    context_fn = partial(plat.create_selective_checkpoint_contexts, policy_fn) if policy_fn else plat.noop_context_fn
    context = plat.async_save_on_cpu if swap_inputs else contextlib.nullcontext
    with context():
        return plat.checkpoint(function, *args, context_fn=context_fn, use_reentrant=False, **kwargs)


checkpoint_wrapper = partial(plat.ckpt_wrapper, checkpoint_fn=checkpoint)
