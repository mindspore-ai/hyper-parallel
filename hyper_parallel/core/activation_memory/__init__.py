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
"""Activation checkpointing related interfaces (PyTorch backend only)."""
from .checkpoint import CheckpointError
from .pinned_memory_pool import PinnedMemoryPool
from .recompute_state import get_recompute_state, is_recomputing
from .swap import SwapManager
from .wrapper import (
    ActivationWrapper,
    AsyncSaveOnCpu,
    CheckpointExcludeWrapper,
    CheckpointWrapper,
    SwapWrapper,
    base_check_fn,
    ckpt_wrapper,
)
from .reentrant_checkpoint import (
    ReentrantCheckpointExcludeWrapper,
    ReentrantCheckpointWrapper,
    clear_reentrant_recompute_session,
    reentrant_checkpoint_exclude_wrapper,
    reentrant_checkpoint_wrapper,
    reentrant_recompute_handle_collector_ctx,
    reentrant_recompute_session_ctx,
)
from .api import (
    CheckpointPolicy,
    clear_recompute_session,
    checkpoint,
    checkpoint_exclude_wrapper,
    checkpoint_wrapper,
    recompute_handle,
    recompute_handle_collector_ctx,
    recompute_session_ctx,
    swap,
    swap_tensor_wrapper,
    swap_wrapper,
)

__all__ = [
    "ActivationWrapper",
    "AsyncSaveOnCpu",
    "CheckpointError",
    "CheckpointExcludeWrapper",
    "CheckpointPolicy",
    "CheckpointWrapper",
    "PinnedMemoryPool",
    "ReentrantCheckpointExcludeWrapper",
    "ReentrantCheckpointWrapper",
    "SwapManager",
    "SwapWrapper",
    "base_check_fn",
    "clear_recompute_session",
    "clear_reentrant_recompute_session",
    "checkpoint",
    "checkpoint_exclude_wrapper",
    "checkpoint_wrapper",
    "ckpt_wrapper",
    "get_recompute_state",
    "is_recomputing",
    "recompute_handle",
    "recompute_handle_collector_ctx",
    "recompute_session_ctx",
    "reentrant_checkpoint_exclude_wrapper",
    "reentrant_checkpoint_wrapper",
    "reentrant_recompute_handle_collector_ctx",
    "reentrant_recompute_session_ctx",
    "swap",
    "swap_tensor_wrapper",
    "swap_wrapper",
]
