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
"""Checkpoint persistence, discovery, and weight-layout conversion."""

from hyper_parallel.components.checkpoint.base import CheckpointerBase
from hyper_parallel.components.checkpoint.config import CheckpointingConfig
from hyper_parallel.components.checkpoint.conversion_ops import (
    AddScalar,
    ConcatenateWithSections,
    DeinterleaveGateQKV,
    DeinterleaveQKV,
    InterleaveGateQKV,
    InterleaveQKV,
    Split,
)
from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    STEP_PREFIX,
    DistributedCheckpointer,
    initialize_optimizer_state,
)
from hyper_parallel.components.checkpoint.huggingface_checkpointer import (
    HuggingFaceCheckpointer,
    resolve_hf_loader,
)
from hyper_parallel.components.checkpoint.huggingface_load_planner import (
    HFLoadPlanner,
    load_hf_checkpoint,
)
from hyper_parallel.components.checkpoint.hf_storage import (
    HuggingFaceStorageReader,
    TorchShardedSafetensorsReader,
)
from hyper_parallel.components.checkpoint.quantized_hf_storage import QuantizedHuggingFaceStorageReader
from hyper_parallel.components.checkpoint.remap_planner import DeferredRead, RemapBlock, RemapLoadPlanner
from hyper_parallel.components.checkpoint.load_groups import LoadReport
from hyper_parallel.components.checkpoint.registry import (
    CHECKPOINTER_REGISTRY,
    build_checkpointer,
)

__all__ = [
    "AddScalar",
    "CHECKPOINTER_REGISTRY",
    "CheckpointerBase",
    "CheckpointingConfig",
    "ConcatenateWithSections",
    "DeinterleaveGateQKV",
    "DeinterleaveQKV",
    "DistributedCheckpointer",
    "DeferredRead",
    "HFLoadPlanner",
    "HuggingFaceCheckpointer",
    "HuggingFaceStorageReader",
    "InterleaveGateQKV",
    "InterleaveQKV",
    "LoadReport",
    "QuantizedHuggingFaceStorageReader",
    "RemapBlock",
    "RemapLoadPlanner",
    "STEP_PREFIX",
    "Split",
    "TorchShardedSafetensorsReader",
    "build_checkpointer",
    "initialize_optimizer_state",
    "load_hf_checkpoint",
    "resolve_hf_loader",
]
