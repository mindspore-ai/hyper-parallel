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
"""
Hyper Parallel Checkpoint Package.

This package provides distributed checkpoint saving and loading capabilities,
including support for tensor sharding, resharding, and layout management.
"""

__all__ = [
    # Main API
    "save",
    "async_save",
    "AsyncSaveResponse",
    "load",
    # Metadata
    "Metadata",
    "MetadataIndex",
    "TensorStorageMetadata",
    "BytesStorageMetadata",
    "ChunkStorageMetadata",
    "TensorProperties",
    "CHUNK_INFO",
    "ChunkInfo",
    # Planner interfaces
    "SavePlanner",
    "LoadPlanner",
    "SavePlan",
    "LoadPlan",
    "WriteItem",
    "ReadItem",
    "WriteItemType",
    "LoadItemType",
    # Standard planners
    "StandardSavePlanner",
    "StandardLoadPlanner",
    # Storage interfaces
    "StorageWriter",
    "StorageReader",
    "StorageInfo",
    "WriteResult",
    # File system storage
    "FileSystemWriter",
    "FileSystemReader",
    # Layout I/O
    "get_current_layout",
    "save_layout",
    "load_layout",
    "combine_layout",
    "get_global_layout",
    # Base API
    "save_checkpoint",
    "load_checkpoint",
    # Resharding
    "ReshardHandler",
]

# Main API
from hyper_parallel.core.distributed_checkpoint.api import (
    AsyncSaveResponse,
    async_save,
    load,
    save,
)

# Metadata structures
from hyper_parallel.core.distributed_checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
    CHUNK_INFO,
    ChunkInfo
)

# Planner interfaces and data structures
from hyper_parallel.core.distributed_checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)

# Standard planner implementations
from hyper_parallel.core.distributed_checkpoint.standard_planner import (
    StandardLoadPlanner,
    StandardSavePlanner,
)

# Storage interfaces and data structures
from hyper_parallel.core.distributed_checkpoint.storage import (
    StorageInfo,
    StorageReader,
    StorageWriter,
    WriteResult,
)

# File system storage implementations
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import (
    FileSystemReader,
    FileSystemWriter,
)

# Layout I/O utilities
from hyper_parallel.core.distributed_checkpoint.layout import (
    combine_layout,
    get_current_layout,
    get_global_layout,
    load_layout,
    save_layout,
)

# Base API (backward compatibility)
from hyper_parallel.core.distributed_checkpoint.loader import load_checkpoint
from hyper_parallel.core.distributed_checkpoint.saver import save_checkpoint

# Resharding utilities
from hyper_parallel.core.distributed_checkpoint.reshard import ReshardHandler
