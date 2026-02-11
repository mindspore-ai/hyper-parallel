# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Checkpoint metadata structures for distributed checkpoint save and load."""
from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass(frozen=True)
class MetadataIndex:
    """
    Index to identify a specific piece of data in the checkpoint.

    Attributes:
        fqn: Fully qualified name of the tensor/object.
        offset: Offset in the tensor (for sharded tensors). Default ().
        index: Index for sharded tensors (None for non-sharded). Default None.
    """
    fqn: str
    offset: tuple = field(default_factory=tuple)
    index: Optional[int] = None


@dataclass(frozen=True)
class ChunkStorageMetadata:
    """
    Metadata for a chunk of storage.

    Represents a portion of a distributed tensor stored in the checkpoint.

    Attributes:
        offsets: Offsets in the global tensor for each dimension.
        sizes: Sizes of the chunk for each dimension.
    """
    offsets: tuple
    sizes: tuple


@dataclass(frozen=True)
class TensorProperties:
    """
    Properties of a tensor.

    Attributes:
        dtype: Data type of the tensor (as string).
        requires_grad: Whether the tensor requires gradients. Default False.
        memory_format: Memory format (optional). Default None.
    """
    dtype: str
    requires_grad: bool = False
    memory_format: Optional[str] = None


@dataclass
class BytesStorageMetadata:
    """Metadata for bytes data stored in checkpoint."""

@dataclass(frozen=True)
class TensorStorageMetadata:
    """
    Metadata for a distributed tensor.

    Contains properties, global size, and list of chunks stored across ranks.

    Attributes:
        properties: Tensor properties (dtype, etc.).
        size: Global size of the tensor.
        chunks: List of chunks stored in the checkpoint. Default [].
    """
    properties: TensorProperties
    size: tuple
    chunks: list[ChunkStorageMetadata] = field(default_factory=list)


@dataclass
class Metadata:
    """
    Global metadata for a checkpoint.

    Contains metadata for all items in the state_dict, along with planner and storage-specific data.

    Attributes:
        state_dict_metadata: Mapping from FQN to storage metadata.
        planner_data: Planner-specific data (optional). Default None.
        storage_data: Storage-specific data (optional). Default None.
        version: Checkpoint format version. Default "1.0".
    """
    state_dict_metadata: dict[str, Union[TensorStorageMetadata, BytesStorageMetadata]]
    planner_data: Any = None  # Planner-specific data (can be any type)
    storage_data: Optional[dict[MetadataIndex, Any]] = None  # Storage mapping: MetadataIndex -> StorageInfo
    version: str = "1.0"
