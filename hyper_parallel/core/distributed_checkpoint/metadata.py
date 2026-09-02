# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
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


CHUNK_INFO = "chunk_info"

@dataclass(frozen=True)
class MetadataIndex:
    """
    Index to identify a specific piece of data in the checkpoint.

    ``index`` is a positional hint into the chunk list, not part of the identity: a chunk is
    identified by its ``fqn`` and ``offset``, and the same chunk is described with ``index=None``
    while plans are being deduplicated and with its ordinal afterwards. Keeping it out of
    ``__eq__``/``__hash__`` (as torch's MetadataIndex does) makes those two spellings compare
    equal, and drops the field from every hash computed over the millions of indices a global
    plan builds.

    Attributes:
        fqn: Fully qualified name of the tensor/object.
        offset: Offset in the tensor (for sharded tensors). Default ().
        index: Index for sharded tensors (None for non-sharded). Not compared or hashed.
            Default None.
    """
    fqn: str
    offset: tuple = field(default_factory=tuple)
    index: Optional[int] = field(default=None, compare=False, hash=False)


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
class ChunkInfo:
    """
    Info for a tensor chunk.

    Represents a portion of a distributed tensor stored in the checkpoint.

    Attributes:
        chunk: Offsets in the global tensor for each dimension.
        global_shape: Sizes of the chunk for each dimension.
        replica_rank_list: Have the same sharded tensor ranks list.
    """
    chunk: ChunkStorageMetadata
    global_shape: tuple[int]
    replica_rank_list: Optional[tuple[int]] = None


# Ordered longest-match-first: "bfloat16" has to be tried before "float16", which is a
# substring of it. A dict would work today but hides that the order is load-bearing.
_DTYPE_ELEMENT_SIZES = (
    ("bfloat16", 2),
    ("float16", 2),
    ("float32", 4),
    ("float64", 8),
    ("int8", 1),
    ("int16", 2),
    ("int32", 4),
    ("int64", 8),
    ("bool", 1),
)
DEFAULT_DTYPE_ELEMENT_SIZE = 4

# A checkpoint names only a handful of distinct dtypes, while this is asked once per plan
# item - tens of thousands of times on a large model - so each name is scanned once and
# answered from here afterwards.
_dtype_element_size_cache: dict[str, int] = {}


def dtype_element_size(dtype: Optional[str]) -> int:
    """
    Bytes one element of ``dtype`` takes.

    The dtype reaches the checkpoint as a framework-specific string such as
    ``"torch.bfloat16"`` or ``"Float32"``, so it is matched by substring rather than by
    equality. Anything unrecognized falls back to :data:`DEFAULT_DTYPE_ELEMENT_SIZE`, which
    keeps the sizes usable as relative weights even for a dtype this table does not name.

    Args:
        dtype (Optional[str]): Dtype name from :class:`TensorProperties`, or None.

    Returns:
        int: Size of one element in bytes.
    """
    if dtype is None:
        return DEFAULT_DTYPE_ELEMENT_SIZE
    key = str(dtype)
    cached = _dtype_element_size_cache.get(key)
    if cached is not None:
        return cached
    lowered = key.lower()
    element_size = DEFAULT_DTYPE_ELEMENT_SIZE
    for name, size in _DTYPE_ELEMENT_SIZES:
        if name in lowered:
            element_size = size
            break
    _dtype_element_size_cache[key] = element_size
    return element_size


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
