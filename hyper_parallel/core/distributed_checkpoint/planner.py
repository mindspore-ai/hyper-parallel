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
"""Planner interfaces and implementations"""
import abc
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from hyper_parallel.core.distributed_checkpoint.metadata import Metadata, MetadataIndex


# Substring -> element size in bytes, matched in order (bfloat16 must be tried before float16
# would match, and int32 before int64). Module level so tensor_storage_size() does not rebuild
# the mapping on every call.
_DTYPE_SUBSTR_TO_ELEM_SIZE = {
    "int32": 4, "int64": 8, "bfloat16": 2, "float16": 2, "float32": 4, "float64": 8
}
_DEFAULT_ELEM_SIZE = 4
# dtype strings come from a handful of distinct values per checkpoint, so memoising the scan
# turns a per-item substring search into a single dict lookup.
_dtype_elem_size_cache: dict[str, int] = {}


def _elem_size_from_dtype(dtype_str: Any) -> int:
    """
    Best-effort element size in bytes for a dtype rendered as a string.

    Args:
        dtype_str (Any): Object whose ``str()`` names the dtype, e.g. ``"torch.bfloat16"``.

    Returns:
        int: Element size in bytes, defaulting to 4 for unrecognised dtypes.
    """
    key = str(dtype_str)
    elem_size = _dtype_elem_size_cache.get(key)
    if elem_size is not None:
        return elem_size
    lowered = key.lower()
    elem_size = _DEFAULT_ELEM_SIZE
    for dtype_name, size in _DTYPE_SUBSTR_TO_ELEM_SIZE.items():
        if dtype_name in lowered:
            elem_size = size
            break
    _dtype_elem_size_cache[key] = elem_size
    return elem_size


class WriteItemType(Enum):
    """Type of write item."""
    TENSOR = "tensor"
    BYTE_IO = "byte_io"


class LoadItemType(Enum):
    """Type of load item."""
    TENSOR = "tensor"
    BYTE_IO = "byte_io"


@dataclass(frozen=True)
class WriteItem:
    """
    Item to be written to storage.

    Represents a single logical item (tensor or bytes) to be saved.

    Attributes:
        index: Metadata index identifying this item.
        type: Type of write item (TENSOR or BYTE_IO).
        tensor_data: Dictionary containing tensor data (for TENSOR type). Default None.
        bytes_io_data: Bytes data (for BYTE_IO type). Default None.
    """
    index: MetadataIndex
    type: WriteItemType
    # Keys: 'chunk' (ChunkStorageMetadata), 'properties' (TensorProperties), 'size' (tuple).
    # Actual tensor data is in planner's tensor cache, not here, to avoid all_gather of tensors.
    tensor_data: Optional[dict[str, Any]] = None
    bytes_io_data: Optional[Union[bytes, Any]] = None  # Bytes or pickle-serializable object

    def tensor_storage_size(self) -> Optional[int]:
        """
        Best-effort storage size estimation in bytes for tensor items.

        Returns:
            Optional[int]: Estimated storage size in bytes for tensor items,
                or None if estimation cannot be performed (e.g., for non-tensor items).
        """
        if self.type != WriteItemType.TENSOR or not self.tensor_data:
            return None

        # Try to estimate from metadata
        chunk = self.tensor_data.get("chunk")
        properties = self.tensor_data.get("properties")
        if chunk is None or properties is None:
            return None

        # Get size from chunk (local chunk size, not global size)
        num = 1
        for dim in chunk.sizes:
            num *= int(dim)
        # Try to get dtype item size from properties
        dtype_str = getattr(properties, "dtype", None)
        if dtype_str is None:
            return int(num)
        return int(num) * _elem_size_from_dtype(dtype_str)



@dataclass(frozen=True)
class ReadItem:
    """
    Item to be read from storage.

    Represents a single logical read operation, mapping from checkpoint storage
    to destination state_dict location.

    Attributes:
        type: Type of load item (TENSOR or BYTE_IO).
        dest_index: Metadata index identifying the destination in state_dict.
        dest_offsets: Offsets into the destination tensor (for TENSOR type).
        storage_index: Metadata index identifying the source in checkpoint.
        storage_offsets: Offsets into the checkpoint storage data.
        lengths: Size of the hypercube to copy (dimensions of the data region).
    """
    type: LoadItemType
    dest_index: MetadataIndex  # Index into the state_dict
    dest_offsets: tuple  # Offsets into destination tensor
    storage_index: MetadataIndex  # Index into the checkpoint
    storage_offsets: tuple  # Offset into the checkpoint data
    lengths: tuple  # Size of the hypercube to copy


@dataclass(frozen=True)
class SavePlan:
    """
    Plan for saving checkpoint.

    Contains write items and optional storage/planner-specific data.

    Frozen so that a plan cannot be edited after it has been handed on: plans are gathered from
    every rank, kept in the planner's cross-call result cache, and passed to hooks that must treat
    them as inputs. Build a changed plan with ``dataclasses.replace`` instead. Note this freezes
    the fields, not the contents -- ``items`` is still a list that could be appended to.

    Attributes:
        items: List of WriteItems to be saved. Default [].
        storage_data: Storage-specific data (optional). Default None.
        planner_data: Planner-specific data (optional). Default None.
    """
    items: list[WriteItem] = field(default_factory=list)
    storage_data: Optional[dict[MetadataIndex, Any]] = None  # Storage-specific data mapping
    planner_data: Any = None  # Planner-specific data (can be any type)


@dataclass(frozen=True)
class LoadPlan:
    """
    Plan for loading checkpoint.

    Contains read items and optional storage/planner-specific data.

    Frozen for the same reason as :class:`SavePlan`.

    Attributes:
        items: List of ReadItems to be loaded. Default [].
        storage_data: Storage-specific data (optional). Default None.
        planner_data: Planner-specific data (optional). Default None.
    """
    items: list[ReadItem] = field(default_factory=list)
    storage_data: Optional[dict[MetadataIndex, Any]] = None  # Storage-specific data mapping
    planner_data: Any = None  # Planner-specific data (can be any type)


class SavePlanner(abc.ABC):
    """Abstract base class for save planners."""

    @abc.abstractmethod
    def configure_planner(self, state_dict: dict[str, Any], **kwargs) -> None:
        """
        Configure the planner with state dict.

        Args:
            state_dict (dict[str, Any]): The state_dict to save.
            **kwargs: Additional keyword arguments (e.g., is_coordinator, rank, remove_redundancy,
                save_to_minimum_rank).
        """

    @abc.abstractmethod
    def build_local_plan(self) -> SavePlan:
        """
        Build local save plan.

        Creates a plan for saving checkpoint data from the current rank's perspective.
        This plan contains WriteItems for all tensors and bytes that this rank needs to save.

        Returns:
            SavePlan: Local save plan containing WriteItems for this rank.
        """

    @abc.abstractmethod
    def build_global_plan(self, all_plans: list[SavePlan]) -> tuple[SavePlan, Metadata]:
        """
        Build this rank's final save plan and the global checkpoint metadata.

        Combines local plans from all ranks: deduplicates redundant data across ranks and assigns
        storage indices. Every rank runs this over the same gathered plans and writes only its own
        shards, so only this rank's plan is returned -- the metadata still describes every rank's.

        Args:
            all_plans (list[SavePlan]): Local save plans from all ranks, indexed by rank.

        Returns:
            tuple[SavePlan, Metadata]: This rank's save plan with storage indices assigned, and
                checkpoint metadata containing information about all saved items.
        """

    @abc.abstractmethod
    def finalize_plan(self, plan: SavePlan) -> SavePlan:
        """
        Finalize the plan.

        Performs any final adjustments to the plan before execution, such as updating
        tensor cache keys or performing planner-specific optimizations.

        Args:
            plan (SavePlan): The plan to finalize.

        Returns:
            SavePlan: The finalized plan ready for execution.
        """

    @abc.abstractmethod
    def get_data(self, item: WriteItem) -> Any:
        """
        Get runtime data for a write item from the current state_dict.

        Args:
            item (WriteItem): The write item to get data for.

        Returns:
            Any: Runtime object to be written for this item.
        """


class LoadPlanner(abc.ABC):
    """Abstract base class for load planners."""

    @abc.abstractmethod
    def configure_planner(self, state_dict: dict[str, Any], metadata: Metadata, **kwargs) -> None:
        """
        Configure the planner with state dict and metadata.

        Args:
            state_dict (dict[str, Any]): The state_dict to load into (modified in-place).
            metadata (Metadata): Checkpoint metadata.
            **kwargs: Additional keyword arguments (e.g., is_coordinator, rank).
        """

    @abc.abstractmethod
    def build_local_plan(self) -> LoadPlan:
        """
        Build local load plan.

        Creates a plan for loading checkpoint data from the current rank's perspective.
        This plan contains ReadItems for all tensors and bytes that this rank needs to load.

        Returns:
            LoadPlan: Local load plan containing ReadItems for this rank.
        """

    @abc.abstractmethod
    def build_global_plan(self, all_plans: list[LoadPlan]) -> LoadPlan:
        """
        Build this rank's final load plan from all local plans.

        Every rank runs this over the same gathered plans and reads only its own shards, so only
        this rank's plan is returned. This method may coordinate across ranks or perform
        optimizations while doing so.

        Args:
            all_plans (list[LoadPlan]): Local load plans from all ranks, indexed by rank.

        Returns:
            LoadPlan: This rank's load plan.
        """

    @abc.abstractmethod
    def finalize_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Finalize the plan.

        Performs any final adjustments to the plan before execution, such as
        performing planner-specific optimizations or validations.

        Args:
            plan (LoadPlan): The plan to finalize.

        Returns:
            LoadPlan: The finalized plan ready for execution.
        """

    @abc.abstractmethod
    def acquire_tensor(self, read_item: ReadItem) -> Any:
        """
        Acquire tensor for read item.

        Returns a tensor slice/view where data should be written.

        Args:
            read_item (ReadItem): Read item to acquire tensor for.

        Returns:
            Any: Acquired tensor slice/view (tensor-like object).
        """

    @abc.abstractmethod
    def apply_tensor(self, read_item: ReadItem, tensor: Any) -> None:
        """
        Apply tensor after reading.

        Args:
            read_item (ReadItem): Read item.
            tensor (Any): Tensor data to apply (tensor-like object).
        """

    @abc.abstractmethod
    def apply_bytes(self, read_item: ReadItem, value: bytes) -> None:
        """
        Apply bytes data.

        Args:
            read_item (ReadItem): The read item specifying the destination.
            value (bytes): The bytes data to deserialize and apply.
        """
