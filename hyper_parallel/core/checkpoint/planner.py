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
"""Planner interfaces and implementations"""
import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from hyper_parallel.core.checkpoint.metadata import (
    Metadata, MetadataIndex
)


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
        size = chunk.sizes
        num = 1
        for dim in size:
            num *= int(dim)
        # Try to get dtype item size from properties
        dtype_str = getattr(properties, "dtype", None)
        if dtype_str is None:
            return int(num)
        # Simple estimation: assume common dtypes
        dtype_to_size_map = {
            "int32": 4, "int64": 8, "bfloat16": 2, "float16": 2, "float32": 4, "float64": 8
        }
        dtype_str_lower = str(dtype_str).lower()
        elem_size = 4  # Default to 4 bytes
        for dtype_name, size in dtype_to_size_map.items():
            if dtype_name in dtype_str_lower:
                elem_size = size
                break
        return int(num) * int(elem_size)



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


@dataclass
class SavePlan:
    """
    Plan for saving checkpoint.

    Contains write items and optional storage/planner-specific data.

    Attributes:
        items: List of WriteItems to be saved. Default [].
        storage_data: Storage-specific data (optional). Default None.
        planner_data: Planner-specific data (optional). Default None.
    """
    items: list[WriteItem] = field(default_factory=list)
    storage_data: Optional[dict[MetadataIndex, Any]] = None  # Storage-specific data mapping
    planner_data: Any = None  # Planner-specific data (can be any type)


@dataclass
class LoadPlan:
    """
    Plan for loading checkpoint.

    Contains read items and optional storage/planner-specific data.

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
    def build_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        """
        Build global plan from all local plans.

        Combines local plans from all ranks into a global plan and creates checkpoint metadata.
        This method may deduplicate redundant data across ranks and assign storage indices.

        Args:
            all_plans (list[SavePlan]): List of local save plans from all ranks.

        Returns:
            tuple[list[SavePlan], Metadata]: Updated global plans (one per rank) and
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
    def get_tensor(self, index: MetadataIndex) -> Any:
        """
        Get tensor data for a given MetadataIndex.

        This method allows storage writers to retrieve tensor data when needed,
        avoiding the need to store tensors in WriteItem.tensor_data (which would
        be transmitted during all_gather operations).

        Args:
            index (MetadataIndex): Metadata index identifying the tensor.

        Returns:
            Any: Tensor data (tensor-like object) or None if not found.
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
    def build_global_plan(self, all_plans: list[LoadPlan]) -> list[LoadPlan]:
        """
        Build global plan from all local plans.

        Combines local plans from all ranks into a global plan. This method may
        coordinate across ranks or perform optimizations.

        Args:
            all_plans (list[LoadPlan]): List of local load plans from all ranks.

        Returns:
            list[LoadPlan]: Updated global load plans (one per rank).
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
