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
"""Standard planner implementations for checkpoint save and load."""
import dataclasses
import pickle
from typing import Any, Optional, Union

from hyper_parallel.core.checkpoint.metadata import (
    Metadata, MetadataIndex, ChunkStorageMetadata,
    TensorStorageMetadata, TensorProperties, BytesStorageMetadata
)
from hyper_parallel.core.checkpoint.planner import (
    SavePlan, SavePlanner, LoadPlan, LoadPlanner,
    WriteItem, WriteItemType, ReadItem, LoadItemType
)
from hyper_parallel.core.checkpoint.reshard import infer_slice_area_by_rank, infer_intersection
from hyper_parallel.core.checkpoint.util import (
    narrow_tensor_by_index,
    chunk_to_area,
    create_chunk_list_for_load,
    remove_redundant_plans,
)
from hyper_parallel.core.dtensor import DTensor, Layout
from hyper_parallel.platform import get_platform


class StandardSavePlanner(SavePlanner):
    """Standard implementation of SavePlanner for distributed checkpoint saving."""

    def __init__(self):
        self.state_dict: Optional[dict[str, Any]] = None
        self.is_coordinator: bool = False
        self.rank: int = 0
        self.remove_redundancy: bool = True
        self.save_to_minimum_rank: bool = True
        self._tensor_cache: dict[MetadataIndex, Any] = {}  # Cache for tensor data

    def configure_planner(self, state_dict: dict[str, Any], **kwargs) -> None:
        """
        Configure planner.

        Args:
            state_dict (dict[str, Any]): The state_dict to save.
            **kwargs: Additional keyword arguments (e.g., is_coordinator, rank, remove_redundancy,
                save_to_minimum_rank).
        """
        self.state_dict = state_dict
        self.is_coordinator = kwargs.get("is_coordinator", False)
        self.rank = kwargs.get("rank", 0)
        self.remove_redundancy = kwargs.get("remove_redundancy", True)
        self.save_to_minimum_rank = kwargs.get("save_to_minimum_rank", True)

    def build_local_plan(self) -> SavePlan:
        """
        Create local save plan.

        Returns:
            SavePlan: Local save plan containing WriteItems for this rank.
        """
        if self.state_dict is None:
            raise RuntimeError("Planner not set up")

        def compute_global_offsets(global_shape: tuple[int, ...], dtensor_layout: Layout) -> tuple[int, ...]:
            """
            Compute the offsets of local tensor in global tensor based on layout.

            Args:
                global_shape (tuple[int, ...]): Global shape of the tensor.
                dtensor_layout (Layout): Layout of the DTensor.

            Returns:
                tuple[int, ...]: Tuple of offsets for each dimension.
            """
            if dtensor_layout is None:
                # If layout is None, return all zeros (no sharding)
                return tuple(0 for _ in global_shape)

            # Validate layout attributes
            if not hasattr(dtensor_layout, 'mesh_shape') or dtensor_layout.mesh_shape is None:
                raise ValueError("Layout must have mesh_shape attribute")
            if not hasattr(dtensor_layout, 'tensor_map') or dtensor_layout.tensor_map is None:
                raise ValueError("Layout must have tensor_map attribute")
            if not hasattr(dtensor_layout, 'rank_list') or dtensor_layout.rank_list is None:
                raise ValueError("Layout must have rank_list attribute")

            current_rank = self.rank
            if current_rank not in dtensor_layout.rank_list:
                raise ValueError(
                    f"Current rank {current_rank} not found in layout's rank_list {dtensor_layout.rank_list}")

            inner_rank_id = dtensor_layout.rank_list.index(current_rank)

            # Calculate slice area using infer_slice_area_by_rank
            slice_area = infer_slice_area_by_rank(
                mesh_shape=dtensor_layout.mesh_shape,
                tensor_map=dtensor_layout.tensor_map,
                rank_id=inner_rank_id,
                full_shape=global_shape
            )

            # Extract offsets (start values) from slice_area
            return tuple(start for start, _ in slice_area)

        items = []
        for fqn, obj in self.state_dict.items():
            # Check if it's a DTensor
            if isinstance(obj, DTensor):
                # Create write item for DTensor
                local_tensor = obj.to_local()
                layout = obj.layout

                # Get chunk metadata with offsets
                if layout:
                    offsets = compute_global_offsets(obj.shape, layout)
                else:
                    offsets = (0,) * len(local_tensor.shape)

                sizes = local_tensor.shape
                chunk = ChunkStorageMetadata(offsets=offsets, sizes=sizes)

                # Get tensor properties
                dtype_str = str(local_tensor.dtype) if hasattr(local_tensor, 'dtype') else 'unknown'
                properties = TensorProperties(dtype=dtype_str)

                # Create write item for this tensor
                index = MetadataIndex(fqn=fqn, offset=offsets, index=None)
                # Store tensor in cache instead of tensor_data
                self._tensor_cache[index] = local_tensor
                write_item = WriteItem(
                    index=index,
                    type=WriteItemType.TENSOR,
                    tensor_data={
                        'chunk': chunk,
                        'properties': properties,
                        'size': obj.shape,
                    }
                )
                items.append(write_item)
            else:
                # Handle non-tensor types (bytes, etc.)
                index = MetadataIndex(fqn=fqn)
                write_item = WriteItem(
                    index=index,
                    type=WriteItemType.BYTE_IO,
                    bytes_io_data=obj
                )
                items.append(write_item)

        return SavePlan(items=items)

    def build_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]:
        """
        Build global plan from all local plans.

        Collects chunks from all ranks, validates consistency, and creates metadata for the checkpoint.

        Args:
            all_plans (list[SavePlan]): List of local plans from all ranks.

        Returns:
            tuple[list[SavePlan], Metadata]: Updated plans and checkpoint metadata.
        """
        # Deduplicate plans if redundancy removal is enabled
        if self.remove_redundancy and len(all_plans) > 1:
            all_plans = remove_redundant_plans(all_plans, save_to_minimum_rank=self.save_to_minimum_rank)

        # Collect all write items by FQN
        fqn_to_chunks: dict[str, list[ChunkStorageMetadata]] = {}
        fqn_to_properties: dict[str, TensorProperties] = {}
        fqn_to_size: dict[str, tuple] = {}
        state_dict_metadata: dict[str, Union[TensorStorageMetadata, BytesStorageMetadata]] = {}

        final_global_plans: list[SavePlan] = []
        for plan in all_plans:
            with_index_items = []
            for item in plan.items:
                if item.type == WriteItemType.TENSOR and item.tensor_data:
                    fqn = item.index.fqn
                    chunk = item.tensor_data['chunk']
                    properties = item.tensor_data['properties']
                    size = item.tensor_data['size']

                    # Validate consistency across ranks
                    if fqn in fqn_to_chunks and (fqn_to_properties[fqn] != properties or fqn_to_size[fqn] != size):
                        raise ValueError(f"The {fqn} in different rank has different properties and size.")

                    # Initialize FQN entry if not exists
                    if fqn not in fqn_to_chunks:
                        fqn_to_properties[fqn] = properties
                        fqn_to_size[fqn] = size
                        fqn_to_chunks[fqn] = []

                    # Update index and add chunk
                    new_index = dataclasses.replace(item.index, index=len(fqn_to_chunks[fqn]))
                    with_index_item = dataclasses.replace(item, index=new_index)
                    with_index_items.append(with_index_item)
                    fqn_to_chunks[fqn].append(chunk)

                elif item.type == WriteItemType.BYTE_IO:
                    with_index_items.append(item)
                    state_dict_metadata[item.index.fqn] = BytesStorageMetadata()
                else:
                    raise ValueError(f"Unsupported write item type: {item.type}")

            final_global_plans.append(dataclasses.replace(plan, items=with_index_items))

        # Create metadata for all tensors
        for fqn, chunks in fqn_to_chunks.items():
            state_dict_metadata[fqn] = TensorStorageMetadata(
                properties=fqn_to_properties[fqn],
                size=fqn_to_size[fqn],
                chunks=chunks
            )

        metadata = Metadata(state_dict_metadata=state_dict_metadata)
        return final_global_plans, metadata

    def _update_tensor_cache(self, plan: SavePlan) -> None:
        """
        Update tensor cache keys to match the finalized plan's MetadataIndex values.

        Updates cache keys for tensors that are in the plan, and removes tensors
        that are not in the plan. The plan's items have been modified by build_global_plan
        (index field added), so we need to update the cache keys accordingly.

        Args:
            plan (SavePlan): Plan with updated MetadataIndex values.
        """
        # Build mapping from (fqn, offset) to updated MetadataIndex from plan
        plan_tensor_map: dict[tuple[str, tuple], MetadataIndex] = {}
        for item in plan.items:
            if item.type == WriteItemType.TENSOR and item.tensor_data:
                key_pair = (item.index.fqn, item.index.offset)
                plan_tensor_map[key_pair] = item.index

        # Update tensor cache keys and remove tensors not in plan
        keys_to_remove = []
        for cached_key in list(self._tensor_cache.keys()):
            key_pair = (cached_key.fqn, cached_key.offset)

            if key_pair in plan_tensor_map:
                # Update cache key if index changed
                new_index = plan_tensor_map[key_pair]
                if cached_key != new_index:
                    tensor = self._tensor_cache.pop(cached_key)
                    self._tensor_cache[new_index] = tensor
            else:
                # Mark for removal if not in plan
                keys_to_remove.append(cached_key)

        # Remove tensors not in plan
        for key in keys_to_remove:
            self._tensor_cache.pop(key, None)

    def finalize_plan(self, plan: SavePlan) -> SavePlan:
        """
        Finalize the plan and update tensor cache keys.

        Updates tensor cache keys to match the finalized plan's MetadataIndex values.
        The plan's items have been modified by build_global_plan (index field added),
        so we need to update the cache keys accordingly.
        Also removes tensors from cache that are not in the finalized plan.

        Args:
            plan (SavePlan): Plan to finalize (with updated MetadataIndex values).

        Returns:
            SavePlan: Finalized plan.
        """
        self._update_tensor_cache(plan)
        return plan

    def get_tensor(self, index: MetadataIndex) -> Any:
        """
        Get tensor data for a given MetadataIndex from the cache.

        Args:
            index (MetadataIndex): Metadata index identifying the tensor.

        Returns:
            Any: Tensor data (tensor-like object) or None if not found.
        """
        return self._tensor_cache.get(index)


def create_read_items_for_chunk_list(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: list[ChunkStorageMetadata],
) -> list[ReadItem]:
    """
    Create ReadItems by matching local chunks (what this rank needs) with
    saved chunks (checkpoint_md.chunks), including resharding overlaps.

    Mirrors torch create_read_items_for_chunk_list behavior.

    Args:
        fqn (str): Fully qualified name of the tensor.
        checkpoint_md (TensorStorageMetadata): Tensor storage metadata from checkpoint.
        local_chunks (list[ChunkStorageMetadata]): List of local chunks needed by this rank.

    Returns:
        list[ReadItem]: List of ReadItems for loading the required data.
    """
    read_items: list[ReadItem] = []
    saved_chunks = checkpoint_md.chunks
    if not local_chunks or not saved_chunks:
        return read_items

    for local_idx, local_chunk in enumerate(local_chunks):
        local_area = chunk_to_area(local_chunk)
        for storage_idx, storage_chunk in enumerate(saved_chunks):
            saved_area = chunk_to_area(storage_chunk)
            overlap = infer_intersection(local_area, saved_area)
            if overlap is None:
                continue

            dest_offsets = tuple(overlap[i][0] - local_chunk.offsets[i] for i in range(len(overlap)))
            storage_offsets = tuple(overlap[i][0] - storage_chunk.offsets[i] for i in range(len(overlap)))
            lengths = tuple(overlap[i][1] - overlap[i][0] for i in range(len(overlap)))

            read_items.append(
                ReadItem(
                    type=LoadItemType.TENSOR,
                    dest_index=MetadataIndex(fqn=fqn, offset=local_chunk.offsets, index=local_idx),
                    dest_offsets=dest_offsets,
                    storage_index=MetadataIndex(fqn=fqn, offset=storage_chunk.offsets, index=storage_idx),
                    storage_offsets=storage_offsets,
                    lengths=lengths,
                )
            )
    return read_items


class StandardLoadPlanner(LoadPlanner):
    """
    Standard implementation of LoadPlanner.

    Iterate state_dict and creates load plans via chunk list for resharding support.
    """

    def __init__(self, allow_partial_load: bool = False):
        self.state_dict: Optional[dict[str, Any]] = None
        self.metadata: Optional[Metadata] = None
        self.is_coordinator: bool = False
        self.rank: int = 0
        self.allow_partial_load = allow_partial_load

    def configure_planner(self, state_dict: dict[str, Any], metadata: Metadata, **kwargs) -> None:
        """
        Configure planner with state dict and metadata.

        Args:
            state_dict (dict[str, Any]): The state_dict to load into (modified in-place).
            metadata (Metadata): Checkpoint metadata.
            **kwargs: Additional keyword arguments (e.g., is_coordinator, rank).
        """
        self.state_dict = state_dict
        self.metadata = metadata
        self.is_coordinator = kwargs.get("is_coordinator", False)
        self.rank = kwargs.get("rank", 0)

    def build_local_plan(self) -> LoadPlan:
        """
        Build local load plan.

        Iterate state_dict and creates load plans via chunk list for resharding support.

        Returns:
            LoadPlan: Local load plan containing ReadItems for this rank.
        """
        if self.state_dict is None or self.metadata is None:
            raise RuntimeError("Planner not configured")

        requests: list[ReadItem] = []
        strict = not self.allow_partial_load
        for fqn, obj in self.state_dict.items():
            if fqn not in self.metadata.state_dict_metadata:
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                continue
            md = self.metadata.state_dict_metadata[fqn]
            if isinstance(md, TensorStorageMetadata):
                obj_size = getattr(obj, "shape", None)
                if obj_size is None or md.size != tuple(obj_size):
                    raise ValueError(
                        f"Size mismatch between saved {md.size} and current: {obj_size} for {fqn}",
                    )
                if isinstance(obj, DTensor):
                    layout = getattr(obj, "layout", None)
                    rank_list = getattr(layout, "rank_list", None) if layout else None
                    if rank_list is None and layout is not None:
                        rank_list = getattr(layout, "_rank_list", None)
                    if layout is not None and rank_list is not None:
                        if get_platform().get_rank() not in rank_list:
                            continue
                try:
                    local_chunks = create_chunk_list_for_load(obj)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid checkpoint metadata for {fqn}, "
                        f"expected BytesStorageMetadata but found {type(md)}",
                    ) from e
                requests += create_read_items_for_chunk_list(fqn, md, local_chunks)
            else:
                requests.append(
                    ReadItem(
                        type=LoadItemType.BYTE_IO,
                        dest_index=MetadataIndex(fqn=fqn),
                        dest_offsets=(0,),
                        storage_index=MetadataIndex(fqn=fqn),
                        storage_offsets=(0,),
                        lengths=(0,),
                    )
                )
        return LoadPlan(items=requests)

    def build_global_plan(self, all_plans: list[LoadPlan]) -> list[LoadPlan]:
        """
        Build global plan from all local plans.

        For now, returns plans as-is. In a more sophisticated implementation, you might need to coordinate across ranks.

        Args:
            all_plans (list[LoadPlan]): List of local plans from all ranks.

        Returns:
            list[LoadPlan]: Global plans (currently returns plans as-is).
        """
        return all_plans

    def finalize_plan(self, plan: LoadPlan) -> LoadPlan:
        """
        Finalize the plan (no-op for default implementation).

        Args:
            plan (LoadPlan): Plan to finalize.

        Returns:
            LoadPlan: Finalized plan.
        """
        return plan

    def acquire_tensor(self, read_item: ReadItem) -> Any:
        """
        Acquire the destination slice (narrow view) for this read_item.

        StorageReader uses this to copy loaded data into the correct region.
        Torch-aligned behavior.

        Args:
            read_item (ReadItem): The read item specifying what to load.

        Returns:
            Any: The destination tensor slice where data should be written
                (tensor-like object).
        """
        if self.state_dict is None:
            raise RuntimeError("Planner not configured")

        fqn = read_item.dest_index.fqn
        if fqn not in self.state_dict:
            raise KeyError(f"Key {fqn} not found in state_dict")

        target = self.state_dict[fqn]
        local_tensor = target.to_local() if isinstance(target, DTensor) else target
        return narrow_tensor_by_index(
            local_tensor,
            read_item.dest_offsets,
            read_item.lengths,
        )

    def apply_tensor(self, read_item: ReadItem, tensor: Any) -> None:
        """
        Apply tensor after reading.

        After read_data copies into the slice, this is no-op when tensor is the
        same slice. When the backend has no copy_ (e.g. mindspore), read_data
        passes the loaded slice here; we copy it into the destination slice.

        Args:
            read_item (ReadItem): The read item that was processed.
            tensor (Any): The tensor data to apply (tensor-like object).
        """
        if tensor is None:
            return
        dest_slice = self.acquire_tensor(read_item)
        if dest_slice is tensor:
            return
        if hasattr(dest_slice, "copy_"):
            dest_slice.copy_(tensor)
        else:
            # Fallback: assign into state_dict if supported
            dest_slice[...] = tensor

    def apply_bytes(self, read_item: ReadItem, value: bytes) -> None:
        """
        Load bytes data into state_dict.

        Args:
            read_item (ReadItem): The read item specifying the destination.
            value (bytes): The bytes data to deserialize and load.
        """
        if self.state_dict is None:
            raise RuntimeError("Planner not set up")

        fqn = read_item.dest_index.fqn
        # Deserialize bytes
        obj = pickle.loads(value)
        self.state_dict[fqn] = obj
