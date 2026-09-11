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

"""Standard planner implementations for checkpoint save and load."""

from dataclasses import dataclass
from collections import defaultdict
import dataclasses
import math
from itertools import compress
import pickle
from typing import Any, Optional, Union

from hyper_parallel.core.distributed_checkpoint.metadata import (
    CHUNK_INFO,
    Metadata,
    MetadataIndex,
    dtype_element_size,
    ChunkStorageMetadata,
    ChunkInfo,
    TensorStorageMetadata,
    TensorProperties,
    BytesStorageMetadata
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    SavePlan,
    SavePlanner,
    LoadPlan,
    LoadPlanner,
    WriteItem,
    WriteItemType,
    ReadItem,
    LoadItemType,
    BroadcastSource,
)
from hyper_parallel.core.distributed_checkpoint.reshard import infer_intersection
from hyper_parallel.core.distributed_checkpoint.ragged_utils import (
    create_ragged_write_items,
    get_ragged_box_tensor,
)
from hyper_parallel.core.distributed_checkpoint.util import (
    narrow_tensor_by_index,
    chunk_to_area,
    create_chunk_list_for_tensor,
    plan_ownership_masks,
    flatten_state_dict,
    set_element,
    dcp_timer_decorator,
    logger,
    platform,
    Tensor,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout, infer_slice_area_by_layout


def _own_plan_index(all_plans: Union[list[SavePlan], list[LoadPlan]], rank: int) -> int:
    """
    Locate this rank's plan in the gathered list.

    Both gather paths return one entry per rank in rank order; without collectives the
    "gather" is just this rank's own plan in a one element list.

    Args:
        all_plans (Union[list[SavePlan], list[LoadPlan]]): Local plans from all ranks.
        rank (int): Rank looking for its own plan.

    Returns:
        int: Index of this rank's plan.

    Raises:
        ValueError: If the gathered list holds no plan for this rank.
    """
    own_index = rank if len(all_plans) > 1 else 0
    if not 0 <= own_index < len(all_plans):
        raise ValueError(
            f"Rank {rank} has no plan of its own among the {len(all_plans)} gathered "
            "plans; the gathered list must hold one plan per rank, in rank order."
        )
    return own_index


@dataclass(frozen=True)
class CachedSaveResult:
    """Cached finalized save result keyed by planner cache namespace."""

    final_plan: SavePlan
    metadata: Metadata


class StandardSavePlanner(SavePlanner):
    """Standard implementation of SavePlanner for distributed checkpoint saving."""

    cached_save_result: dict[str, CachedSaveResult] = {}

    def __init__(
            self,
            enable_plan_caching: bool = True,
            remove_redundancy: bool = True,
            save_to_minimum_rank: bool = False,
    ):
        self.state_dict: Optional[dict[str, Any]] = None
        self.is_coordinator: bool = False
        self.rank: int = 0
        self.remove_redundancy: bool = remove_redundancy
        self.save_to_minimum_rank: bool = save_to_minimum_rank
        self.flatten_state_dict: bool = True
        self._enable_plan_caching: bool = enable_plan_caching
        self._cached_plans_key: str = self.__class__.__name__

    def configure_planner(self, state_dict: dict[str, Any], **kwargs) -> None:
        """
        Configure planner.

        Args:
            state_dict (dict[str, Any]): The state_dict to save.
            **kwargs: Additional keyword arguments (e.g., is_coordinator, rank, remove_redundancy,
                save_to_minimum_rank).
        """
        self.is_coordinator = kwargs.get("is_coordinator", False)
        self.rank = kwargs.get("rank", 0)
        self.remove_redundancy = kwargs.get("remove_redundancy", self.remove_redundancy)
        self.save_to_minimum_rank = kwargs.get("save_to_minimum_rank", self.save_to_minimum_rank)
        self.flatten_state_dict = kwargs.get("flatten_state_dict", True)

        use_collectives = bool(kwargs.get("use_collectives", True))
        if not use_collectives:
            self.remove_redundancy = False
            self._enable_plan_caching = False
        elif "enable_plan_caching" in kwargs:
            self._enable_plan_caching = bool(kwargs["enable_plan_caching"])

        if self.flatten_state_dict:
            state_dict, self.name_mapping = flatten_state_dict(state_dict)
        self.state_dict = state_dict
        self._cached_plans_key = self._build_cache_key(state_dict)

    def _build_cache_key(self, state_dict: dict[str, Any]) -> str:
        """Build a stable cache namespace from state_dict keys."""
        return f"{self.__class__.__name__}:{'||'.join(state_dict.keys())}"

    @dcp_timer_decorator
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
            slice_area = infer_slice_area_by_layout(
                dtensor_layout,
                inner_rank_id,
                global_shape,
            )
            # Extract offsets (start values) from slice_area
            return tuple(start for start, _ in slice_area)

        items = []
        for fqn, obj in self.state_dict.items():
            # Check if it's a DTensor
            if isinstance(obj, DTensor):
                if obj.layout is not None and obj.layout.ragged_shard is not None:
                    items.extend(create_ragged_write_items(fqn, obj))
                    continue
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
            elif isinstance(obj, Tensor):
                # Create write item for platform.Tensor: build single chunk with tensor's own size
                dtype_str = str(obj.dtype) if hasattr(obj, 'dtype') else 'unknown'
                properties = TensorProperties(dtype=dtype_str)
                # handle Tensor with shard information
                if hasattr(obj, CHUNK_INFO):
                    if not isinstance(getattr(obj, CHUNK_INFO), ChunkInfo):
                        raise ValueError("The attr CHUNK_INFO should be a ChunkInfo instance")
                    chunk = getattr(obj, CHUNK_INFO).chunk
                # Single chunk covering the whole tensor (offsets=0, sizes=shape)
                else:
                    chunk = ChunkStorageMetadata(
                        offsets=(0,) * len(obj.shape),
                        sizes=obj.shape,
                    )
                index = MetadataIndex(fqn=fqn, offset=chunk.offsets, index=None)
                write_item = WriteItem(
                    index=index,
                    type=WriteItemType.TENSOR,
                    tensor_data={
                        'chunk': chunk,
                        'properties': properties,
                        'size': getattr(obj, CHUNK_INFO).global_shape if hasattr(obj, CHUNK_INFO) else obj.shape,
                    }
                )
                items.append(write_item)
            else:
                # Handle non-tensor types (bytes, etc.)
                index = MetadataIndex(fqn=fqn)
                write_item = WriteItem(
                    index=index,
                    type=WriteItemType.BYTE_IO,
                    bytes_io_data=None
                )
                items.append(write_item)

        return SavePlan(
            items=items,
            planner_data=self.name_mapping if self.flatten_state_dict else None,
        )

    @dcp_timer_decorator
    def build_global_plan(self, all_plans: list[SavePlan]) -> tuple[SavePlan, Metadata]:
        """
        Build this rank's final save plan and the global checkpoint metadata.

        Every rank receives all local plans from the gather and runs this itself, and each one
        goes on to write only its own shards, so only this rank's plan is rebuilt here. The other
        ranks' items are still walked -- the metadata is global, and an item's chunk index is its
        position in that global chunk list -- but no plan object is built for them.

        Args:
            all_plans (list[SavePlan]): Local plans from all ranks, indexed by rank.

        Returns:
            tuple[SavePlan, Metadata]: This rank's plan with chunk indices assigned, and the
                checkpoint metadata describing every rank's chunks.

        Raises:
            ValueError: If an item has an unsupported type.
        """
        own_index = _own_plan_index(all_plans, self.rank)

        # Redundant items are skipped through a per-plan mask rather than by materialising
        # deduplicated plans: the loop below walks plan.items anyway.
        masks = (
            plan_ownership_masks(all_plans, save_to_minimum_rank=self.save_to_minimum_rank)
            if self.remove_redundancy and len(all_plans) > 1
            else None
        )

        # FQN -> (properties, size, chunks), collecting every rank's chunks in gather order.
        fqn_info: dict[str, tuple[TensorProperties, tuple, list[ChunkStorageMetadata]]] = {}
        state_dict_metadata: dict[str, Union[TensorStorageMetadata, BytesStorageMetadata]] = {}
        own_items: list[WriteItem] = []

        for plan_index, plan in enumerate(all_plans):
            is_own_plan = plan_index == own_index
            items = plan.items if masks is None else compress(plan.items, masks[plan_index])
            for item in items:
                if item.type == WriteItemType.TENSOR and item.tensor_data:
                    tensor_data = item.tensor_data
                    chunks = self._global_chunks_for(
                        item.index.fqn, tensor_data['properties'], tensor_data['size'], fqn_info)
                    if is_own_plan:
                        # Set index (platform.Tensor has exactly one chunk). replace() copies the
                        # remaining fields by name, so adding a field to either dataclass cannot
                        # silently drop it here; only this rank's items are rebuilt, so the cost
                        # is per-rank, not per-cluster.
                        new_index = dataclasses.replace(item.index, index=len(chunks))
                        own_items.append(dataclasses.replace(item, index=new_index))
                    chunks.append(tensor_data['chunk'])

                elif item.type == WriteItemType.BYTE_IO:
                    if is_own_plan:
                        own_items.append(item)
                    state_dict_metadata[item.index.fqn] = BytesStorageMetadata()
                else:
                    raise ValueError(f"Unsupported write item type: {item.type}")

        # Create metadata for all tensors
        for fqn, (properties, size, chunks) in fqn_info.items():
            state_dict_metadata[fqn] = TensorStorageMetadata(
                properties=properties,
                size=size,
                chunks=chunks
            )

        metadata = Metadata(state_dict_metadata=state_dict_metadata)
        if self.flatten_state_dict:
            merged_mapping = {}
            for p in all_plans:
                merged_mapping.update(p.planner_data)
            metadata.planner_data = merged_mapping
        return dataclasses.replace(all_plans[own_index], items=own_items), metadata

    @staticmethod
    def _global_chunks_for(fqn: str, properties: TensorProperties, size: tuple, fqn_info: dict) -> list:
        """
        Return the global chunk list for ``fqn``, registering it on first sight.

        Args:
            fqn (str): Fully qualified name of the tensor.
            properties (TensorProperties): Properties this rank reported for it.
            size (tuple): Global shape this rank reported for it.
            fqn_info (dict): FQN -> (properties, size, chunks), grown in place.

        Returns:
            list: The chunk list every rank's chunks for this FQN are appended to.

        Raises:
            ValueError: If another rank described this FQN with different properties or size.
        """
        info = fqn_info.get(fqn)
        if info is None:
            chunks = []
            fqn_info[fqn] = (properties, size, chunks)
            return chunks
        known_properties, known_size, chunks = info
        if known_properties != properties or known_size != size:
            raise ValueError(f"The {fqn} in different rank has different properties and size, "
                             f"properties: {known_properties} != {properties}, "
                             f"size: or {known_size} != {size}.")
        return chunks

    def finalize_plan(self, plan: SavePlan) -> SavePlan:
        """
        Finalize the plan.

        Args:
            plan (SavePlan): Plan to finalize.

        Returns:
            SavePlan: Finalized plan.
        """
        return plan

    def get_cached(self) -> Optional[CachedSaveResult]:
        """Return cached finalized plan and metadata when plan caching is enabled."""
        if (
            not self._enable_plan_caching
            or self._cached_plans_key not in StandardSavePlanner.cached_save_result
        ):
            return None
        return StandardSavePlanner.cached_save_result[self._cached_plans_key]

    def cache_result(self, final_plan: SavePlan, metadata: Metadata) -> None:
        """Store finalized plan and metadata in the class-level planner cache."""
        if not self._enable_plan_caching:
            return
        StandardSavePlanner.cached_save_result[self._cached_plans_key] = CachedSaveResult(
            final_plan=final_plan,
            metadata=metadata,
        )

    def get_data(self, item: WriteItem) -> Any:
        """
        Get current runtime data from state_dict for a write item.

        Args:
            item (WriteItem): Write item describing what to write.

        Returns:
            Any: Runtime object to be written.
        """
        if self.state_dict is None:
            raise RuntimeError("Planner not set up")
        fqn = item.index.fqn
        if fqn not in self.state_dict:
            raise KeyError(f"Key {fqn} not found in state_dict")
        obj = self.state_dict[fqn]
        if item.type == WriteItemType.TENSOR:
            if isinstance(obj, DTensor):
                if obj.layout is not None and obj.layout.ragged_shard is not None:
                    return platform.detach(get_ragged_box_tensor(obj, item.index)).to("cpu")
                return platform.detach(obj.to_local()).to("cpu")
            if isinstance(obj, Tensor):
                return platform.detach(obj).to("cpu")
            raise TypeError(f"Write item {fqn} expected tensor-like object, got {type(obj)}")
        if item.type == WriteItemType.BYTE_IO:
            return obj
        raise TypeError(f"Unsupported write item type: {item.type}")


def create_read_items_for_chunk_list(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: list[ChunkStorageMetadata],
    broadcastable: bool = True,
) -> list[ReadItem]:
    """
    Create ReadItems by matching local chunks (what this rank needs) with
    saved chunks (checkpoint_md.chunks), including resharding overlaps.

    Mirrors torch create_read_items_for_chunk_list behavior.

    Args:
        fqn (str): Fully qualified name of the tensor.
        checkpoint_md (TensorStorageMetadata): Tensor storage metadata from checkpoint.
        local_chunks (list[ChunkStorageMetadata]): List of local chunks needed by this rank.
        broadcastable (bool): Whether a collective could write into where these reads land.
            Default True; see :attr:`ReadItem.broadcastable`.

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
                    broadcastable=broadcastable,
                )
            )
    return read_items


def _is_on_host(obj: Any) -> bool:
    """
    Whether a state dict entry's local buffer sits in host memory rather than on the device.

    An entry that stayed on the host is read by every rank that wants it rather than read
    once and sent, because sending it is not worth what it costs and often is not possible
    at all. What stays on the host is little and few: an optimizer keeps its step counter
    there while the moments beside it follow the parameter onto the device, one fp32 scalar
    per parameter tensor, so a model of nine thousand of them leaves 36 KiB a rank -- read
    out of a file that is already open, beside the other shards of that file. The broadcast
    that would replace it costs a collective and two passes through a staging buffer, and
    on the host network of a large job that collective is a tree over TCP whose latency
    alone outweighs the read.

    And a load that broadcasts at all does so through groups raised on the accelerator
    library, which holds no backend for host memory and refuses a tensor kept there.

    What a broadcast carries is the local buffer, the same one :func:`_shard_buffer` hands
    the collective, so a DTensor is asked about its shard rather than about the whole tensor
    it is part of. An entry that does not say where it lives is taken to be where the rest
    of the load is.

    Args:
        obj (Any): State dict entry.

    Returns:
        bool: True only when the entry says it is in host memory.
    """
    local = obj.to_local() if isinstance(obj, DTensor) else obj
    return getattr(getattr(local, "device", None), "type", None) == "cpu"


class StandardLoadPlanner(LoadPlanner):
    """
    Standard implementation of LoadPlanner.

    Iterate state_dict and creates load plans via chunk list for resharding support.
    """

    def __init__(self, allow_partial_load: bool = False, broadcast_replicated_tensors: bool = False):
        """
        Args:
            allow_partial_load (bool): If True, allow loading when checkpoint has fewer keys than state_dict.
                Default False.
            broadcast_replicated_tensors (bool): If True, a tensor that several ranks load
                identically is read by one of them and sent to the rest. Off by default;
                ``configure_planner`` overrides this from :func:`load`.
        """
        self.state_dict: Optional[dict[str, Any]] = None
        self.metadata: Optional[Metadata] = None
        self.is_coordinator: bool = False
        self.rank: int = 0
        self.allow_partial_load = allow_partial_load
        self.broadcast_replicated_tensors: bool = broadcast_replicated_tensors
        self.flatten_state_dict: bool = True

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
        self.broadcast_replicated_tensors = kwargs.get(
            "broadcast_replicated_tensors", self.broadcast_replicated_tensors
        )
        self.flatten_state_dict = kwargs.get("flatten_state_dict", True)
        self.original_state_dict = state_dict
        if self.flatten_state_dict:
            state_dict, self.name_mapping = flatten_state_dict(state_dict)
        self.state_dict = state_dict

    def _rank_owns_dtensor_shard(self, obj: Any) -> bool:
        """
        Check whether the current rank appears in the rank list of a DTensor layout.

        Objects that are not DTensors, and DTensors whose layout does not carry a rank list,
        are always owned by the current rank.

        Args:
            obj (Any): State dict entry to check.

        Returns:
            bool: False only when the layout has a rank list the current rank is absent from.
        """
        if not isinstance(obj, DTensor):
            return True
        layout = getattr(obj, "layout", None)
        if layout is None:
            return True
        rank_list = getattr(layout, "rank_list", None) if layout else None
        if rank_list is None:
            rank_list = getattr(layout, "_rank_list", None)
        if rank_list is None:
            return True
        return self.rank in rank_list

    @dcp_timer_decorator
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
                if fqn.endswith(('matched_adamw_rms', 'step')):
                    continue
                if strict:
                    raise RuntimeError(f"Missing key in checkpoint state_dict: {fqn}.")
                continue
            md = self.metadata.state_dict_metadata[fqn]
            if isinstance(md, TensorStorageMetadata):
                obj_size = getattr(obj, CHUNK_INFO).global_shape if hasattr(obj, CHUNK_INFO) \
                    else getattr(obj, "shape", None)
                if obj_size is None or md.size != tuple(obj_size):
                    raise ValueError(
                        f"Size mismatch between saved {md.size} and current: {obj_size} for {fqn}",
                    )
                if not self._rank_owns_dtensor_shard(obj):
                    continue
                # Both DTensor and platform.Tensor: create local chunks and read items
                local_chunks = create_chunk_list_for_tensor(obj)
                requests += create_read_items_for_chunk_list(
                    fqn, md, local_chunks, broadcastable=not _is_on_host(obj),
                )
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

    @dcp_timer_decorator
    def build_global_plan(self, all_plans: list[LoadPlan]) -> LoadPlan:
        """
        Decide, across all ranks, who reads which shard and who is sent it instead.

        A shard held by several ranks needs only one of them to reach storage. This finds
        those groups in the gathered plans, picks the reader of each, and marks the items of
        this rank's plan with that decision. Shards are handed out one at a time and are not
        tied to one another: the two shards of a tensor go to two groups and two broadcasts,
        even when the same ranks hold both. Nothing is dropped from the plan either, since a
        rank that receives a shard still has to take part in the broadcast carrying it.

        Every rank runs this over the same gathered plans and reads only its own shards, so
        only this rank's plan is built. The assignment is a pure function of the gathered
        plans, so the ranks of a group name the same reader without one more round of
        communication.

        Args:
            all_plans (list[LoadPlan]): Local plan of every rank, indexed by global rank.

        Returns:
            LoadPlan: This rank's plan, with the items of its replicated shards marked.
        """
        own_plan = all_plans[_own_plan_index(all_plans, self.rank)]

        # A world of one has no one to broadcast with.
        if not self.broadcast_replicated_tensors or len(all_plans) < 2:
            return own_plan

        # Which ranks hold each shard, and how many elements reading it moves. dest_index
        # names the shard -- which local chunk of which tensor -- and ranks naming the same
        # one hold the same buffer, since a chunk and the checkpoint layout together settle
        # what has to be read for it. BYTE_IO items are left out: each rank rebuilds its
        # own pickled object, not storage a collective could write into.
        #
        # A shard some rank reported as not worth sending is left out for much the same
        # reason: what sits in host memory is little enough that reading it costs less than
        # moving it, and the group a broadcast would go through may hold no backend for it
        # anyway. One rank saying so is enough, and every rank reads the same report out of
        # the gather, so they all drop the same shards and go on naming the same readers
        # for the rest.
        replicas: dict[MetadataIndex, set[int]] = defaultdict(set)
        group_elements: dict[MetadataIndex, int] = defaultdict(int)
        unsendable: set[MetadataIndex] = set()
        for rank, plan in enumerate(all_plans):
            for item in plan.items:
                if item.type is LoadItemType.TENSOR:
                    replicas[item.dest_index].add(rank)
                    group_elements[item.dest_index] += math.prod(item.lengths)
                    if not item.broadcastable:
                        unsendable.add(item.dest_index)

        item_size = {
            fqn: dtype_element_size(getattr(getattr(md, "properties", None), "dtype", None))
            for fqn, md in self.metadata.state_dict_metadata.items()
        }
        # Every member of a group counted its own read into group_elements, and they all read
        # the same, so dividing by the group size gives back what the one reader will move.
        shards = [
            (item_size[index.fqn] * group_elements[index] // len(ranks), index, tuple(sorted(ranks)))
            for index, ranks in replicas.items() if len(ranks) > 1 and index not in unsendable
        ]
        if not shards:
            return own_plan

        # Heaviest shard first, each to the rank of its group that has been given the fewest
        # bytes so far: the costly reads are placed while the ranks are still even, and the
        # small ones fill the gaps afterwards. Shards are weighed rather than counted so
        # that one rank does not end up with all the large ones. Equal ones keep the order
        # replicas was filled in, which follows the gathered plans and so is the same on
        # every rank -- keep it that way, since ranks that disagreed here would name
        # different readers and hang on one another broadcasts.
        shards.sort(key=lambda shard: -shard[0])
        load_bytes = [0] * len(all_plans)
        sources: dict[MetadataIndex, BroadcastSource] = {}
        for nbytes, index, group_ranks in shards:
            src_rank = min(group_ranks, key=lambda rank: (load_bytes[rank], rank))
            load_bytes[src_rank] += nbytes
            sources[index] = BroadcastSource(group_ranks=group_ranks, src_rank=src_rank)

        logger.info(
            "[rank=%d] >>> %d replicated shards: busiest rank reads %d of %d bytes",
            self.rank, len(shards), max(load_bytes), sum(load_bytes),
        )

        # Mark this rank's items whether it reads or receives: the item is what tells a rank
        # which side of the broadcast it is on. A shard is held by one group, so an item
        # naming it is enough -- which rank the plan belongs to does not come into it.
        return dataclasses.replace(own_plan, items=[
            dataclasses.replace(item, source=sources[item.dest_index])
            if item.dest_index in sources else item
            for item in own_plan.items
        ])

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
        if (
                isinstance(target, DTensor)
                and target.layout is not None
                and target.layout.ragged_shard is not None
        ):
            box_tensor = get_ragged_box_tensor(target, read_item.dest_index)
            return narrow_tensor_by_index(
                box_tensor,
                read_item.dest_offsets,
                read_item.lengths,
            )

        local_tensor = target.to_local() if isinstance(target, DTensor) else target
        local_tensor = platform.detach(local_tensor)
        return narrow_tensor_by_index(
            local_tensor,
            read_item.dest_offsets,
            read_item.lengths,
        )

    def apply_tensor(self, read_item: ReadItem, tensor: Any) -> None:
        """
        Apply tensor after reading.

        Nothing to do: ``_copy_tensor_to_target`` already wrote the loaded slice into the
        destination, through ``copy_`` when the backend has one and through ``[...]``
        otherwise, so by the time this runs the state dict entry already holds the data.
        Kept as the hook other planners override.

        Args:
            read_item (ReadItem): The read item that was processed.
            tensor (Any): The tensor data to apply (tensor-like object).
        """

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
        if self.flatten_state_dict:
            set_element(self.original_state_dict, self.name_mapping[fqn], obj)



class _DcpMergeLoadPlanner(StandardLoadPlanner):
    """Load planner that builds distributed checkpoint from dcp into fully ``state_dict`` (in-place)."""

    def __init__(self) -> None:
        super().__init__()

    def configure_planner(self, state_dict: dict[str, Any], metadata: Metadata, **kwargs) -> None:
        if len(state_dict) > 0:
            raise ValueError(
                "state_dict must be empty for _DcpMergeLoadPlanner; "
                "it is populated in-place from checkpoint metadata."
            )

        if metadata is None:
            raise ValueError("metadata must not be None for _DcpMergeLoadPlanner.")

        self.is_coordinator = kwargs.get("is_coordinator", False)
        for k, v in metadata.state_dict_metadata.items():
            if isinstance(v, TensorStorageMetadata):
                v = platform.empty(
                    platform.list_to_size(v.size),
                    dtype=platform.str_to_dtype(v.properties.dtype),
                )

            state_dict[k] = v
            if metadata.planner_data is not None and k in metadata.planner_data:
                set_element(state_dict, metadata.planner_data[k], v)

        super().configure_planner(
            state_dict,
            metadata,
            is_coordinator=self.is_coordinator,
            flatten_state_dict=True,
        )
