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

"""Category parameter with dtensor."""

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional, Any

import torch
import torch.distributed as dist

from hyper_parallel.core.optimizer.dtensor_compat import (
    DTensor,
    DeviceMesh,
    Shard,
    StridedShard,
)

@dataclass(frozen=True)
class ParamLayoutSpec:
    """Shape-free but ndim-aware parameter layout.

    Same layout means:
        1. same tensor ndim;
        2. same shard mesh dims;
        3. same shard tensor dims;
        4. same replicate mesh dims.

    Example:
        tensor_ndim = 3
        placements = (Shard(0), Replicate(), Replicate(), Replicate())

    Then:
        shard_axes = ((0, 0),)
        replicate_mesh_dims = (1, 2, 3)

    Meaning:
        mesh dim 0 shards tensor dim 0;
        mesh dim 1/2/3 are replicated.
    """

    tensor_ndim: int

    # Each item is:
    #   (mesh_dim, tensor_dim)
    shard_axes: Tuple[Tuple[int, int], ...]

    # Mesh dims that are replicated.
    replicate_mesh_dims: Tuple[int, ...]

    @property
    def shard_mesh_dims(self) -> Tuple[int, ...]:
        """Return the mesh dimensions used for sharding."""
        return tuple(mesh_dim for mesh_dim, _ in self.shard_axes)

    @property
    def shard_tensor_dims(self) -> Tuple[int, ...]:
        """Return the tensor dimensions that are sharded."""
        return tuple(tensor_dim for _, tensor_dim in self.shard_axes)

    @property
    def is_last2d_sharded(self) -> bool:
        """Whether any shard axis falls on the last 2 tensor dimensions.

        Newton-Schulz iteration operates on the last 2 dims.
        If either dim is sharded, allgather is needed before NS.
        """
        for _, tensor_dim in self.shard_axes:
            if tensor_dim >= self.tensor_ndim - 2:
                return True
        return False


@dataclass(frozen=True)
class CommDomainKey:
    """Communication-domain key based on mesh dimensions.

    This key describes the communication domain logically by relying on the
    underlying DeviceMesh identifiers and the specific dims involved.
    """

    mesh_shape: Tuple[int, ...]
    mesh_rank_list: Tuple[int, ...]

    replicate_mesh_dims: Tuple[int, ...]
    shard_mesh_dims: Tuple[int, ...]

    @property
    def has_replicate_redundancy(self) -> bool:
        """Check if there is any replicate redundancy in the mesh."""
        return len(self.replicate_mesh_dims) > 0

    @property
    def has_shard_group(self) -> bool:
        """Check if there are any sharded mesh dimensions."""
        return len(self.shard_mesh_dims) > 0


@dataclass(frozen=True)
class HSDPGroupKey:
    """Final grouping key for HSDP.

    Same key means:
        1. same communication domain;
        2. same tensor ndim;
        3. same shard / replicate axis layout.

    Shape is not part of this key.
    """

    comm_key: CommDomainKey
    axis_spec: ParamLayoutSpec


@dataclass
class ParamShardSpec:
    """Per-parameter shard metadata used during HSDP grouping."""

    device_mesh: DeviceMesh
    shard_mesh_dims: Tuple[int, ...]
    replicate_mesh_dims: Tuple[int, ...]
    replicate_pgs: Tuple[dist.ProcessGroup, ...]
    shard_pgs: Tuple[dist.ProcessGroup, ...]
    axis_spec: ParamLayoutSpec


@dataclass(frozen=True)
class ParamRecord:
    """A lightweight parameter record."""

    index: int
    param: DTensor


@dataclass
class HSDPCommGroup:
    """HSDP communication group.

    Same group means same domain, ndim, and layout.
    Stores sequences of ProcessGroups aligned with mesh dims.
    """

    comm_key: CommDomainKey
    layout_spec: ParamLayoutSpec

    # Tuple of runtime process groups, one for each relevant mesh dimension
    replicate_pgs: Tuple[dist.ProcessGroup, ...] = ()
    shard_pgs: Tuple[dist.ProcessGroup, ...] = ()

    records: List[ParamRecord] = field(default_factory=list)

    @property
    def params(self) -> List[DTensor]:
        """Return all DTensors within this communication group."""
        return [record.param for record in self.records]

    def add_param(self, index: int, param: DTensor) -> None:
        """Add a parameter record to the group."""
        self.records.append(
            ParamRecord(
                index=index,
                param=param,
            )
        )

    def __len__(self) -> int:
        return len(self.records)


def _normalize_tensor_dim(dim: int, tensor_ndim: int) -> int:
    """Normalize tensor dim to non-negative dim."""
    original_dim = dim

    if dim < 0:
        dim += tensor_ndim

    if dim < 0 or dim >= tensor_ndim:
        raise ValueError(
            f"Invalid shard dim {original_dim} for tensor ndim {tensor_ndim}."
        )

    return dim


def extract_param_shard_spec(dtensor: DTensor) -> ParamShardSpec:
    """Extract all shard metadata and native process groups from one DTensor."""
    device_mesh = dtensor.device_mesh
    placements = dtensor.placements
    tensor_ndim = len(dtensor.shape)

    shard_mesh_dims: List[int] = []
    replicate_mesh_dims: List[int] = []
    shard_axes: List[Tuple[int, int]] = []
    replicate_pgs: List[dist.ProcessGroup] = []
    shard_pgs: List[dist.ProcessGroup] = []

    for mesh_dim_idx, placement in enumerate(placements):
        pg = device_mesh.get_group(mesh_dim_idx) if hasattr(device_mesh, "get_group") else None

        if placement.is_replicate():
            replicate_mesh_dims.append(mesh_dim_idx)
            replicate_pgs.append(pg)

        elif placement.is_shard():
            if isinstance(placement, StridedShard):
                placement_for_grouping = Shard(placement.dim)
            else:
                placement_for_grouping = placement

            tensor_dim = _normalize_tensor_dim(
                placement_for_grouping.dim,
                tensor_ndim,
            )

            shard_mesh_dims.append(mesh_dim_idx)
            shard_axes.append((mesh_dim_idx, tensor_dim))
            shard_pgs.append(pg)

        else:
            raise ValueError(
                f"Unsupported placement type in HSDP parameter grouping: "
                f"{type(placement).__name__}."
            )

    axis_spec = ParamLayoutSpec(
        tensor_ndim=tensor_ndim,
        shard_axes=tuple(shard_axes),
        replicate_mesh_dims=tuple(replicate_mesh_dims),
    )

    return ParamShardSpec(
        device_mesh=device_mesh,
        shard_mesh_dims=tuple(shard_mesh_dims),
        replicate_mesh_dims=tuple(replicate_mesh_dims),
        replicate_pgs=tuple(replicate_pgs),
        shard_pgs=tuple(shard_pgs),
        axis_spec=axis_spec,
    )


def build_comm_domain_key(shard_spec: ParamShardSpec) -> CommDomainKey:
    """Build grouping key utilizing DeviceMesh properties."""
    mesh_rank_list = ()
    if hasattr(shard_spec.device_mesh, "rank_list"):
        mesh_rank_list = tuple(shard_spec.device_mesh.rank_list)

    return CommDomainKey(
        mesh_shape = tuple(getattr(shard_spec.device_mesh, "mesh_shape", None) or shard_spec.device_mesh.mesh.shape),
        mesh_rank_list=mesh_rank_list,
        replicate_mesh_dims=shard_spec.replicate_mesh_dims,
        shard_mesh_dims=shard_spec.shard_mesh_dims,
    )


def group_parameters_for_hsdp(
        params: List[DTensor],
) -> Tuple[List[DTensor], List[HSDPCommGroup]]:
    """Group parameters relying on native DeviceMesh topology."""
    no_comm_params: List[DTensor] = []
    groups: Dict[HSDPGroupKey, HSDPCommGroup] = {}

    for param_index, param in enumerate(params):
        if not isinstance(param, DTensor):
            no_comm_params.append(param)
            continue

        shard_spec = extract_param_shard_spec(param)
        comm_key = build_comm_domain_key(shard_spec)

        if not comm_key.has_replicate_redundancy and not comm_key.has_shard_group:
            no_comm_params.append(param)
            continue

        group_key = HSDPGroupKey(
            comm_key=comm_key,
            axis_spec=shard_spec.axis_spec,  # ParamLayoutSpec
        )

        if group_key not in groups:
            groups[group_key] = HSDPCommGroup(
                comm_key=comm_key,
                layout_spec=shard_spec.axis_spec,
                replicate_pgs=shard_spec.replicate_pgs,
                shard_pgs=shard_spec.shard_pgs,
            )

        groups[group_key].add_param(param_index, param)

    return no_comm_params, list(groups.values())


@dataclass
class HSDPGroupAssignment:
    """Optimizer assignment for one HSDP communication group."""

    owned_records: List[ParamRecord]
    all_records: List[ParamRecord]

    # param_index -> (dim_0_rank, dim_1_rank, ...)
    owner_by_index: Dict[int, Tuple[int, ...]]

    # Record the rank and size for each dimension, and the list of cur_rank within replicate_groups.
    replicate_group_ranks: Tuple[int, ...]
    replicate_sizes: Tuple[int, ...]

    replicate_pgs: Tuple[dist.ProcessGroup, ...] = ()
    shard_pgs: Tuple[dist.ProcessGroup, ...] = ()

    is_shard: bool = False
    layout_spec: Optional[ParamLayoutSpec] = None

    @property
    def owned_params(self) -> List[DTensor]:
        """Return the parameters owned by the current rank."""
        return [record.param for record in self.owned_records]

    @property
    def all_params(self) -> List[DTensor]:
        """Return all parameters in this assignment group."""
        return [record.param for record in self.all_records]

    @property
    def is_replicated(self) -> bool:
        """Check if the group spans across multiple ranks."""
        return any(s > 1 for s in self.replicate_sizes)

    def owner_rank_coord(self, record: ParamRecord) -> Tuple[int, ...]:
        """Get the rank coordinates of the owner of a given record."""
        return self.owner_by_index.get(record.index, ())

    def is_owned(self, record: ParamRecord) -> bool:
        """Check if a given record is owned by the current replicate group rank."""
        return self.owner_rank_coord(record) == self.replicate_group_ranks

    def __str__(self) -> str:
        shard_ranks = [list(dist.get_process_group_ranks(pg)) for pg in self.shard_pgs if pg is not None]
        replicate_ranks = [list(dist.get_process_group_ranks(pg)) for pg in self.replicate_pgs if pg is not None]

        owned_names = [getattr(p, "model_name", f"p_{i}") for i, p in enumerate(self.owned_params)]
        all_names = [getattr(p, "model_name", f"p_{i}") for i, p in enumerate(self.all_params)]

        return (
            f"HSDPGroupAssignment( \n"
            f"owned_params={owned_names}, \n"
            f"all_params={all_names}, \n"
            f"replicate_group_ranks={self.replicate_group_ranks}, \n"
            f"replicate_sizes={self.replicate_sizes}, \n"
            f"is_shard={self.is_shard}, \n"
            f"shard_pg_ranks={shard_ranks}, \n"
            f"replicate_pg_ranks={replicate_ranks}, \n"
            f"layout_spec={self.layout_spec}) \n"
        )

    __repr__ = __str__


def get_multi_dim_logical_info(
        device_mesh: DeviceMesh,
        mesh_dims: Sequence[int]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Obtain the independent relative ranks and sizes of the parameters across multiple replicate dimensions."""
    if not mesh_dims:
        return (), ()

    coords = device_mesh.get_coordinate()
    if coords is None:
        return (-1,) * len(mesh_dims), (1,) * len(mesh_dims)

    ranks = tuple(coords[dim] for dim in mesh_dims)
    sizes = tuple(device_mesh.size(dim) for dim in mesh_dims)

    return ranks, sizes


def build_owner_by_size(
        records: List[ParamRecord],
        replicate_sizes: Tuple[int, ...],
) -> Dict[int, Tuple[int, ...]]:
    """Build deterministic owner map across a multi-dimensional replicate grid."""
    valid_records = [
        record for record in records
        if getattr(record.param, "requires_grad", True)
    ]

    if not valid_records:
        return {}

    if not replicate_sizes:
        return {record.index: () for record in valid_records}

    dim_ranges = [range(s) for s in replicate_sizes]
    all_coords = list(itertools.product(*dim_ranges))

    sorted_records = sorted(
        valid_records,
        key=lambda record: (-record.param.numel(), record.index),
    )

    # The greedy strategy assigns the task to the node with the lowest load.
    coord_loads = {coord: 0 for coord in all_coords}
    owner_by_index: Dict[int, Tuple[int, ...]] = {}

    for record in sorted_records:
        best_coord = min(
            all_coords,
            key=lambda c: (coord_loads[c], c),
        )

        owner_by_index[record.index] = best_coord
        coord_loads[best_coord] += record.param.numel()

    return owner_by_index


def select_owned_records(
        records: List[ParamRecord],
        owner_by_index: Dict[int, Tuple[int, ...]],
        replicate_group_ranks: Tuple[int, ...],
) -> List[ParamRecord]:
    """Select records owned by current multi-dimensional replicate rank."""
    if any(r < 0 for r in replicate_group_ranks):
        return []

    return [
        record for record in records
        if owner_by_index.get(record.index) == replicate_group_ranks
    ]


def chunk_update_by_layout(
        global_update: torch.Tensor,
        param: "DTensor",
        layout_spec: "ParamLayoutSpec",
) -> torch.Tensor:
    """Slice a full update back to the local shard using narrow."""
    if not hasattr(param, "device_mesh") or layout_spec is None or not layout_spec.shard_axes:
        return global_update

    device_mesh = param.device_mesh
    mesh_coordinates = device_mesh.get_coordinate()

    shard_axes = layout_spec.shard_axes
    local_update = global_update

    # Apply each shard axis in order. `narrow` returns a view and avoids
    # creating all chunks when only the local rank's chunk is needed.
    for mesh_dim, tensor_dim in shard_axes:
        num_chunks = device_mesh.size(mesh_dim)

        if num_chunks <= 1:
            continue

        local_rank = mesh_coordinates[mesh_dim]
        chunk_size = local_update.size(tensor_dim) // num_chunks

        local_update = local_update.narrow(tensor_dim, local_rank * chunk_size, chunk_size)

    # Non-dim0 slicing may produce a non-contiguous view.
    if not local_update.is_contiguous():
        local_update = local_update.contiguous()

    return local_update


def _get_or_alloc_buffer(
        cache: Optional[Dict],
        key: Any,
        numel: int,
        dtype: torch.dtype,
        device: torch.device,
) -> torch.Tensor:
    """Return a cached buffer with at least `numel` elements."""
    if cache is not None and key in cache:
        buf = cache[key]
        if buf.numel() >= numel:
            return buf
        buf = torch.empty(numel, dtype=dtype, device=device)
        cache[key] = buf
        return buf

    buf = torch.empty(numel, dtype=dtype, device=device)
    if cache is not None:
        cache[key] = buf
    return buf


def _early_return_tensors(
        local_tensors: List[torch.Tensor],
        keep_indices: Optional[set],
) -> List[Optional[torch.Tensor]]:
    """Return tensors directly when no communication is needed."""
    if keep_indices is None:
        return list(local_tensors)
    return [t if i in keep_indices else None for i, t in enumerate(local_tensors)]


def _prepare_gather_inputs(
        current_tensors: List[torch.Tensor],
        tensor_dim: int,
        alignment_elements: int,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int, Tuple[int, ...]]], int]:
    """Move shard dim to dim0 and compute padding metadata.

    Returns:
        (gather_inputs, param_meta, total_padded_numel)
        param_meta item: (offset, actual_numel, padded_numel, rest_shape)
    """
    gather_inputs: List[torch.Tensor] = []
    param_meta: List[Tuple[int, int, int, Tuple[int, ...]]] = []
    total_padded_numel = 0

    for t in current_tensors:
        tensor_dim_norm = tensor_dim % t.dim()

        # Put shard dim at dim0 so all-gather can concatenate along dim0.
        if tensor_dim_norm == 0 and t.is_contiguous():
            gi = t
        else:
            gi = t.movedim(tensor_dim_norm, 0).contiguous()

        actual_numel = gi.numel()
        padded_numel = ((actual_numel + alignment_elements - 1) // alignment_elements) * alignment_elements

        gather_inputs.append(gi)
        param_meta.append((total_padded_numel, actual_numel, padded_numel, tuple(gi.shape[1:])))
        total_padded_numel += padded_numel

    return gather_inputs, param_meta, total_padded_numel


def _pack_and_allgather(
        gather_inputs: List[torch.Tensor],
        param_meta: List[Tuple[int, int, int, Tuple[int, ...]]],
        total_padded_numel: int,
        axis_idx: int,
        dtype: torch.dtype,
        device: torch.device,
        shard_pg: dist.ProcessGroup,
        shard_size: int,
        buffer_cache: Optional[Dict],
) -> torch.Tensor:
    """Pack local shards into one buffer, all-gather, return gathered view.

    Returns:
        gathered_view with shape [shard_size, total_padded_numel]
    """
    cache_key = ("fused_allgather", axis_idx, dtype, device)
    pack_buffer = _get_or_alloc_buffer(
        buffer_cache, cache_key, total_padded_numel,
        dtype, device,
    )[:total_padded_numel]

    # Copy only real data. Padding is not zeroed because it is never read.
    for gi, (offset, actual_numel, _, _) in zip(gather_inputs, param_meta):
        pack_buffer[offset:offset + actual_numel].copy_(gi.view(-1))

    gathered_numel = total_padded_numel * shard_size
    cache_key_out = ("fused_allgather_out", axis_idx, dtype, device)
    gathered_buffer = _get_or_alloc_buffer(
        buffer_cache, cache_key_out, gathered_numel,
        dtype, device,
    )[:gathered_numel]

    dist.all_gather_into_tensor(gathered_buffer, pack_buffer, group=shard_pg)

    # Rank-major layout: [rank0_pack | rank1_pack | ...]
    return gathered_buffer.view(shard_size, total_padded_numel)


def _unpack_gathered_results(
        gathered_view: torch.Tensor,
        gather_inputs: List[torch.Tensor],
        param_meta: List[Tuple[int, int, int, Tuple[int, ...]]],
        current_tensors: List[torch.Tensor],
        tensor_dim: int,
        shard_size: int,
        n_params: int,
        is_last_axis: bool,
        keep_indices: Optional[set],
) -> List[Optional[torch.Tensor]]:
    """Slice gathered buffer back to per-parameter full tensors."""
    new_tensors: List[Optional[torch.Tensor]] = []
    for i in range(n_params):
        # Only the final output can be skipped.
        # Earlier axis results may be needed by the next shard-axis gather.
        if is_last_axis and keep_indices is not None and i not in keep_indices:
            new_tensors.append(None)
            continue

        offset, actual_numel, _, rest_shape = param_meta[i]
        dim0_size = gather_inputs[i].shape[0]

        # Pick this parameter from every rank.
        # This slice is usually non-contiguous because data is rank-major.
        param_slice = gathered_view[:, offset:offset + actual_numel]

        # Materialize as contiguous: [rank0_param | rank1_param | ...]
        param_data = param_slice.contiguous()

        # Restore full tensor with gathered dim0.
        result = param_data.view(dim0_size * shard_size, *rest_shape)

        # Move dim0 back to the original shard dimension.
        tensor_dim_norm = tensor_dim % current_tensors[i].dim()
        if tensor_dim_norm == 0:
            new_tensors.append(result)
        else:
            new_tensors.append(result.movedim(0, tensor_dim_norm))

    return new_tensors


def fused_allgather_dtensor_params(
        local_tensors: List[torch.Tensor],
        shard_pgs: Sequence[dist.ProcessGroup],
        layout_spec: ParamLayoutSpec,
        buffer_cache: Optional[Dict] = None,
        keep_indices: Optional[set] = None,
) -> List[Optional[torch.Tensor]]:
    """Fuse many parameter shards into one all-gather per shard axis.

    Flow:
      1. Move shard dim to dim0 if needed.
      2. Pack all local shards into one flat buffer.
      3. Run one all-gather on the packed buffer.
      4. Slice gathered buffer back to per-parameter full tensors.
      5. On the last shard axis, only unpack `keep_indices`.
    """
    if not shard_pgs or not local_tensors:
        return _early_return_tensors(local_tensors, keep_indices)

    n_params = len(local_tensors)
    device = local_tensors[0].device
    dtype = local_tensors[0].dtype
    alignment_bytes = 512
    element_size = local_tensors[0].element_size()
    alignment_elements = max(1, alignment_bytes // element_size)

    # Keep only real shard axes that need communication.
    active_axes = []
    for axis_idx, ((_, tensor_dim), shard_pg) in enumerate(zip(layout_spec.shard_axes, shard_pgs)):
        if shard_pg is None:
            continue
        shard_size = dist.get_world_size(shard_pg)
        if shard_size <= 1:
            continue
        active_axes.append((axis_idx, tensor_dim, shard_pg, shard_size))

    if not active_axes:
        return _early_return_tensors(local_tensors, keep_indices)

    # After each shard axis, this becomes the partially gathered result.
    current_tensors: List[torch.Tensor] = list(local_tensors)

    for active_pos, (axis_idx, tensor_dim, shard_pg, shard_size) in enumerate(active_axes):
        is_last_axis = active_pos == len(active_axes) - 1

        gather_inputs, param_meta, total_padded_numel = _prepare_gather_inputs(
            current_tensors, tensor_dim, alignment_elements,
        )

        gathered_view = _pack_and_allgather(
            gather_inputs, param_meta, total_padded_numel,
            axis_idx, dtype, device, shard_pg, shard_size, buffer_cache,
        )

        new_tensors = _unpack_gathered_results(
            gathered_view, gather_inputs, param_meta,
            current_tensors, tensor_dim, shard_size,
            n_params, is_last_axis, keep_indices,
        )

        if is_last_axis:
            return new_tensors

        # Safe because keep_indices is only applied on the last axis.
        current_tensors = new_tensors  # type: ignore[assignment]

    return current_tensors
