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
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Optional

import torch.distributed as dist

from hyper_parallel.core.optimizer.dtensor_compat import (
    DTensor,
    DeviceMesh,
    Shard,
    StridedShard,
)

logger = logging.getLogger(__name__)


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
    shard_meta: Optional["ParamShardMeta"] = None


@dataclass(frozen=True)
class ParamShardStageMeta:
    """Per-stage shard metadata for uneven all-gather and slicing."""

    tensor_dim: int
    shard_size: int
    cur_rank_in_shard_group: int
    rank_shapes: Tuple[Tuple[int, ...], ...]
    split_sizes: Tuple[int, ...]
    pad_shape: Tuple[int, ...]


@dataclass(frozen=True)
class ParamShardMeta:
    """Runtime shard metadata for one DTensor parameter."""

    global_shape: Tuple[int, ...]
    global_numel: int
    local_shape: Tuple[int, ...]
    local_numel: int
    stage_metas: Tuple[ParamShardStageMeta, ...]


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
        mesh_shape=tuple(getattr(shard_spec.device_mesh, "mesh_shape", None) or shard_spec.device_mesh.mesh.shape),
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

    @staticmethod
    def _format_device_mesh(device_mesh: Optional[DeviceMesh]) -> Optional[str]:
        """Format device mesh as DeviceMesh((name=size, ...)) for debug logs."""
        if device_mesh is None:
            return None

        mesh_shape = tuple(getattr(device_mesh, "mesh_shape", None) or device_mesh.mesh.shape)
        mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None)

        if mesh_dim_names:
            dims_desc = ", ".join(
                f"{name}={size}" for name, size in zip(mesh_dim_names, mesh_shape)
            )
        else:
            dims_desc = ", ".join(
                f"dim_{idx}={size}" for idx, size in enumerate(mesh_shape)
            )

        return f"DeviceMesh(({dims_desc}))"

    def __str__(self) -> str:
        shard_ranks = [list(dist.get_process_group_ranks(pg)) for pg in self.shard_pgs if pg is not None]
        replicate_ranks = [list(dist.get_process_group_ranks(pg)) for pg in self.replicate_pgs if pg is not None]

        owned_names = [getattr(p, "model_name", f"p_{i}") for i, p in enumerate(self.owned_params)]
        all_names = [getattr(p, "model_name", f"p_{i}") for i, p in enumerate(self.all_params)]
        device_mesh = self.all_records[0].param.device_mesh if self.all_records else None
        device_mesh_str = self._format_device_mesh(device_mesh)

        return (
            f"HSDPGroupAssignment( \n"
            f"owned_params={owned_names}, \n"
            f"all_params={all_names}, \n"
            f"replicate_pg_ranks={replicate_ranks}, \n"
            f"replicate_sizes={self.replicate_sizes}, \n"
            f"is_shard={self.is_shard}, \n"
            f"shard_pg_ranks={shard_ranks}, \n"
            f"device_mesh={device_mesh_str}, \n"
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
