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
"""Parameter grouping for distributed Muon optimizer.

Groups DTensor parameters by their sharding strategy to determine which
parameters need communication (all-gather before Newton-Schulz iteration)
and which can be updated locally.  For communication-requiring groups,
computes the replicate rank list (communication domain) so that ranks
holding the same shard can exchange data.

Only uses the self-developed DTensor / DeviceMesh / Placement APIs —
never PyTorch's official DTensor.
"""

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Placement, Replicate, Shard


@dataclass
class ShardInfo:
    """Complete sharding metadata extracted from a DTensor.

    Attributes:
        tensor_ndim: Number of dimensions of the global tensor.
        placements: The placement for each mesh dimension
            (length == device_mesh.ndim).
        device_mesh: The DeviceMesh the tensor is distributed over.
        shard_dims: Set of tensor dimensions that are sharded
            (i.e. every ``Shard(dim).dim`` across all mesh dims).
        replicate_mesh_dims: Indices of mesh dimensions whose placement
            is ``Replicate``.
    """

    tensor_ndim: int
    placements: Sequence[Placement]
    device_mesh: DeviceMesh
    shard_dims: set = field(default_factory=set)
    replicate_mesh_dims: list = field(default_factory=list)


@dataclass
class CommParamGroup:
    """A group of DTensor parameters that share identical sharding.

    Attributes:
        params: Parameters in this group.
        shard_info: The common ``ShardInfo`` for all members.
        replicate_group: Sorted rank list forming the communication
            domain — ranks that hold identical copies of each shard.
    """

    params: List[DTensor] = field(default_factory=list)
    shard_info: Optional[ShardInfo] = None
    replicate_group: List[int] = field(default_factory=list)


def _validate_param_ndim(dtensor: DTensor) -> None:
    """Validate that the parameter has at least 2 dimensions.

    Muon's Newton-Schulz iteration operates on matrices (2-D tensors);
    1-D parameters such as biases and layer-norm scales cannot participate
    and must be handled by a different optimizer (e.g. AdamW).

    Args:
        dtensor: Input DTensor parameter.

    Raises:
        ValueError: If the tensor has fewer than 2 dimensions.
    """
    ndim = len(dtensor.shape)
    if ndim < 2:
        raise ValueError(
            f"Muon optimizer requires parameters with at least 2 dimensions, "
            f"but got a {ndim}-D parameter with shape {dtensor.shape}. "
            f"1-D parameters (biases, norm scales, etc.) should use a "
            f"different optimizer such as AdamW."
        )


def extract_shard_info(dtensor: DTensor) -> ShardInfo:
    """Extract complete sharding information from a DTensor.

    Args:
        dtensor: Input DTensor parameter.  Must have at least 2
            dimensions — 1-D parameters are incompatible with Muon.

    Returns:
        ShardInfo with tensor dimensionality, placements, mesh,
        the set of sharded tensor dims, and the list of replicated
        mesh-dim indices.

    Raises:
        ValueError: If the tensor has fewer than 2 dimensions.
    """
    _validate_param_ndim(dtensor)
    device_mesh = dtensor.device_mesh
    placements = dtensor.placements
    tensor_ndim = len(dtensor.shape)

    shard_dims: set = set()
    replicate_mesh_dims: list = []

    for mesh_dim_idx, placement in enumerate(placements):
        if placement.is_replicate():
            replicate_mesh_dims.append(mesh_dim_idx)
        elif placement.is_shard():
            shard_dims.add(placement.dim)

    return ShardInfo(
        tensor_ndim=tensor_ndim,
        placements=placements,
        device_mesh=device_mesh,
        shard_dims=shard_dims,
        replicate_mesh_dims=replicate_mesh_dims,
    )


def calculate_replicate_group(
        dtensor: DTensor,
        shard_info: Optional[ShardInfo] = None,
) -> List[int]:
    """Compute the replicate rank list (communication domain) for a DTensor.

    The replicate group is the set of ranks that hold identical copies of
    every shard of this tensor.  When multiple mesh dimensions are
    replicated, the result is the Cartesian product of the per-dimension
    coordinate ranges — i.e. all ranks reachable by independently varying
    the current rank's coordinates along each replicated mesh dimension.

    Args:
        dtensor: Input DTensor parameter.
        shard_info: Optional pre-computed ``ShardInfo`` to avoid
            redundant extraction.

    Returns:
        Sorted list of ranks that replicate the same data.  If no mesh
        dimension is replicated, returns ``[device_mesh.rank]``.
    """
    if shard_info is None:
        shard_info = extract_shard_info(dtensor)

    device_mesh = shard_info.device_mesh

    if not shard_info.replicate_mesh_dims:
        return [device_mesh.rank]

    mesh_shape = device_mesh.mesh_shape
    rank_list = device_mesh.rank_list
    current_rank = device_mesh.rank
    ndim = len(mesh_shape)

    # Compute the current rank's multi-dimensional coordinate.
    idx = rank_list.index(current_rank)
    coord = [0] * ndim
    temp = idx
    for i in range(ndim - 1, -1, -1):
        coord[i] = temp % mesh_shape[i]
        temp //= mesh_shape[i]

    # Row-major strides for converting coordinates back to flat index.
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * mesh_shape[i + 1]

    # Cartesian product of coordinate ranges along replicated mesh dims.
    dim_ranges = [range(mesh_shape[d]) for d in shard_info.replicate_mesh_dims]
    replicate_ranks: set = set()
    for combo in itertools.product(*dim_ranges):
        new_coord = coord.copy()
        for dim_idx, val in zip(shard_info.replicate_mesh_dims, combo):
            new_coord[dim_idx] = val
        flat_idx = sum(new_coord[i] * strides[i] for i in range(ndim))
        replicate_ranks.add(rank_list[flat_idx])

    return sorted(replicate_ranks)


def _is_no_comm_param(shard_info: ShardInfo) -> bool:
    """Return True when the parameter does not need communication.

    A parameter is no-comm when none of its sharded dimensions touch
    the last two dimensions of the tensor.  This covers:
    - Fully replicated parameters (shard_dims is empty)
    - Parameters sharded only on dims < tensor_ndim - 2

    Note: 1-D parameters are rejected upstream by ``_validate_param_ndim``
    and never reach this function.
    """
    if not shard_info.shard_dims:
        return True
    # Last two dims are at indices (ndim-2) and (ndim-1).
    last_two = {shard_info.tensor_ndim - 2, shard_info.tensor_ndim - 1}
    return shard_info.shard_dims.isdisjoint(last_two)


def _placements_key(placements: Sequence[Placement]) -> tuple:
    """Build a hashable key from a placements sequence for grouping.

    Two placements are "identical" when they have the same type and
    the same parameters (e.g. ``Shard(0)`` == ``Shard(0)``,
    ``Replicate()`` == ``Replicate()``).
    """
    parts: list = []
    for p in placements:
        if p.is_shard():
            parts.append(("Shard", p.dim))
        elif p.is_replicate():
            parts.append(("Replicate",))
        elif p.is_partial():
            parts.append(("Partial", p.reduce_op))
        else:
            parts.append((type(p).__name__,))
    return tuple(parts)


def group_parameters_by_sharding(
        params: List[DTensor],
) -> Tuple[List[DTensor], List[CommParamGroup]]:
    """Group DTensor parameters by their sharding strategy.

    Two-level classification:
      1. **no_comm_params** — sharding does not involve the last two
         tensor dimensions (fully replicated, or sharded only on
         earlier dims).
      2. **comm_params_same_shard** — sharding touches at least one of
         the last two dims; further sub-grouped so that parameters with
         identical placements share one ``CommParamGroup``.

    Args:
        params: All DTensor parameters of the model.  Every parameter
            must have at least 2 dimensions — 1-D parameters (biases,
            norm scales, etc.) are incompatible with Muon and will
            raise ``ValueError``.

    Returns:
        A tuple ``(no_comm_params, comm_params_same_shard)`` where
        ``no_comm_params`` is a flat list and
        ``comm_params_same_shard`` is a list of ``CommParamGroup``
        objects, each carrying its replicate group.

    Raises:
        ValueError: If any parameter has fewer than 2 dimensions.
    """
    no_comm_params: List[DTensor] = []
    # key (placements tuple) -> CommParamGroup
    comm_groups: dict = {}

    for param in params:
        shard_info = extract_shard_info(param)

        if _is_no_comm_param(shard_info):
            no_comm_params.append(param)
            continue

        key = _placements_key(shard_info.placements)

        if key not in comm_groups:
            # First param with this sharding pattern — compute the
            # replicate group once and reuse for all members.
            replicate_group = calculate_replicate_group(param, shard_info)
            comm_groups[key] = CommParamGroup(
                shard_info=shard_info,
                replicate_group=replicate_group,
            )

        comm_groups[key].params.append(param)

    return no_comm_params, list(comm_groups.values())
