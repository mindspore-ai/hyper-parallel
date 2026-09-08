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
"""Lower TP placement transitions to differentiable local-tensor collectives."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from hyper_parallel.core.dtensor.placement_types import Partial, Placement, Replicate, Shard
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)
platform = get_platform()


@dataclass
class TPExecutionOp:
    """Executable local-tensor operation for one TP placement transition."""

    kind: str
    group: object
    group_size: int
    group_rank: int
    tensor_dim: Optional[int] = None
    reduce_op: str = "sum"

    def execute(self, tensor: Any) -> Any:
        """Execute the differentiable collective selected during lowering."""
        if self.kind == "all_gather":
            return platform.differentiable_all_gather_concat(
                tensor,
                self.group,
                self.group_size,
                self.tensor_dim,
            )
        if self.kind == "all_reduce":
            return platform.differentiable_all_reduce(
                tensor,
                self.reduce_op,
                self.group,
            )
        if self.kind == "reduce_scatter":
            return platform.differentiable_reduce_scatter(
                tensor,
                self.group_size,
                self.tensor_dim,
                self.reduce_op,
                self.group,
            )
        if self.kind == "all_reduce_shard":
            reduced = platform.differentiable_all_reduce(
                tensor,
                self.reduce_op,
                self.group,
            )
            return platform.chunk(
                reduced,
                self.tensor_dim,
                self.group_size,
                self.group_rank,
            )
        raise ValueError(f"Unsupported TP execution operation: {self.kind!r}")


@dataclass(frozen=True)
class TPCollectiveLowerer:
    """Lower supported TP-only placement differences to execution operations."""

    mesh_dim_names: tuple[str, ...]
    group: object
    group_size: int
    group_rank: int
    backend: str

    def __call__(
        self,
        src: Sequence[Placement],
        dst: Sequence[Placement],
    ) -> Optional[TPExecutionOp]:
        """Return an execution operation or ``None`` for the generic fallback."""
        if tuple(src) == tuple(dst):
            return None

        tp_axis = self.mesh_dim_names.index("tp")
        if any(
            axis != tp_axis and src_placement != dst_placement
            for axis, (src_placement, dst_placement) in enumerate(zip(src, dst))
        ):
            return None

        src_tp = src[tp_axis]
        dst_tp = dst[tp_axis]
        kind = None
        tensor_dim = None
        reduce_op = "sum"
        if isinstance(src_tp, Shard) and isinstance(dst_tp, Replicate):
            kind = "all_gather"
            tensor_dim = src_tp.dim
        elif isinstance(src_tp, Partial) and isinstance(dst_tp, Replicate):
            kind = "all_reduce"
            reduce_op = src_tp.reduce_op
        elif isinstance(src_tp, Partial) and isinstance(dst_tp, Shard):
            kind = "reduce_scatter"
            tensor_dim = dst_tp.dim
            reduce_op = src_tp.reduce_op
        if kind is None or reduce_op != "sum":
            return None

        if kind == "reduce_scatter" and "gloo" in self.backend:
            kind = "all_reduce_shard"
        return TPExecutionOp(
            kind=kind,
            group=self.group,
            group_size=self.group_size,
            group_rank=self.group_rank,
            tensor_dim=tensor_dim,
            reduce_op=reduce_op,
        )


def create_tp_collective_lowerer(
    mesh: Any,
    mesh_dim_names: Sequence[str],
    *,
    collective_backend: Optional[str] = None,
) -> Optional[TPCollectiveLowerer]:
    """Resolve immutable TP runtime metadata and create the boundary lowerer."""
    mesh_dim_names = tuple(mesh_dim_names)
    if "tp" not in mesh_dim_names:
        return None

    tp_mesh = mesh if mesh_dim_names == ("tp",) else mesh["tp"]
    group = tp_mesh.get_group()
    mesh_ranks = tuple(tp_mesh.rank_list)
    group_ranks = tuple(platform.get_process_group_ranks(group))
    if mesh_ranks != group_ranks:
        logger.warning(
            "TP mesh rank order %s differs from process group rank order %s; "
            "falling back to DTensor redistribution.",
            mesh_ranks,
            group_ranks,
        )
        return None

    backend = collective_backend or platform.get_backend(group)
    return TPCollectiveLowerer(
        mesh_dim_names=mesh_dim_names,
        group=group,
        group_size=tp_mesh.size(),
        group_rank=tp_mesh.get_local_rank(),
        backend=str(backend).lower(),
    )


__all__ = [
    "TPCollectiveLowerer",
    "TPExecutionOp",
    "create_tp_collective_lowerer",
]
