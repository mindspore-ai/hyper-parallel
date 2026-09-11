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


def classify_tp_transition(
    src: Sequence[Placement],
    dst: Sequence[Placement],
    mesh_dim_names: Sequence[str],
) -> Optional[dict]:
    """Classify one placement transition for the TP collective path.

    Single source of truth shared by the runtime lowerer
    (``TPCollectiveLowerer``) and the codegen boundary-form classifier
    (``hyper_parallel.codegen.plan.boundary_forms``), so the emit-time form
    decision and the install-time lowering can never disagree about what a
    transition means.

    Args:
        src: Source placements aligned to ``mesh_dim_names``.
        dst: Target placements aligned to ``mesh_dim_names``.
        mesh_dim_names: Axis names the placements resolve against.

    Returns:
        ``{"kind": "identity"}`` when no communication is needed;
        ``{"kind": "all_gather", "tensor_dim": d}`` for Shard -> Replicate on
        the tp axis; ``{"kind": "all_reduce", "reduce_op": "sum"}`` for
        Partial -> Replicate; ``{"kind": "reduce_scatter", "tensor_dim": d,
        "reduce_op": "sum"}`` for Partial -> Shard; ``None`` when the
        transition is not lowerable on this path (a non-tp axis differs, the
        placement pair is unsupported, or the reduce op is not ``sum``).
    """
    if tuple(src) == tuple(dst):
        return {"kind": "identity"}
    if "tp" not in mesh_dim_names:
        return None
    tp_axis = tuple(mesh_dim_names).index("tp")
    if any(
        axis != tp_axis and src_placement != dst_placement
        for axis, (src_placement, dst_placement) in enumerate(zip(src, dst))
    ):
        return None

    src_tp = src[tp_axis]
    dst_tp = dst[tp_axis]
    if isinstance(src_tp, Shard) and isinstance(dst_tp, Replicate):
        return {"kind": "all_gather", "tensor_dim": src_tp.dim}
    if isinstance(src_tp, Partial) and isinstance(dst_tp, Replicate):
        if src_tp.reduce_op != "sum":
            return None
        return {"kind": "all_reduce", "reduce_op": "sum"}
    if isinstance(src_tp, Partial) and isinstance(dst_tp, Shard):
        if src_tp.reduce_op != "sum":
            return None
        return {
            "kind": "reduce_scatter",
            "tensor_dim": dst_tp.dim,
            "reduce_op": "sum",
        }
    return None


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
        result = classify_tp_transition(src, dst, self.mesh_dim_names)
        if result is None or result["kind"] == "identity":
            return None
        return self.execution_op(
            result["kind"],
            tensor_dim=result.get("tensor_dim"),
            reduce_op=result.get("reduce_op", "sum"),
        )

    def execution_op(
        self,
        kind: str,
        tensor_dim: Optional[int] = None,
        reduce_op: str = "sum",
    ) -> TPExecutionOp:
        """Build an execution op for one bare-operator call by kind.

        The codegen bare-operator runtime (``runtime.TPOperators``) calls this
        with the kind the generated forward named, so a statically emitted
        ``reduce_scatter`` runs the exact same gloo remap (all_reduce + local
        chunk) the placement-driven ``__call__`` path applies.
        """
        # Gloo has no reduce_scatter: lower to all_reduce + local chunk, the
        # same remap the bare-operator runtime (``TPOperators``) applies.
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
    "classify_tp_transition",
    "create_tp_collective_lowerer",
]
