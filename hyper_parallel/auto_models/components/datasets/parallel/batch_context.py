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
"""Distributed topology used while preparing each model micro-batch."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchParallelContext:
    """TP/CP runtime topology with reserved PP field-routing metadata."""

    tp_rank: int = 0
    tp_size: int = 1
    tp_group: Any = None
    cp_rank: int = 0
    cp_size: int = 1
    cp_group: Any = None
    pp_rank: int = 0
    pp_size: int = 1
    pp_group: Any = None
    pp_shared_data: bool = False

    def reads_data(self) -> bool:
        """Return whether this rank advances its local DataLoader iterator."""
        # ``pp_shared_data`` is retained for the future stage-aware router.
        # RuntimeBatchAdapter rejects this mode unless a router is supplied.
        reads_pipeline_data = not self.pp_shared_data or self.pp_rank == 0
        return self.tp_rank == 0 and self.cp_rank == 0 and reads_pipeline_data


def _get_submesh(device_mesh: Any, dimension: str) -> Any:
    """Return one named DeviceMesh dimension when it exists."""
    if device_mesh is None:
        return None
    dimension_names = getattr(device_mesh, "mesh_dim_names", ())
    if dimension not in dimension_names:
        return None
    submesh = device_mesh[dimension]
    return submesh


def create_batch_parallel_context(
        mesh_context: Any,
        *,
        pp_shared_data: bool = False,
) -> BatchParallelContext:
    """Create runtime batch topology from the Trainer mesh.

    Args:
        mesh_context: Trainer mesh state containing parallel sizes and ranks.
        pp_shared_data: Whether only PP rank zero owns the source iterator.

    Returns:
        Runtime batch ownership and process-group context.
    """
    device_mesh = getattr(mesh_context, "device_mesh", None)
    tp_mesh = _get_submesh(device_mesh, "tp")
    cp_mesh = _get_submesh(device_mesh, "cp")
    pp_mesh = _get_submesh(device_mesh, "pp")
    batch_context = BatchParallelContext(
        tp_rank=int(getattr(mesh_context, "tp_rank", 0)),
        tp_size=int(getattr(mesh_context, "tp_size", 1)),
        tp_group=tp_mesh.get_group() if tp_mesh is not None else None,
        cp_rank=int(getattr(mesh_context, "cp_rank", 0)),
        cp_size=int(getattr(mesh_context, "cp_size", 1)),
        cp_group=cp_mesh.get_group() if cp_mesh is not None else None,
        pp_rank=int(getattr(mesh_context, "pp_rank", 0)),
        pp_size=int(getattr(mesh_context, "pp_size", 1)),
        pp_group=pp_mesh.get_group() if pp_mesh is not None else None,
        pp_shared_data=pp_shared_data,
    )
    return batch_context


__all__ = ["BatchParallelContext", "create_batch_parallel_context"]
