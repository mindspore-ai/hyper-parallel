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
"""vLLM worker topology and HP parameter ownership."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.distributed._builder.applier import build_expert_mesh
from hyper_parallel.distributed.mesh import MeshContext
from rl.unified_execution.contract import parse_unified_payload
from rl.unified_execution.execution_plan import has_unified_ep



@dataclass(frozen=True)
class InferenceParallelContext:
    """HP tensor-parallel mesh owned by this vLLM worker."""

    mesh_context: Any = None
    tp_size: int = 1
    tp_rank: int = 0
    sequence_parallel: bool = False
    ep_size: int = 1
    ep_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    pcp_size: int = 1
    pcp_rank: int = 0


def _worker_topology(vllm_config: Any, payload: dict) -> tuple[int, bool, int, int]:
    """Check the vLLM worker topology before creating HP meshes."""
    topology = payload.get("parallelism", {})
    parallel = vllm_config.parallel_config
    if getattr(parallel, "enable_expert_parallel", False) or getattr(parallel, "enable_eplb", False):
        raise ValueError(
            "Unified MoE does not use vLLM engine flags for expert ownership. Configure expert_parallel_size."
        )
    tp_size = int(topology.get("tp", parallel.tensor_parallel_size))
    cp_size = int(topology.get("cp", 1))
    pp_size = int(topology.get("pp", parallel.pipeline_parallel_size))
    sequence_parallel = bool(topology.get("sequence_parallel", False))
    ep_size = payload.get("expert_parallelism", {}).get("inference", 1)
    dp_size = int(getattr(parallel, "data_parallel_size", 1))
    expert_domain = dp_size * cp_size * tp_size
    if ep_size < 1 or expert_domain % ep_size:
        raise ValueError("HP expert parallel size must divide the initialized DP x PCP x TP domain")
    if ep_size > 1 and not has_unified_ep(payload.get("plan_overrides", [])):
        raise ValueError("Unified HP EP requires the shared EP recipe in both eager and graph mode")
    if ep_size == 1 and cp_size > 1 and vllm_config.model_config.is_moe:
        raise ValueError("Unified MoE PCP requires graph-capable HP EP > 1")
    if cp_size not in (1, 2, 4) or pp_size != 1:
        raise ValueError("Unified inference requires PCP1/PCP2/PCP4 and PP=1")
    if (getattr(parallel, "decode_context_parallel_size", 1) != 1
            or getattr(parallel, "prefill_context_parallel_size", 1) != cp_size):
        raise ValueError("Unified inference requires matching PCP and DCP=1")
    if min(tp_size, cp_size, pp_size) < 1:
        raise ValueError("Unified parallel sizes must be positive")
    if parallel.tensor_parallel_size != tp_size or parallel.pipeline_parallel_size != 1:
        raise ValueError("vLLM worker topology must match HP TP with PP=1")
    return tp_size, sequence_parallel, ep_size, dp_size


def _native_pcp_rank(pcp_size: int) -> int:
    """Read the existing vLLM PCP rank after checking the configured group width."""
    if pcp_size == 1:
        return 0
    # vLLM remains optional in training and configuration processes.
    from vllm.distributed import get_pcp_group  # pylint: disable=C0415

    pcp_group = get_pcp_group()
    if pcp_group.world_size != pcp_size:
        raise ValueError("Initialized vLLM PCP group differs from the unified topology")
    return pcp_group.rank_in_group


def build_parallel_context(vllm_config: Any, payload: dict[str, Any]) -> InferenceParallelContext:
    """Give HP the existing worker TP group; HP owns sharding and redistribution."""
    tp_size, sequence_parallel, ep_size, dp_size = _worker_topology(vllm_config, payload)
    parallel = vllm_config.parallel_config
    pcp_size = int(getattr(parallel, "prefill_context_parallel_size", 1))
    if not dist.is_initialized():
        if tp_size != 1 or ep_size != 1 or pcp_size != 1:
            raise ValueError("Parallel inference requires vLLM to initialize distributed workers first")
        return InferenceParallelContext(sequence_parallel=sequence_parallel)
    # vLLM is an optional dependency in training and configuration processes.
    from vllm.distributed import get_tp_group  # pylint: disable=C0415

    tp_group = get_tp_group()
    pcp_rank = _native_pcp_rank(pcp_size)
    if len(tp_group.ranks) != tp_size:
        raise ValueError("Initialized vLLM process groups differ from the unified topology")
    device = vllm_config.device_config.device
    device_type = torch.device(device).type
    expert_mesh = None
    compute_dp_size = dp_size * pcp_size if ep_size > 1 else dp_size
    full_mesh = None
    if pcp_size > 1 and ep_size > 1:
        from vllm.distributed import get_ep_group  # pylint: disable=C0415

        domain = get_ep_group().ranks
        if len(domain) != dp_size * pcp_size * tp_size:
            raise ValueError("vLLM MoE communication domain differs from DP x PCP x TP")
        # Fold PCP into HP's construction domain without activating training CP.
        # Create all groups globally before registering rank-local native TP groups.
        full_mesh = init_device_mesh(device_type, (compute_dp_size, tp_size),
                                     mesh_dim_names=("dp", "tp"), rank_list=tuple(domain))
        expert_mesh = build_expert_mesh(full_mesh, ep_size)
    mesh = DeviceMesh.from_group(tp_group.device_group, device_type, mesh_dim_names=("tp",))
    tp_rank = mesh["tp"].get_local_rank()
    dp_rank = int(getattr(parallel, "data_parallel_rank", 0))
    if ep_size > 1:
        # vLLM synchronizes MoE DP schedulers (including dummy forwards). Reuse
        # those ranks and use the HP mesh to define published expert shards.
        from vllm.distributed import get_dp_group, get_ep_group  # pylint: disable=C0415

        if not vllm_config.model_config.is_moe:
            raise ValueError("HP EP requires vLLM's synchronized MoE DP scheduler")
        domain = get_ep_group().ranks
        if len(domain) != dp_size * pcp_size * tp_size:
            raise ValueError("vLLM MoE communication domain differs from DP x PCP x TP")
        # Native DP/PCP scheduling stays separate from HP's merged construction axis.
        rank_grid = [domain[index * tp_size:(index + 1) * tp_size] for index in range(compute_dp_size)]
        dp_group = full_mesh["dp"].get_group() if pcp_size > 1 else get_dp_group().device_group
        mesh = DeviceMesh.from_group(
            [dp_group, tp_group.device_group], device_type,
            mesh=rank_grid, mesh_dim_names=("dp", "tp"),
        )
        dp_rank = mesh["dp"].get_local_rank()
        if rank_grid[dp_rank] != list(tp_group.ranks):
            raise ValueError("vLLM rank ordering differs from the HP DP x TP mesh")
        if expert_mesh is None:
            expert_mesh = build_expert_mesh(mesh, ep_size)
    ep_rank = expert_mesh["ep"].get_local_rank() if expert_mesh is not None else 0
    context = MeshContext(
        device_mesh=mesh, tp_size=tp_size, tp_rank=tp_rank,
        fsdp_moe_mesh=expert_mesh, ep_size=ep_size, ep_rank=ep_rank, dp_size=compute_dp_size, dp_rank=dp_rank,
        sequence_parallel=sequence_parallel, loss_parallel=False,
    )
    return InferenceParallelContext(
        mesh_context=context, tp_size=tp_size, tp_rank=tp_rank,
        sequence_parallel=sequence_parallel, ep_size=ep_size, ep_rank=ep_rank, dp_size=dp_size,
        dp_rank=dp_rank // pcp_size if ep_size > 1 else dp_rank,
        pcp_size=pcp_size, pcp_rank=pcp_rank,
    )


def payload_from_vllm_config(vllm_config: Any) -> dict[str, Any]:
    """Read the model-owned contract; obsolete platform payloads are rejected."""
    if "hyper_rl_unified_execution" in vllm_config.additional_config:
        raise ValueError("Move the unified execution contract to model_config.hf_config via hf_overrides")
    payload = getattr(vllm_config.model_config.hf_config, "hyper_rl_unified_execution", None)
    return parse_unified_payload(payload)
