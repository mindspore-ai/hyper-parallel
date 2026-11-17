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
"""Worker topology, scheduler row bounds and dedicated MC2 communicator ownership."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, init_device_mesh
from hyper_parallel.distributed._builder.applier import build_expert_mesh
from hyper_parallel.distributed.mesh import MeshContext
from rl.unified_execution.core.contract import parse_unified_payload
from rl.unified_execution.core.execution_plan import unified_ep_options


logger = logging.getLogger(__name__)

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
    expert_runtime: Any = None


def _worker_topology(vllm_config: Any, payload: dict) -> tuple[int, bool, int, int]:
    """Check the vLLM worker topology before creating HP meshes."""
    topology = payload.get("parallelism", {})
    parallel = vllm_config.parallel_config
    if getattr(parallel, "enable_expert_parallel", False) or getattr(parallel, "enable_eplb", False):
        raise ValueError("Unified HP inference does not delegate expert parallelism or EPLB to vLLM")
    tp_size = int(topology.get("tp", parallel.tensor_parallel_size))
    cp_size = int(topology.get("cp", 1))
    pp_size = int(topology.get("pp", parallel.pipeline_parallel_size))
    sequence_parallel = bool(topology.get("sequence_parallel", False))
    ep_size = payload.get("expert_parallelism", {}).get("inference", 1)
    dp_size = int(getattr(parallel, "data_parallel_size", 1))
    expert_domain = dp_size * cp_size * tp_size
    if ep_size < 1 or expert_domain % ep_size:
        raise ValueError("HP expert parallel size must divide the initialized DP x PCP x TP domain")
    if (ep_size > 1 and not vllm_config.model_config.enforce_eager
            and unified_ep_options(payload.get("plan_overrides", [])) is None):
        raise ValueError("Unified HP EP requires enforce_eager=true or the shared graph EP recipe")
    if (cp_size > 1 and vllm_config.model_config.is_moe
            and (ep_size <= 1 or unified_ep_options(payload.get("plan_overrides", [])) is None)):
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


def build_parallel_context(vllm_config: Any, payload: dict[str, Any]) -> InferenceParallelContext:
    """Give HP the existing worker TP group; HP owns sharding and redistribution."""
    tp_size, sequence_parallel, ep_size, dp_size = _worker_topology(vllm_config, payload)
    parallel = vllm_config.parallel_config
    pcp_size = int(getattr(parallel, "prefill_context_parallel_size", 1))
    pcp_rank = 0
    if not dist.is_initialized():
        if tp_size != 1 or ep_size != 1 or pcp_size != 1:
            raise ValueError("Parallel inference requires vLLM to initialize distributed workers first")
        return InferenceParallelContext(sequence_parallel=sequence_parallel)
    # vLLM is an optional dependency in training and configuration processes.
    from vllm.distributed import get_tp_group  # pylint: disable=C0415

    tp_group = get_tp_group()
    if pcp_size > 1:
        from vllm.distributed import get_pcp_group  # pylint: disable=C0415

        pcp_group = get_pcp_group()
        if pcp_group.world_size != pcp_size:
            raise ValueError("Initialized vLLM PCP group differs from the unified topology")
        pcp_rank = pcp_group.rank_in_group
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
        # those ranks, while HP alone owns expert sharding and all-to-all.
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
    expert_runtime = None
    if expert_mesh is not None:

        expert_runtime = build_expert_runtime(vllm_config, payload, expert_mesh["ep"].get_group(), domain)
    return InferenceParallelContext(
        expert_runtime=expert_runtime, mesh_context=context, tp_size=tp_size, tp_rank=tp_rank,
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


@dataclass(frozen=True)
class InferenceExpertRuntime:
    """Own a dedicated MC2 communicator for the lifetime of the worker model."""

    backend: str
    comm_name: str = ""
    group: Any = None
    max_token_rows: int = 0
    row_alignment: int = 1
    dp_size: int = 1


def inference_row_bounds(vllm_config: Any, payload: dict) -> tuple[int, int]:
    """Bound scheduler and graph rows before HP's optional SP padding.

    PCP may replicate decode rows, so dividing the scheduler budget by PCP is
    unsafe. Keep the full budget; each execution uses its actual local bucket.
    Capture descriptors can round decode sizes up to a whole query length.
    """
    scheduler = vllm_config.scheduler_config
    compilation = vllm_config.compilation_config
    alignment = (vllm_config.parallel_config.tensor_parallel_size
                 if payload.get("parallelism", {}).get("sequence_parallel", False) else 1)
    decode_length = 1 + (getattr(vllm_config, "num_speculative_tokens", 0) or 0)
    sizes = list(compilation.cudagraph_capture_sizes or [])
    sizes.append(compilation.max_cudagraph_capture_size or 0)
    graph_limit = max(((size + decode_length - 1) // decode_length) * decode_length for size in sizes)
    limit = max(scheduler.max_num_batched_tokens, graph_limit)
    if limit < 1 or alignment < 1:
        raise ValueError("Inference EP requires positive scheduler budget and row alignment")
    return ((limit + alignment - 1) // alignment) * alignment, alignment


def _mc2_unavailable_reason(*, device_name: str, world_size: int, mc2_capacity: int,
                            config: Any, dtype: Any, available: bool, hccl_buffer_mb: int) -> str:
    """Check the A2 nonquantized fullmesh contract, independently of benchmark shapes."""
    if not available:
        return "native MC2 dispatch/combine V2 operators are unavailable"
    if "910B" not in device_name:
        return "the fullmesh adapter requires an Ascend A2 device"
    if dtype not in (torch.bfloat16, torch.float16):
        return "nonquantized MC2 requires BF16 or FP16"
    # The operator supports larger domains as well as the single-server path.
    if world_size not in (*range(2, 9), 16, 32, 64, 128, 192, 256, 384):
        return "EP size is outside the A2 fullmesh operator domain sizes"
    if not 1 <= mc2_capacity <= 256:
        return "A2 fullmesh requires a per-rank mc2_capacity between 1 and 256"
    hidden = getattr(config, "hidden_size", 0)
    experts = getattr(config, "num_experts", 0)
    top_k = getattr(config, "num_experts_per_tok", 0)
    if not 0 < hidden <= 10240 or hidden % 32:
        return "A2 fullmesh requires hidden_size <= 10240 and aligned to 32"
    if not 0 < experts <= 1024 or experts % world_size:
        return "A2 fullmesh requires at most 1024 experts evenly partitioned over EP"
    if not 1 <= top_k <= min(16, experts):
        return "A2 fullmesh requires top-k <= min(16, num_experts)"
    # Both directions need a window sized for the worst owner, not average routing.
    required_bytes = 4 * mc2_capacity * world_size * min(experts // world_size, top_k) * hidden + 4 * 1024**2
    required_mb = (required_bytes + 1024**2 - 1) // 1024**2
    if hccl_buffer_mb < required_mb:
        return f"A2 fullmesh requires HCCL_BUFFSIZE >= {required_mb} MiB, got {hccl_buffer_mb}"
    return ""


def select_ep_backend(requested: str, *, device_name: str, world_size: int,
                      mc2_capacity: int, config: Any, dtype: Any, available: bool,
                      hccl_buffer_mb: int = 200) -> str:
    """Select by operator capability and mc2_capacity, regardless of eager/graph mode.

    Auto uses AllGather when the fullmesh contract is not met. Explicit MC2
    reports the unsupported condition before communicator creation or capture.
    """
    if requested not in ("auto", "mc2", "allgather"):
        raise ValueError("Unknown unified inference EP backend")
    if requested == "allgather":
        return "allgather"
    reason = _mc2_unavailable_reason(
        device_name=device_name, world_size=world_size, mc2_capacity=mc2_capacity, config=config,
        dtype=dtype, available=available, hccl_buffer_mb=hccl_buffer_mb,
    )
    if requested == "mc2" and reason:
        raise ValueError(f"Requested MC2 is unsupported: {reason}")
    if reason:
        logger.info("Unified inference EP selects AllGather: %s", reason)
    return "allgather" if reason else "mc2"


def build_expert_runtime(vllm_config: Any, payload: dict, group: Any,
                         domain: list[int]) -> InferenceExpertRuntime | None:
    """Build disjoint MC2 groups in native rank order, once per worker.

    Ordinary HP collectives never use these dedicated communicators. Backend
    agreement is checked on the existing HP group before any MC2 group is made.
    """
    options = unified_ep_options(payload.get("plan_overrides", []))
    if options is None or group.size() == 1:
        return None
    # Optional NPU introspection runs once before constructing the inference model.
    import torch_npu  # pylint: disable=C0415

    size = group.size()
    max_rows, alignment = inference_row_bounds(vllm_config, payload)
    dp_size = vllm_config.parallel_config.data_parallel_size
    device = torch.device(vllm_config.device_config.device)
    hccl_buffer_mb = int(os.environ.get("HCCL_BUFFSIZE", "200"))
    backend = select_ep_backend(
        options["inference_backend"], device_name=torch.npu.get_device_name(), world_size=size,
        mc2_capacity=options["mc2_capacity"], config=vllm_config.model_config.hf_config,
        dtype=vllm_config.model_config.dtype, hccl_buffer_mb=hccl_buffer_mb,
        available=hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        and hasattr(torch_npu, "npu_moe_distribute_combine_v2"),
    )

    # Startup agreement prevents peers from entering different collective paths.
    # This host-visible check runs before tracing, never inside the MoE hot path.
    selected = torch.tensor([backend == "mc2", options["mc2_capacity"], max_rows, alignment, dp_size, hccl_buffer_mb],
                            device=device, dtype=torch.int32)
    choices = torch.empty(size * selected.numel(), device=device, dtype=torch.int32)
    dist.all_gather_into_tensor(choices, selected, group=group)
    if not bool((choices.view(size, -1) == selected).all().item()):
        raise ValueError("All ranks in an HP EP group must agree on backend, scheduler/graph bounds and HCCL_BUFFSIZE")
    if group.rank() == 0:
        logger.info("Unified inference EP backend=%s ranks=%s mc2_capacity=%d max_token_rows=%d alignment=%d",
                    backend, dist.get_process_group_ranks(group), options["mc2_capacity"], max_rows, alignment)
    if backend == "allgather":
        return InferenceExpertRuntime(backend, max_token_rows=max_rows, row_alignment=alignment, dp_size=dp_size)

    own_ranks = dist.get_process_group_ranks(group)
    subgroups = [list(domain[first:first + size]) for first in range(0, len(domain), size)]
    if own_ranks not in subgroups:
        raise ValueError("Dedicated MC2 group must preserve HP expert ownership and rank order")
    # HP EP subgroups are disjoint. Only their members join this local barrier;
    # another subgroup may explicitly select AllGather without creating MC2.
    mc2_group = dist.new_group(own_ranks, backend="hccl", use_local_synchronization=True)
    comm_name = mc2_group._get_backend(device).get_hccl_comm_name(group.rank())  # pylint: disable=protected-access
    return InferenceExpertRuntime(backend, comm_name, mc2_group, max_rows, alignment, dp_size)
