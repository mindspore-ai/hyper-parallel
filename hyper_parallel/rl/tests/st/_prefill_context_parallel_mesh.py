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
"""Real HCCL validation of HP expert groups across the native DP x PCP x TP domain."""

import argparse
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch_npu
import yaml

from rl.unified_execution.runtimes.vllm.parallel import build_parallel_context
from rl.unified_execution.runtimes.vllm.weights import LogicalWeightLoader
from hyper_parallel.core.dtensor.dtensor import distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def group_for(groups: list[list[int]]) -> SimpleNamespace:
    """Create every fixture group in the same order on every participating process."""
    selected = None
    for ranks in groups:
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            selected = SimpleNamespace(ranks=ranks, device_group=group, world_size=len(ranks),
                                       rank_in_group=ranks.index(dist.get_rank()))
    return selected


def check_expert_weights(context, device: str) -> dict:
    """Check every owned expert element after checkpoint loading and two publications."""
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_moe", num_experts=8, hidden_size=4,
                                   moe_intermediate_size=2, tie_word_embeddings=False)
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    expert_mesh = context.mesh_context.fsdp_moe_mesh
    experts = layer.mlp.experts
    for name, shape in (("gate_up_proj", (8, 4, 4)), ("down_proj", (8, 2, 4))):
        shard = distribute_tensor(torch.zeros(shape, device=device), expert_mesh, [Replicate(), Shard(0)])
        parameter = torch.nn.Parameter(shard.to_local())
        parameter._sharding_spec = shard.layout  # pylint: disable=protected-access
        experts.register_parameter(name, parameter)
    loader = LogicalWeightLoader(model)
    local_experts = 8 // context.ep_size
    first = context.ep_rank * local_experts

    def logical(expert, projection, version):
        shape = (4, 2) if projection == 2 else (2, 4)
        values = torch.arange(8, device=device, dtype=torch.float32).reshape(shape)
        return values + 100 * expert + 10 * projection + version

    for version in range(3):
        loader.begin()
        for expert in reversed(range(8)):
            for projection, name in enumerate(("gate_proj", "up_proj", "down_proj")):
                value = logical(expert, projection, version)
                loader.load([(f"model.layers.0.mlp.experts.{expert}.{name}.weight", value)])
                value.zero_()
        loader.finish()
        for local_index, expert in enumerate(range(first, first + local_experts)):
            gate_up = torch.cat((logical(expert, 0, version).T, logical(expert, 1, version).T), dim=-1)
            if (not torch.equal(experts.gate_up_proj[local_index], gate_up)
                    or not torch.equal(experts.down_proj[local_index], logical(expert, 2, version).T)):
                raise AssertionError(f"Expert weight differs: expert={expert}, version={version}")
    return {"owned_experts": list(range(first, first + local_experts)), "weight_versions_checked": 3,
            "local_gate_up_shape": list(experts.gate_up_proj.shape), "all_weight_elements_exact": True}


def check_case(dp_size: int, pcp_size: int, tp_size: int, ep_size: int) -> dict:
    """Adopt real groups in vLLM DP x PCP x TP order, then exercise HP collectives."""
    world = dist.get_world_size()
    grid = torch.arange(world).reshape(dp_size, pcp_size, tp_size)
    tp = group_for(grid.reshape(-1, tp_size).tolist())
    dp = group_for(grid.permute(1, 2, 0).reshape(-1, dp_size).tolist())
    pcp = group_for(grid.permute(0, 2, 1).reshape(-1, pcp_size).tolist())
    domain = group_for([list(range(world))])
    native = ModuleType("vllm.distributed")
    native.get_tp_group = lambda: tp
    native.get_dp_group = lambda: dp
    native.get_pcp_group = lambda: pcp
    native.get_ep_group = lambda: domain
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size, pipeline_parallel_size=1,
                                        prefill_context_parallel_size=pcp_size, decode_context_parallel_size=1,
                                        data_parallel_size=dp_size, data_parallel_rank=dp.rank_in_group),
        device_config=SimpleNamespace(device=f"npu:{int(os.environ['LOCAL_RANK'])}"),
        # This fixture owns FP32 miniature expert weights. Supply the current
        # scheduler/graph contract so startup can resolve its AllGather runtime.
        scheduler_config=SimpleNamespace(max_num_batched_tokens=512),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[32, 64], max_cudagraph_capture_size=64),
        model_config=SimpleNamespace(enforce_eager=True, is_moe=True, dtype=torch.float32,
                                     hf_config=SimpleNamespace(hidden_size=4, num_experts=8,
                                                               num_experts_per_tok=2)),
    )
    payload = {"parallelism": {"tp": tp_size, "cp": pcp_size, "pp": 1},
               "expert_parallelism": {"inference": ep_size}}
    recipe = Path(__file__).resolve().parents[2] / "examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml"
    payload["plan_overrides"] = yaml.safe_load(recipe.read_text(encoding="utf-8"))["train"]["plan_overrides"]
    with patch.dict("sys.modules", {"vllm.distributed": native}):
        context = build_parallel_context(config, payload)
    expert = context.mesh_context.fsdp_moe_mesh["ep"]
    ranks = dist.get_process_group_ranks(expert.get_group())
    rank = dist.get_rank()
    expected = list(range(rank // ep_size * ep_size, (rank // ep_size + 1) * ep_size))
    if ranks != expected:
        raise AssertionError(f"Expert domain differs: actual={ranks}, expected={expected}")
    value = torch.tensor(float(dist.get_rank() + 1), device=config.device_config.device)
    dist.all_reduce(value, group=expert.get_group())
    if float(value) != sum(rank + 1 for rank in ranks):
        raise AssertionError("Expert collective returned values from a different group")
    if context.mesh_context.cp_size != 1:
        raise AssertionError("Inference HP model must not activate the training CP wrapper")
    compute_dp = dp_size * pcp_size
    if context.dp_size != dp_size or context.mesh_context.dp_size != compute_dp:
        raise AssertionError("Native scheduling DP and HP construction domain were mixed")
    return {"dp": dp_size, "pcp": pcp_size, "tp": tp_size, "ep": ep_size,
            "rank": rank, "expert_ranks": ranks, "sum": float(value), "passed": True,
            **check_expert_weights(context, config.device_config.device)}


def main() -> None:
    """Run only with four devices explicitly assigned by the ST launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    try:
        if dist.get_world_size() != 4:
            raise ValueError("PCP mesh test requires exactly four ranks")
        cases = [(1, 2, 2, 2), (1, 2, 2, 4), (2, 2, 1, 2), (2, 2, 1, 4), (2, 1, 2, 4)]
        rows = [check_case(*case) for case in cases]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{dist.get_rank()}.json").write_text(json.dumps(rows, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
