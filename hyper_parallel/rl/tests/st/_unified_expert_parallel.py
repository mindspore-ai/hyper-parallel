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
"""Device acceptance of production EP transports, graph replay and gradients."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from rl.unified_execution.core.execution_plan import UNIFIED_EP_TARGET
from rl.unified_execution.ops.moe.training_ep import training_ep
from rl.unified_execution.ops.moe.combine import ordered_combine
from rl.unified_execution.ops.moe.allgather_ep import allgather_ep
from rl.unified_execution.ops.moe.mc2_ep import mc2_ep
from rl.unified_execution.runtimes.vllm.parallel import build_expert_runtime
from rl.unified_execution.runtimes.vllm.moe import InferenceExpertExecutor, execute_inference_ep
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.distributed.expert_parallel.experts import bind_local_expert_forward


def _experts(count: int, offset: int) -> torch.nn.Module:
    source = torch.nn.Module()
    source.num_experts, source.hidden_size, source.intermediate_size = count, 2048, 768
    source.is_transposed = False
    source.config = SimpleNamespace(hidden_act="silu", initializer_range=.02)
    gate, down = [], []
    for expert in range(offset, offset + count):
        torch.manual_seed(12000 + expert)
        gate.append(torch.randn(1536, 2048, device="npu", dtype=torch.bfloat16) * .02)
        down.append(torch.randn(2048, 768, device="npu", dtype=torch.bfloat16) * .02)
    source.gate_up_proj = torch.nn.Parameter(torch.stack(gate))
    source.down_proj = torch.nn.Parameter(torch.stack(down))
    module = torch.nn.Module()
    module.experts = GroupedExperts(module=source)
    # The fixture already owns a shard; HP's binding reads the local tensor shape.
    bind_local_expert_forward(module, 1, use_grouped_gemm=True)
    return module



def _reference_forward(full: torch.nn.Module, states: torch.Tensor, ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    """Evaluate each local token against a complete expert bank without EP routing."""
    slots = ids.shape[1]
    routes = full.experts(states.repeat_interleave(slots, 0), ids.flatten())
    return ordered_combine(routes.view(states.shape[0], slots, states.shape[1]), weights)


def _update_weights(module: torch.nn.Module, full: torch.nn.Module) -> None:
    """Apply the same pointwise update to the local shard and independent full bank."""
    for model in (module, full):
        for parameter in model.parameters():
            parameter.add_(torch.full_like(parameter, .0002))


def _equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.isfinite(actual).all() or not torch.equal(actual.contiguous().view(torch.uint8),
                                                         expected.contiguous().view(torch.uint8)):
        raise AssertionError(f"EP bitwise gate failed: max_abs={(actual.float() - expected.float()).abs().max()}")


def _gradients(module: torch.nn.Module, states: torch.Tensor, ids: torch.Tensor,
               weights: torch.Tensor, options: dict) -> list[dict]:
    size, rank, local = options["world_size"], options["rank"], options["local_experts"]

    def gather(value: torch.Tensor) -> torch.Tensor:
        output = torch.empty((size * value.shape[0], *value.shape[1:]), device=value.device, dtype=value.dtype)
        dist.all_gather_into_tensor(output, value.detach().contiguous(), group=options["group"])
        return output

    x = states.clone().requires_grad_()
    probs = weights.clone().requires_grad_()
    upstream = torch.randn_like(states)
    training_ep(module, x, ids, probs, **options).float().mul(upstream).sum().backward()
    actual = [x.grad, probs.grad, module.experts.gate_up_proj.grad, module.experts.down_proj.grad]
    full = _experts(128, 0)
    with torch.no_grad():
        full.experts.gate_up_proj.copy_(gather(module.experts.gate_up_proj))
        full.experts.down_proj.copy_(gather(module.experts.down_proj))
    all_x = gather(states).requires_grad_()
    all_probs = gather(weights).requires_grad_()
    all_ids, all_upstream = gather(ids), gather(upstream)
    routes = full.experts(all_x.repeat_interleave(8, 0), all_ids.flatten()).view(size * states.shape[0], 8, 2048)
    ordered_combine(routes, all_probs).float().mul(all_upstream).sum().backward()
    rows = states.shape[0]
    expected = [all_x.grad[rank * rows:(rank + 1) * rows], all_probs.grad[rank * rows:(rank + 1) * rows],
                full.experts.gate_up_proj.grad[rank * local:(rank + 1) * local],
                full.experts.down_proj.grad[rank * local:(rank + 1) * local]]
    report = []
    for name, value, target in zip(("input", "router_weights", "gate_up_proj", "down_proj"), actual, expected):
        torch.testing.assert_close(value, target, rtol=.02, atol=2e-4)
        report.append({"name": name, "elements": value.numel(),
                       "max_abs": float((value.float() - target.float()).abs().max()),
                       "relative_l2": float((value.float() - target.float()).norm()
                                            / target.float().norm().clamp_min(1e-12))})
    return report



def _capacity_switch(module: torch.nn.Module, full: torch.nn.Module, runtime: object,
                     options: dict, capacity: int) -> list[dict]:
    """Alternate real transports and replay large buckets with uneven/empty peers."""
    dispatch = InferenceExpertExecutor(module.experts, options, runtime.backend, runtime.comm_name,
                              capacity, runtime.max_token_rows, 1, options["world_size"])
    context = SimpleNamespace(dp_metadata=None, batch_descriptor=None, no_compile_layers={"fixture": dispatch})
    compiled = torch.compile(execute_inference_ep, backend="eager", fullgraph=True, dynamic=True)
    records = []
    with patch("vllm.forward_context._forward_context", context), torch.no_grad():
        for scheduled in (capacity, capacity + 1, 300, 512, capacity, 0):
            rows = scheduled if options["rank"] == 0 else (0 if scheduled in (0, 300) else 3)
            counts = [scheduled] + [0 if scheduled in (0, 300) else 3] * (options["world_size"] - 1)
            context.dp_metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(counts))
            states = torch.randn(rows, 2048, device="npu", dtype=torch.bfloat16)
            ids = torch.zeros(rows, 8, device="npu", dtype=torch.int64)
            weights = torch.zeros(rows, 8, device="npu", dtype=torch.float32)

            def candidate():
                return compiled(states, ids, weights, "fixture")

            def prepare(step):
                logits = torch.randn(rows, 128, device="npu", dtype=torch.float32)
                if step % 2:
                    logits[:, :8] += 20
                probs, selected = logits.softmax(-1).topk(8, dim=-1)
                ids.copy_(selected)
                weights.copy_(probs / probs.sum(-1, keepdim=True))
                states.normal_()

            prepare(0)
            for _ in range(3):
                candidate()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = candidate()
            for step in range(3):
                prepare(step)
                expected = _reference_forward(full, states, ids, weights)
                _equal(candidate(), expected)
                graph.replay()
                torch.npu.synchronize()
                _equal(captured, expected)
            _update_weights(module, full)
            graph.replay()
            torch.npu.synchronize()
            _equal(captured, candidate())
            records.append({"scheduled": scheduled, "rows": rows, "backend": dispatch.select_backend_and_bucket(rows, context)[0],
                            "exact": True, "graph_replays": 3, "weight_update": True})
            del graph, captured
    return records


def main() -> None:
    """Test real ownership, changing routes, empty ranks, graph updates and all gradients."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--capacity", type=int, default=32)
    parser.add_argument("--backend", choices=("mc2", "allgather"), required=True)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--capacity-switch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    torch.npu.set_deterministic_level(3)
    domain = list(range(dist.get_world_size()))
    group = None
    for first in range(0, len(domain), args.ep):
        ranks = domain[first:first + args.ep]
        created_group = dist.new_group(ranks, backend="hccl")
        if dist.get_rank() in ranks:
            group = created_group
    rank, local = group.rank(), 128 // args.ep
    config = SimpleNamespace(device_config=SimpleNamespace(device="npu"), model_config=SimpleNamespace(
        dtype=torch.bfloat16, hf_config=SimpleNamespace(hidden_size=2048, moe_intermediate_size=768,
                                                       num_experts=128, num_experts_per_tok=8)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max(512, args.capacity * 2)),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[args.capacity],
                                            max_cudagraph_capture_size=args.capacity),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1))
    payload = {"plan_overrides": [{"match": "*.mlp.forward", "when": "ep", "replace": {"kind": "method",
               "implementation": {"_target_": UNIFIED_EP_TARGET, "mc2_capacity": args.capacity,
                                  "inference_backend": args.backend}}}]}
    runtime = build_expert_runtime(config, payload, group, domain)
    module = _experts(local, rank * local)
    full = _experts(128, 0)
    # Different subgroups must carry different data to detect cross-group mixing.
    torch.manual_seed(21000 + dist.get_rank())
    options = {"group": group, "rank": rank, "world_size": args.ep, "local_experts": local}
    report = {"rank": dist.get_rank(), "backend": runtime.backend, "checks": [], "passed": False}
    args.output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        (args.output / f"rank-{dist.get_rank()}.json").write_text(json.dumps(report, indent=2))

    for scenario in range(4):
        rows = (args.capacity, args.capacity // 2, 1, 0)[(rank + scenario) % 4] if scenario < 3 else 0
        x = torch.randn(rows, 2048, device="npu", dtype=torch.bfloat16)
        ids = torch.zeros(rows, 8, device="npu", dtype=torch.int64)
        weights = torch.zeros(rows, 8, device="npu", dtype=torch.float32)

        def prepare(step: int) -> None:
            logits = torch.randn(rows, 128, device="npu", dtype=torch.float32)
            if step % 2:
                logits[:, :8] += 20
            probs, selected = logits.softmax(-1).topk(8, dim=-1)
            probs /= probs.sum(-1, keepdim=True)
            ids.copy_(selected)
            weights.copy_(probs)
            x.normal_()

        def candidate() -> torch.Tensor:
            if runtime.backend == "mc2":
                return mc2_ep(x, ids, weights, experts=module.experts, comm_name=runtime.comm_name,
                              rank=rank, world_size=args.ep, local_experts=local, mc2_capacity=args.capacity)
            return allgather_ep(x, ids, weights, experts=module.experts, bucket_rows=args.capacity, **options)

        with torch.no_grad():
            prepare(0)
            for _ in range(3):
                candidate()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = candidate()
            for step in range(6):
                prepare(step)
                expected = _reference_forward(full, x, ids, weights)
                actual = candidate()
                dynamic = training_ep(module, x, ids, weights, **options)
                graph.replay()
                torch.npu.synchronize()
                _equal(actual, expected)
                _equal(dynamic, expected)
                _equal(captured, expected)
                report["checks"].append({"scenario": scenario, "step": step, "rows": rows, "exact": True})
                save()
            _update_weights(module, full)
            graph.replay()
            torch.npu.synchronize()
            _equal(captured, candidate())
        del graph, captured
    if args.capacity_switch:
        report["capacity_switch"] = _capacity_switch(module, full, runtime, options, args.capacity)
        save()
    # Training is deliberately larger than the inference graph bucket.
    with torch.no_grad():
        states = torch.randn(args.capacity + 1, 2048, device="npu", dtype=torch.bfloat16)
        probs, ids = torch.randn(args.capacity + 1, 128, device="npu").softmax(-1).topk(8, dim=-1)
        probs /= probs.sum(-1, keepdim=True)
        actual = training_ep(module, states, ids, probs, **options)
        expected = _reference_forward(full, states, ids, probs)
        _equal(actual, expected)
        report["training_above_inference_capacity"] = True
    if args.gradients:
        states = torch.randn(4, 2048, device="npu", dtype=torch.bfloat16)
        probs, ids = torch.randn(4, 128, device="npu").softmax(-1).topk(8, dim=-1)
        probs /= probs.sum(-1, keepdim=True)
        report["gradients"] = _gradients(module, states, ids, probs, options)
    report["passed"] = True
    save()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
