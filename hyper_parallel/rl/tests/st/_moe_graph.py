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
"""Real HCCL numerical and replay acceptance for the shared fixed-capacity MoE."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import

from rl.unified_execution.ops.moe.fixed_capacity_ep import fixed_capacity_ep
from hyper_parallel.distributed.expert_parallel.experts import (
    _local_swiglu_expert_forward,
    ep_routed_forward,
)


class Experts(torch.nn.Module):
    """Small trainable experts using HP's existing local grouped implementation."""

    def __init__(self, count: int, hidden: int, intermediate: int) -> None:
        """Create finite synthetic weights in the original HP expert layout."""
        super().__init__()
        self.local_expert_count = count
        self.gate_up_proj = torch.nn.Parameter(torch.randn(count, 2 * intermediate, hidden,
                                                          device="npu", dtype=torch.bfloat16) * 0.03)
        self.down_proj = torch.nn.Parameter(torch.randn(count, hidden, intermediate,
                                                       device="npu", dtype=torch.bfloat16) * 0.03)
        self._ep_use_grouped_gemm = True

    def forward(self, states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Keep both old and new dispatch on precisely the same expert kernel."""
        return _local_swiglu_expert_forward(self, states, indices)


def router(module: torch.nn.Module, unused_states: torch.Tensor) -> tuple:
    """Expose this case's independently generated real dynamic routing tensors."""
    del unused_states
    return module.route_ids, module.route_weights


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Compare all elements, rejecting nonfinite values separately."""
    if actual.numel() == 0:
        return {"equal": True, "max_abs": 0.0, "finite": True, "elements": 0}
    return {"equal": torch.equal(actual, expected),
            "max_abs": (actual.float() - expected.float()).abs().max().item(),
            "finite": bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
            "elements": actual.numel()}


def run(args: argparse.Namespace) -> None:
    """Test old/new eager, changing-route graph replay, and trainable gradients."""
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    size, group = dist.get_world_size(), dist.group.WORLD
    torch.npu.set_deterministic_level(3)
    torch.manual_seed(1234 + rank)
    capacity, hidden, local_experts, slots = 16, 256, 4, 4
    module = torch.nn.Module()
    module.experts = Experts(local_experts, hidden, 128)
    states = torch.randn(capacity, hidden, device="npu", dtype=torch.bfloat16)
    ids = torch.zeros(capacity, slots, device="npu", dtype=torch.int64)
    weights = torch.zeros(capacity, slots, device="npu", dtype=torch.float32)

    def candidate(x: torch.Tensor, selected: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Exercise the production transport with trainable local experts."""
        return fixed_capacity_ep(x, selected, probs, expert_forward=module.experts, group=group, rank=rank,
                                 world_size=size, local_experts=local_experts, capacity=capacity)

    def prepare(step: int) -> int:
        """Change routes, load skew and valid rows while retaining buffer addresses."""
        torch.manual_seed(9000 + step * size + rank)
        rows = (capacity, 7, 1, 0, capacity)[(step + rank) % 5]
        states.copy_(torch.randn_like(states))
        logits = torch.randn(capacity, local_experts * size, device="npu")
        if step % 2:
            logits[:, :slots] += 20  # All routes concentrate on one expert owner.
        selected_weights, selected_ids = logits.softmax(-1).topk(slots, dim=-1)
        selected_weights /= selected_weights.sum(-1, keepdim=True)
        ids.copy_(selected_ids)
        weights.copy_(selected_weights)
        ids[rows:] = -1
        weights[rows:] = 0
        return rows

    checks = []
    with torch.no_grad():
        prepare(0)
        for _ in range(3):
            candidate(states, ids, weights)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = candidate(states, ids, weights)
        for step in range(12):
            rows = prepare(step)
            module.route_ids, module.route_weights = ids[:rows], weights[:rows]
            old = ep_routed_forward(module, states[:rows].unsqueeze(0), router_fn=router, ep_group=group)[0]
            eager = candidate(states, ids, weights)
            graph.replay()
            torch.npu.synchronize()
            checks.append({"step": step, "rows": rows, "old_new": difference(eager[:rows], old),
                           "replay": difference(captured, eager),
                           "padding_zero": bool((captured[rows:] == 0).all())})
            if not all(checks[-1][key]["equal"] and checks[-1][key]["finite"] for key in ("old_new", "replay")):
                raise RuntimeError(f"EP numerical mismatch: {checks[-1]}")
    prepare(0)
    module.route_ids = ids.clamp_min(0)
    grads = []
    for original in (True, False):
        module.zero_grad(set_to_none=True)
        x = states.detach().clone().requires_grad_()
        probs = weights.detach().clone().requires_grad_()
        module.route_weights = probs
        out = (ep_routed_forward(module, x.unsqueeze(0), router_fn=router, ep_group=group)[0]
               if original else candidate(x, module.route_ids, probs))
        out.float().square().sum().backward()
        grads.append([x.grad.detach().clone(), probs.grad.detach().clone(),
                      *[parameter.grad.detach().clone() for parameter in module.parameters()]])
    gradient_checks = [difference(new, old) for new, old in zip(grads[1], grads[0], strict=True)]
    for new, old in zip(grads[1], grads[0], strict=True):
        torch.testing.assert_close(new, old, rtol=0.02, atol=2e-4)
    if not all(row["finite"] for row in gradient_checks):
        raise RuntimeError(f"Nonfinite gradients: {gradient_checks}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"rank": rank, "ep": size, "checks": checks, "gradients": gradient_checks}
    (args.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


def test_worker() -> None:
    """Execute inside the distributed pytest subprocess only."""
    run(SimpleNamespace(output=Path(os.environ["RL_MOE_GRAPH_OUTPUT"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
