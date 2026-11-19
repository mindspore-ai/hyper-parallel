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

"""Distributed full-gradient and optimizer parity for the collective backend.

Run directly with torchrun. CPU/gloo is supported for development; --npu
selects HCCL and NPU. No external MoonEP package is used.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from hyper_parallel import DTensor, SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.distributed.expert_parallel.moonep_collective import (
    collective_expert_forward,
)
from hyper_parallel.distributed.expert_parallel.moonep_collective_dedup import (
    collective_token_forward,
)


def local(tensor):
    """Read the local shard for distributed comparisons."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class Experts(nn.Module):
    """Minimal stacked packed SwiGLU holder with module hook boundaries."""

    def __init__(self, gate, down):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate.clone())
        self.down_proj = nn.Parameter(down.clone())
        self.local_expert_count = gate.shape[0]

    def forward(self, states, indices, weights=None):
        """Execute load-balanced expert assignments."""
        if weights is None:
            return collective_expert_forward(self, states, indices, group=self.ep_group)
        return collective_token_forward(self, states, indices, weights, group=self.ep_group,
                                        diagnostics=self.diagnostics)


def reference(states, ids, weights, gate, down):
    """Independent dense token-by-token SwiGLU reference."""
    rows = []
    for token in range(states.shape[0]):
        choices = []
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            projected = F.linear(states[token], gate[expert])
            left, right = projected.chunk(2)
            expert_output = F.linear(F.silu(left) * right, down[expert])
            result_dtype = torch.promote_types(expert_output.dtype, weights.dtype)
            choices.append(expert_output.to(result_dtype) * weights[token, choice].to(result_dtype))
        rows.append(torch.stack(choices).sum(0))
    return (torch.stack(rows) if rows else states.to(torch.promote_types(states.dtype, weights.dtype)) * 0
            + (gate.sum() + down.sum() + weights.sum()) * 0)


def main(argv=None):
    """Compare outputs, complete gradients and optimizer updates across ranks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--mixed-route", action="store_true")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--fp64", action="store_true")
    parser.add_argument("--optimizer", choices=("both", "sgd", "adamw"), default="both")
    args = parser.parse_args(argv)
    if args.npu:
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("npu" if args.npu else "cpu")
    dist.init_process_group("hccl" if args.npu else "gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    if args.fsdp:
        mesh = init_device_mesh(device.type, (1, size), mesh_dim_names=("fsdp", "ep"))
    experts_per_rank, hidden, intermediate = max(2, args.top_k), 32, 64
    dtype = torch.float64 if args.fp64 else torch.bfloat16 if args.bf16 else torch.float32
    atol, rtol = (1e-10, 1e-8) if args.fp64 else (5e-3, 5e-2) if args.bf16 else (3e-6, 3e-5)
    optimizers = ((torch.optim.SGD, torch.optim.AdamW) if args.optimizer == "both" else
                  (torch.optim.SGD,) if args.optimizer == "sgd" else (torch.optim.AdamW,))
    for optimizer_type in optimizers:
        for pattern in ("uniform", "paired", "owner_hot", "hot", "empty_rank", "sparse", "empty_all"):
            torch.manual_seed(42)
            gate = (torch.randn(size * experts_per_rank, 2 * intermediate, hidden) * .05).to(device=device, dtype=dtype)
            down = (torch.randn(size * experts_per_rank, hidden, intermediate) * .05).to(device=device, dtype=dtype)
            lo = rank * experts_per_rank
            module = Experts(gate[lo:lo + experts_per_rank], down[lo:lo + experts_per_rank])
            module._ep_use_grouped_gemm = args.grouped
            module.diagnostics = {}
            module.ep_group = mesh["ep"].get_group("ep") if args.fsdp else dist.group.WORLD
            if args.fsdp:
                for name in ("gate_up_proj", "down_proj"):
                    parameter = getattr(module, name)
                    setattr(module, name, nn.Parameter(DTensor.from_local(parameter.detach(), mesh["ep"], (Shard(0),))))
                fully_shard(module, mesh=mesh["fsdp"], reshard_after_forward=True)
            ref_gate, ref_down = nn.Parameter(gate), nn.Parameter(down)
            optimizer = optimizer_type(module.parameters(), lr=.01)
            ref_optimizer = optimizer_type((ref_gate, ref_down), lr=.01)
            for step in range(3):
                if step == 1:
                    saved = {name: value.detach().clone() for name, value in module.state_dict().items()}
                    with SkipDTensorDispatch(), torch.no_grad():
                        for parameter in module.parameters():
                            parameter.add_(1)
                    module.load_state_dict(saved)
                optimizer.zero_grad(set_to_none=True)
                ref_optimizer.zero_grad(set_to_none=True)
                for micro in range(2):
                    for source in range(size):
                        torch.manual_seed(100 + source + 11 * micro + 31 * step)
                        tokens = 0 if pattern == "empty_all" or (pattern == "empty_rank" and source == 0) else 5
                        if pattern == "sparse":
                            tokens = 1 if source == 0 else 0
                        x = torch.randn(tokens, hidden).to(device=device, dtype=dtype).requires_grad_()
                        ids = torch.randint(size * experts_per_rank, (tokens, args.top_k), device=device)
                        if pattern == "paired":
                            ids[:] = ((source + 1) % size) * experts_per_rank + torch.arange(args.top_k, device=device) % experts_per_rank
                        if pattern == "owner_hot":
                            ids[:] = torch.arange(args.top_k, device=device)
                        if pattern in ("hot", "empty_rank"):
                            ids.zero_()
                        route = torch.rand(tokens, args.top_k, device=device, dtype=torch.float32 if args.mixed_route else dtype).requires_grad_()
                        ref_output = reference(x, ids, route, ref_gate, ref_down)
                        ref_output.square().sum().backward()
                        if source == rank:
                            local_x = x.detach().clone().requires_grad_()
                            local_route = route.detach().clone().requires_grad_()
                            local_ids = ids
                            expected = ref_output.detach()
                            expected_x_grad = x.grad.clone()
                            expected_route_grad = route.grad.clone()
                    if args.assignment:
                        expanded = local_x[:, None, :].expand(-1, args.top_k, -1).reshape(-1, hidden)
                        inputs = (expanded, local_ids.flatten())
                    else:
                        inputs = (local_x, local_ids, local_route)
                    result = checkpoint(module, *inputs, use_reentrant=False) if micro == 1 else module(*inputs)
                    output = ((result.reshape(-1, args.top_k, hidden) * local_route[..., None]).sum(1)
                              if args.assignment else result)
                    if not args.assignment and pattern == "paired":
                        counters = module.diagnostics
                        if rank == 0 and step == 0 and micro == 0:
                            print(f"DEDUP {optimizer_type.__name__} paired rank0: {counters}", flush=True)
                        if counters["unique_token_rank_rows"] != tokens:
                            raise AssertionError(f"Expected one hidden row per token/destination: {counters}")
                        if size > 1 and counters["remote_hidden_bytes_after"] * args.top_k != counters["remote_hidden_bytes_before"]:
                            raise AssertionError(f"Expected exact top-k hidden-byte reduction: {counters}")
                    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)
                    output.square().sum().backward()
                    torch.testing.assert_close(local_x.grad, expected_x_grad, atol=atol, rtol=rtol)
                    torch.testing.assert_close(local_route.grad, expected_route_grad, atol=atol, rtol=rtol)
                torch.testing.assert_close(local(module.gate_up_proj.grad), ref_gate.grad[lo:lo + experts_per_rank],
                                           atol=atol, rtol=rtol)
                torch.testing.assert_close(local(module.down_proj.grad), ref_down.grad[lo:lo + experts_per_rank],
                                           atol=atol, rtol=rtol)
                with SkipDTensorDispatch():
                    optimizer.step()
                ref_optimizer.step()
                torch.testing.assert_close(local(module.gate_up_proj), ref_gate[lo:lo + experts_per_rank], atol=atol, rtol=rtol)
                torch.testing.assert_close(local(module.down_proj), ref_down[lo:lo + experts_per_rank], atol=atol, rtol=rtol)
            if rank == 0:
                print(f"PASS {optimizer_type.__name__} {pattern}: 3 steps, accumulation=2, checkpoint", flush=True)
    forbidden = [name for name in sys.modules if name.split(".")[0] in ("moonep", "npu_patch")]
    if forbidden:
        raise AssertionError(f"Collective backend loaded external MoonEP modules: {forbidden}")
    dist.destroy_process_group()


def test_collective_fp32():
    """ST entry: EP8 with expert FSDP1, accumulation and checkpoint replay."""
    main(["--npu", "--fsdp"])


def test_collective_bf16_mixed_sgd():
    """ST entry: preserve FP32 routing weights with BF16 token payloads."""
    main(["--npu", "--bf16", "--mixed-route", "--optimizer=sgd"])


if __name__ == "__main__":
    main()
