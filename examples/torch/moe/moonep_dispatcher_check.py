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
"""Run the MoonEP EP dispatcher against the all-to-all one, and report capacity.

Two things are printed, because together they decide whether this backend is
usable for a shape:

* ``capacity`` - the router's real per-destination row counts against MoonEP's
  ``NvS = S*K`` budget.  MoonEP's planner rebalances as soon as a destination
  exceeds it, and a rebalanced plan needs redundant experts this backend does
  not own, so the dispatcher raises instead of returning wrong results.
* ``numerics`` - forward output and input gradient against the standalone
  (all-experts-local) reference, i.e. the same acceptance the EP precision ST
  applies, through the same ``MoE`` module the library ships.

Run inside the ``hyper-parallel-env`` container from the repository root::

    export ASCEND_MOONEP_HEAP=shmem HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
    torchrun --nproc_per_node=4 examples/torch/moe/moonep_dispatcher_check.py
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.components.modules.moe import MoE
from hyper_parallel.core.expert_parallel.expert_parallel import ExpertParallel


def parse_args() -> argparse.Namespace:
    """Parse the check's shape arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--moe-hidden", type=int, default=128)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=16, help="tokens per rank")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_moe(dim: int, hidden: int, experts: int, top_k: int, seed: int) -> MoE:
    """Build a seeded MoE whose experts are still local (all of them)."""
    torch.manual_seed(seed)
    return MoE(dim=dim, hidden_dim=hidden, num_experts=experts, top_k=top_k,
               score_before_experts=True)


def run_forward_backward(model: MoE, x: torch.Tensor):
    """Return ``(output, input_grad)`` for one seeded forward/backward."""
    inputs = x.clone().detach().requires_grad_(True)
    out = model(inputs)
    loss = out.float().pow(2).mean()
    with SkipDTensorDispatch():
        loss.backward()
    return out.detach(), inputs.grad.clone()


def report_capacity(model: MoE, tokens: int, group) -> None:
    """Print each rank's routed rows against MoonEP's per-rank budget."""
    device = next(model.parameters()).device
    x = torch.randn(tokens, model.experts.w1.shape[-1], device=device)
    _scores, _selected, counts = model.router(x, None)
    world = dist.get_world_size(group)
    gathered = [torch.empty_like(counts, dtype=torch.int32) for _ in range(world)]
    dist.all_gather(gathered, counts.to(torch.int32), group=group)
    counts_all = torch.stack(gathered).long()
    rows = int(counts.sum())
    per_destination = counts_all.sum(dim=0).reshape(world, -1).sum(dim=1)
    budget = rows  # NvS = S * K with one route per row
    top = int(per_destination.max())
    verdict = "fits" if top <= budget else "OVERFLOWS -> planner would rebalance"
    print(f"[capacity] rows/rank={rows} budget(NvS)={budget} "
          f"per-destination={per_destination.tolist()} max={top} -> {verdict}",
          flush=True)


def main() -> None:
    """Compare the MoonEP EP dispatch path against the standalone reference."""
    args = parse_args()
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    if args.experts % world:
        raise SystemExit(f"experts ({args.experts}) must divide world ({world})")

    reference = build_moe(args.dim, args.moe_hidden, args.experts, args.top_k,
                          args.seed).to(device)
    report_capacity(reference, args.tokens, dist.group.WORLD)

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world,), mesh_dim_names=("ep",))
    dispatcher = os.environ.get("HP_EP_TOKEN_DISPATCHER", "all_to_all").strip()
    ep_model = build_moe(args.dim, args.moe_hidden, args.experts, args.top_k,
                         args.seed).to(device)
    ep_model.load_state_dict(reference.state_dict())
    ExpertParallel(token_dispatcher=dispatcher).apply(ep_model.experts, mesh)

    torch.manual_seed(args.seed + 1)
    x = torch.randn(args.tokens, args.dim, device=device)
    ref_out, ref_grad = run_forward_backward(reference, x)
    ep_out, ep_grad = run_forward_backward(ep_model, x)

    out_diff = float((ep_out - ref_out).abs().max())
    grad_diff = float((ep_grad - ref_grad).abs().max())
    close = torch.allclose(ep_out, ref_out, rtol=1e-3, atol=1e-3)
    grad_close = torch.allclose(ep_grad, ref_grad, rtol=1e-3, atol=1e-3)
    print(f"[numerics] rank={rank} dispatcher={dispatcher} out_diff={out_diff:.3e} "
          f"close={close} grad_diff={grad_diff:.3e} close={grad_close}", flush=True)

    dist.destroy_process_group()
    if not (close and grad_close):
        raise SystemExit(f"[rank {rank}] numerics mismatch")


if __name__ == "__main__":
    main()
