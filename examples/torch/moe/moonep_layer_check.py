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
"""Run one MoE layer's expert computation through a rebalanced MoonEP plan.

This is the numerical acceptance for step ③ of
``docs/moonep_ep_integration.md``: everything except the weight *prefetch* is
exercised end to end, on the rebalanced layout a real router produces.

Per rank, with ``K = 1`` rows and a plan built by
``hyper_parallel.distributed.expert_parallel.moonep_plan``:

1. dispatch the rows and their route weights to their experts' owners;
2. walk the receive buffer segment by segment using ``cu_seqlens`` plus
   ``experts_to_copy``, run each segment's expert on those rows and scale them
   by their route weights - local experts use this rank's weights, redundant
   segments use the owner's (gathered here; MoonEP's ``prefetch_weight``
   replaces that gather in step ④);
3. combine back and compare against the standalone reference
   ``y[t] = sum_k w[t,k] * expert_e(x[t])``.

Run inside the ``hyper-parallel-env`` container from the repository root::

    export ASCEND_MOONEP_HEAP=shmem HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
    torchrun --nproc_per_node=4 examples/torch/moe/moonep_layer_check.py \
        --rows 48 --experts 8 --hidden 32 --sink-experts 2
"""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401  # pylint: disable=unused-import  (registers the npu device)
from moonep import Buffer, MoonEPCommPlan
from npu_patch import ShmemRuntime
from npu_patch._device import local_device_index

from hyper_parallel.distributed.expert_parallel.moonep_plan import build_moonep_plan


def parse_args() -> argparse.Namespace:
    """Parse the layer test's shape arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=48, help="routed rows per rank")
    parser.add_argument("--experts", type=int, default=8, help="total experts E")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--sink-experts", type=int, default=2,
                        help="route every rank's rows onto this many leading experts "
                             "(forces MoonEP to rebalance)")
    parser.add_argument("--num-sms", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Compare the MoonEP-pathed layer output against the standalone reference."""
    args = parse_args()
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    device_id = local_device_index(rank)
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    rows, experts, hidden = args.rows, args.experts, args.hidden
    if experts % world:
        raise SystemExit(f"experts ({experts}) must divide world ({world})")
    epn = experts // world
    if not 0 < args.sink_experts <= epn:
        raise SystemExit(f"--sink-experts must be in (0, {epn}]")

    buffer = Buffer(rows, hidden, 1, experts, world, B=epn, num_sms=args.num_sms,
                    use_udma=False, group=group, explicitly_destroy=True)
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=group, enable_udma=False)
    buffer._require_ctx()  # pylint: disable=protected-access

    # One SwiGLU-ish expert per expert id: y = silu(x @ w1) @ w2.
    torch.manual_seed(7)
    w1 = torch.randn(experts, hidden, hidden, dtype=torch.float32) / (hidden ** 0.5)
    w2 = torch.randn(experts, hidden, hidden, dtype=torch.float32) / (hidden ** 0.5)
    w1, w2 = w1.to(device), w2.to(device)
    # Each rank owns `epn` experts; gathering those chunks in rank order rebuilds
    # the global expert table (step ④ replaces this with MoonEP's prefetch).
    mine_w1 = w1[rank * epn:(rank + 1) * epn].contiguous()
    mine_w2 = w2[rank * epn:(rank + 1) * epn].contiguous()
    all_w1 = [torch.empty_like(mine_w1) for _ in range(world)]
    all_w2 = [torch.empty_like(mine_w2) for _ in range(world)]
    dist.all_gather(all_w1, mine_w1, group=group)
    dist.all_gather(all_w2, mine_w2, group=group)
    all_w1 = torch.stack(all_w1).reshape(experts, hidden, hidden)
    all_w2 = torch.stack(all_w2).reshape(experts, hidden, hidden)

    # Routing: every rank piles its rows onto the leading experts (rank 0's),
    # which is far past NvS, so the plan MUST rebalance.
    span = rows // args.sink_experts
    counts = torch.zeros(experts, dtype=torch.int32)
    counts[:args.sink_experts] = span
    counts[args.sink_experts - 1] += rows - span * args.sink_experts
    ids = torch.repeat_interleave(
        torch.arange(args.sink_experts, dtype=torch.int32), counts[:args.sink_experts])
    counts = counts.to(device)
    counts_all = [torch.empty_like(counts) for _ in range(world)]
    dist.all_gather(counts_all, counts, group=group)
    plan_fields = build_moonep_plan(torch.stack(counts_all), counts, ids.to(device),
                                    rank, epn, rows)

    torch.manual_seed(11)
    x = torch.randn(rows, hidden, dtype=torch.float32, device=device)
    weights = torch.rand(rows, dtype=torch.float32, device=device)
    plan = MoonEPCommPlan(
        dst=plan_fields["dst"], experts_to_copy=plan_fields["experts_to_copy"],
        zero_fill_ranges=plan_fields["zero_fill_ranges"],
        remote_stats=plan_fields["remote_stats"], N=rows, R=world, E=experts,
        B=epn, NvS=rows, K=1,
        dup_groups=torch.zeros((rows, 3), dtype=torch.int32, device=device),
        dup_loffs=torch.zeros((rows,), dtype=torch.int32, device=device),
        dup_counts=torch.zeros((2,), dtype=torch.int32, device=device),
    )

    received, weights_local, _cu, plan = buffer.dispatch(
        x.to(torch.bfloat16), route_weights_sk=weights.unsqueeze(1), plan=plan,
        zero_copy=True)
    cu = plan_fields["cu_seqlens"].cpu()
    prefetch = plan_fields["experts_to_copy"][rank].cpu()

    # Segment walk: groups < E are that global expert's rows, groups >= E are the
    # prefetch slots named by experts_to_copy. Redundant segments use the owner's
    # weights (here gathered; step ④ replaces this with prefetch_weight).
    out_local = torch.zeros(rows, hidden, dtype=torch.float32, device=device)
    start = 0
    layout = []
    for group_id in range(experts + epn):
        end = int(cu[group_id])
        if end <= start:
            continue
        expert = group_id if group_id < experts else int(prefetch[group_id - experts])
        if expert < 0:
            raise SystemExit(f"[rank {rank}] group {group_id} holds rows but has no expert")
        seg = received[start:end].float()
        seg = F.silu(seg @ all_w1[expert]) @ all_w2[expert]
        out_local[start:end] = seg * weights_local[start:end].float().unsqueeze(1)
        layout.append((expert, end - start))
        start = end

    combined, _, _ = buffer.combine(plan=plan, hidden_nvsh=out_local.to(torch.bfloat16))

    # Standalone reference over the same routing.
    # The reference sees the same bf16 payload the transport moved.
    x_bf16 = x.to(torch.bfloat16).float()
    reference = torch.empty(rows, hidden, dtype=torch.float32, device=device)
    for row in range(rows):
        expert = int(ids[row])
        row_out = F.silu(x_bf16[row:row + 1] @ all_w1[expert]) @ all_w2[expert]
        reference[row] = row_out * weights[row]

    diff = float((combined.float() - reference).abs().max())
    close = torch.allclose(combined.float(), reference, rtol=5e-3, atol=5e-3)
    print(f"[rank {rank}] rows={rows} E={experts} epn={epn} layout={layout} "
          f"max_abs_diff={diff:.3e} close={close} -> "
          f"{'PASS' if close else 'FAIL'}", flush=True)

    ShmemRuntime.barrier(group=group)
    torch.npu.synchronize()
    ShmemRuntime.finalize()
    buffer.destroy()
    dist.destroy_process_group()
    if not close:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
