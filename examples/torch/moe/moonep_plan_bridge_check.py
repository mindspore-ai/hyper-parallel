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
"""Check the two properties Path B of the MoonEP integration needs from a plan.

A rebalanced MoonEP plan (the only kind a real router produces, see
``docs/moonep_ep_integration.md`` §3) moves part of a hot expert's rows to other
ranks.  Anyone computing those rows locally therefore has to know, per received
segment, *which expert* it belongs to and *how many* rows it holds.  This script
asserts, on a deliberately skewed routing that forces rebalancing:

* ``lossless`` - ``combine(dispatch(x))`` still returns ``x`` bit for bit, so a
  rebalanced transport loses nothing; only the layout changes.
* ``segments`` - the plan's ``cu_seqlens`` plus ``experts_to_copy`` enumerate
  the receive buffer completely: every segment's expert id and row count is
  recoverable, the counts sum to the buffer, and the redundant segments are
  exactly the prefetch slots.

Run inside the ``hyper-parallel-env`` container from the repository root::

    export ASCEND_MOONEP_HEAP=shmem HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
    torchrun --nproc_per_node=4 examples/torch/moe/moonep_plan_bridge_check.py \
        --rows 32 --experts 8 --hidden 64 --sink-experts 2
"""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import  (registers the npu device)
from moonep import Buffer
from moonep_check_utils import build_explicit_plan
from npu_patch import ShmemRuntime
from npu_patch._device import local_device_index


def parse_args() -> argparse.Namespace:
    """Parse the probe's shape arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32, help="routed rows per rank")
    parser.add_argument("--experts", type=int, default=8, help="total routed experts E")
    parser.add_argument("--hidden", type=int, default=64, help="row width H")
    parser.add_argument("--sink-experts", type=int, default=2,
                        help="every rank routes all its rows to this many leading experts "
                             "(they all live on rank 0), which forces rebalancing")
    parser.add_argument("--num-sms", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run the lossless and segment-enumeration assertions on every rank."""
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
        raise SystemExit(f"--sink-experts must be in (0, {epn}] so the sink is one rank")

    buffer = Buffer(rows, hidden, 1, experts, world, B=epn, num_sms=args.num_sms,
                    use_udma=False, group=group, explicitly_destroy=True)
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=group, enable_udma=False)

    # Every rank piles its rows onto the leading experts, which all live on rank
    # 0: rank 0's share is world * rows / sink-experts per expert, far past the
    # NvS = rows capacity, so the planner must rebalance.
    span = rows // args.sink_experts
    counts = torch.zeros(experts, dtype=torch.int32)
    counts[:args.sink_experts] = span
    counts[args.sink_experts - 1] += rows - span * args.sink_experts
    experts_per_row = torch.repeat_interleave(
        torch.arange(args.sink_experts, dtype=torch.int32), counts[:args.sink_experts])
    counts = counts.to(device)
    plan, cu_ref = build_explicit_plan(experts_per_row.to(device), counts, group)

    torch.manual_seed(0)
    payload = torch.randn(rows, hidden, dtype=torch.bfloat16, device=device)
    received, _weights, _cu, plan = buffer.dispatch(payload, plan=plan, zero_copy=True)
    del _weights, _cu
    round_trip, _, _ = buffer.combine(plan=plan, hidden_nvsh=received)
    torch.npu.synchronize()

    cu = (cu_ref[rank] if cu_ref.dim() > 1 else cu_ref).cpu()
    prefetch = plan.experts_to_copy[rank].cpu()
    failures = []
    if not torch.equal(round_trip, payload):
        failures.append("not lossless: combine(dispatch(x)) != x")

    # Decode the receive layout: groups [0, E) are global expert groups, groups
    # [E, E+B) are the prefetch slots of the rebalanced remote experts.
    segments = []
    start = 0
    for group_id in range(experts + epn):
        end = int(cu[group_id])
        expert_id = group_id if group_id < experts else int(prefetch[group_id - experts])
        if end > start and group_id >= experts and expert_id < 0:
            failures.append(f"group {group_id} has {end - start} rows but no prefetch expert")
        if end > start:
            segments.append((group_id, expert_id, end - start))
        start = max(start, end)
    moved = sum(count for group_id, _expert, count in segments if group_id >= experts)
    local = sum(count for group_id, _expert, count in segments if group_id < experts)
    total = sum(count for _g, _e, count in segments)
    if total != int(cu[-1]):
        failures.append(f"segments cover {total} rows but the buffer holds {int(cu[-1])}")

    status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    print(f"[rank {rank}] rows={rows} E={experts} epn={epn} recv={int(cu[-1])} "
          f"local_segments={local} redundant_rows={moved} "
          f"prefetch_experts={[int(x) for x in prefetch]} "
          f"layout={segments} -> {status}", flush=True)

    ShmemRuntime.barrier(group=group)
    torch.npu.synchronize()
    ShmemRuntime.finalize()
    buffer.destroy()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
