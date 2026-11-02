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
"""Check the vectorised MoonEP planner against MoonEP's installed C++ host planner.

``hyper_parallel.distributed.expert_parallel.moonep_plan`` rebuilds the plan with
tensor ops because the reference walks tokens one at a time in Python.  Both
must agree field for field, including the rebalancing branch: this script builds
both plans on several routings (uniform, moderately skewed, and hard-skewed so
the planner must migrate rows onto spare capacity) and compares
``dst`` / ``cu_seqlens`` / ``experts_to_copy`` / ``remote_stats`` /
``zero_fill_ranges``.

Run inside the ``hyper-parallel-env`` container from the repository root::

    export ASCEND_MOONEP_HEAP=shmem HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
    torchrun --nproc_per_node=4 examples/torch/moe/moonep_plan_parity_check.py \
        --rows 48 --experts 8 --hidden 32
"""
from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  # pylint: disable=unused-import  (registers the npu device)
from moonep import Buffer
from npu_patch import ShmemRuntime
from npu_patch._device import local_device_index
from npu_patch.planning_host import build_plan_host

from hyper_parallel.distributed.expert_parallel.moonep_plan import (
    _build_moonep_plan_python,
)


def parse_args() -> argparse.Namespace:
    """Parse the check's shape arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=48, help="routed rows per rank")
    parser.add_argument("--experts", type=int, default=8, help="total experts E")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--num-sms", type=int, default=32)
    return parser.parse_args()


def routing(rows: int, experts: int, world: int, rank: int, mode: str) -> torch.Tensor:
    """Return ``[rows]`` global expert ids in expert-major order for this rank."""
    generator = torch.Generator(device="cpu").manual_seed(1234 + rank)
    if mode == "uniform":
        weights = torch.ones(experts)
    elif mode == "skewed":
        weights = torch.rand(experts, generator=generator) + 0.2
    else:  # "sink": everything onto the leading rank's experts
        weights = torch.zeros(experts)
        weights[:experts // world] = 1.0
    drawn = torch.multinomial(weights, rows, replacement=True, generator=generator)
    return torch.sort(drawn.to(torch.int32)).values


def main() -> None:
    """Compare both planners on three routings and report per-field agreement."""
    args = parse_args()
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device_id = local_device_index(rank)
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    rows, experts, hidden = args.rows, args.experts, args.hidden
    if experts % world:
        raise SystemExit(f"experts ({experts}) must divide world ({world})")
    epn = experts // world

    buffer = Buffer(rows, hidden, 1, experts, world, B=epn, num_sms=args.num_sms,
                    use_udma=False, group=dist.group.WORLD, explicitly_destroy=True)
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=dist.group.WORLD, enable_udma=False)

    failures = []
    invariant_notes = []
    for mode in ("uniform", "skewed", "sink"):
        ids = routing(rows, experts, world, rank, mode).to(device)
        counts = torch.bincount(ids.long(), minlength=experts).to(torch.int32)
        counts_all = [torch.empty_like(counts) for _ in range(world)]
        dist.all_gather(counts_all, counts, group=dist.group.WORLD)
        counts_all = torch.stack(counts_all)

        reference = build_plan_host(counts_all, counts, ids, rank, epn, rows)
        ref_dst = reference["dst"]
        ref_cu = reference["cu_seqlens"]
        ref_copy = reference["experts_to_copy"]
        ref_stats = reference["remote_stats"]
        ref_fill = reference["zero_fill_ranges"]
        ref_alloc = reference["alloc"]
        mine = _build_moonep_plan_python(counts_all, counts, ids, rank, epn, rows)

        checks = {
            "dst": torch.equal(mine["dst"].cpu(), ref_dst.cpu()),
            "cu_seqlens": torch.equal(mine["cu_seqlens"].cpu(), ref_cu.cpu()),
            "experts_to_copy": torch.equal(mine["experts_to_copy"].cpu(), ref_copy.cpu()),
            "remote_stats": torch.equal(mine["remote_stats"].cpu(), ref_stats.cpu()),
            "zero_fill": torch.equal(mine["zero_fill_ranges"].cpu(), ref_fill.cpu()),
            "alloc": torch.equal(mine["alloc"].cpu(), ref_alloc.cpu()),
        }
        # Both planners must at least conserve rows and respect the per-rank
        # capacity; if the reference itself breaks that where they differ, the
        # divergence is the reference's, not the vectorised planner's.
        total = counts_all.sum(dim=0).long().cpu()
        ref_a = ref_alloc.cpu().long()
        mine_a = mine["alloc"].cpu().long()
        notes = []
        for name, alloc in (("ref", ref_a), ("mine", mine_a)):
            # alloc is the kernel's [R, E] layout: column sums are per expert,
            # row sums are what each destination receives.
            conserved = torch.equal(alloc.sum(dim=0), total)
            over = int((alloc.sum(dim=1) - rows).clamp(min=0).sum())
            notes.append(f"{name}:conserved={conserved},over_capacity_rows={over}")
        invariant_notes.append(f"rank{rank}/{mode} " + " ".join(notes))
        if not torch.equal(mine_a.sum(dim=0), total) or int(
                (mine_a.sum(dim=1) - rows).clamp(min=0).sum()) > 0:
            failures.append(f"{mode}: my alloc breaks conservation or capacity")
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            failures.append(f"{mode}: {bad}")
        span = mine["local_span"]
        print(f"[rank {rank}] mode={mode:7s} rows={rows} "
              f"copy_ref={[int(x) for x in ref_copy[rank]]} "
              f"copy_mine={[int(x) for x in mine['experts_to_copy'][rank]]} "
              f"dst_ref[:6]={[int(x) for x in ref_dst[:6]]} "
              f"dst_mine[:6]={[int(x) for x in mine['dst'][:6]]} span={span} "
              f"-> {'PASS' if not bad else 'FAIL ' + ','.join(bad)}", flush=True)

    ShmemRuntime.barrier(group=dist.group.WORLD)
    torch.npu.synchronize()
    ShmemRuntime.finalize()
    buffer.destroy()
    if rank == 0:
        for line in invariant_notes:
            print(f"[invariants] {line}", flush=True)
    dist.destroy_process_group()
    if failures:
        raise SystemExit(f"planner mismatch: {failures}")
    if rank == 0:
        print("PLAN PARITY PASS", flush=True)


if __name__ == "__main__":
    main()
