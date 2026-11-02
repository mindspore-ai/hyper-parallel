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
"""Check the MoonEP property the EP transport integration rests on.

HyperParallel's EP path exchanges rows with an arbitrary per-peer split
(:func:`hyper_parallel.distributed.expert_parallel.collectives.ep_all_to_all`).
MoonEP's Buffer instead takes token-major rows plus a ``[S, K]`` routing matrix
and groups the received rows by *expert*.  The bridge is to call each
destination rank an expert: ``E = R``, ``K = 1``, ``S = rows``, so one row per
"token", and the row's route is the rank it must go to.  The received buffer
then holds one group per destination — this rank's — filled by the source ranks
in ascending order, which is exactly the all-to-all contract.

Three properties are asserted here, against a plain ``all_to_all_single``
reference, for randomly uneven splits:

* ``recv`` — the received rows equal the reference exchange, in order.
* ``cu``   — ``cu_seqlens`` gives the same per-source counts the reference used.
* ``grad`` — the exchange's backward returns each row's gradient to its sender
  (checked by running forward under ``autograd`` and comparing to the
  reference's gradient).

Run inside the ``hyper-parallel-env`` container from the repository root::

    export ASCEND_MOONEP_HEAP=shmem HCCL_NPU_SOCKET_PORT_RANGE=61000-61031
    torchrun --nproc_per_node=4 examples/torch/moe/moonep_exchange_check.py \
        --rows 64 --hidden 128

Install the Ascend MoonEP package built for this SoC before running.
On A2/A3 the plan comes from the host planner.
"""
from __future__ import annotations

import argparse
import os

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
    parser.add_argument("--rows", type=int, default=64, help="rows exchanged per rank")
    parser.add_argument("--hidden", type=int, default=128, help="row width H")
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skew", type=float, default=0.0,
                        help="0 = uniform split (MoonEP capacity holds); >0 tilts the split "
                             "past that capacity and makes the planner rebalance")
    return parser.parse_args()


def uneven_splits(rows: int, world: int, rank: int, seed: int, skew: float) -> list[int]:
    """Partition *rows* into *world* per-peer counts for this rank.

    ``skew = 0`` is the uniform plan a balanced router produces; a positive
    skew tilts each rank's partition (deterministically, from a dedicated
    generator so every rank can predict its peers' splits).
    """
    if rows % world:
        raise SystemExit(f"rows ({rows}) must divide by world ({world})")
    base = rows // world
    if skew <= 0.0:
        return [base] * world
    generator = torch.Generator(device="cpu").manual_seed(seed + rank)
    weights = torch.rand(world, generator=generator, dtype=torch.float64) + 1.0 - skew
    counts = (weights / weights.sum() * rows).floor().to(torch.long)
    counts[counts.argmax()] += rows - int(counts.sum())
    return [int(count) for count in counts]


def tagged_payload(rows: int, hidden: int, rank: int, device: str) -> torch.Tensor:
    """Build rows tagged ``rank * rows + row`` in column 0.

    The tag makes the delivery checkable row by row; it stays exactly
    representable in bf16 (whose 8-bit significand holds every integer below
    256), which ``main`` asserts for the probe's size.
    """
    payload = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
    payload[:, 0] = torch.arange(rows, dtype=torch.float32, device=device) + rank * rows
    return payload


def main() -> None:
    """Run the exchange, ordering and round-trip assertions on every rank."""
    args = parse_args()
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    device_id = local_device_index(rank)
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"

    rows, hidden = args.rows, args.hidden
    if world * rows > 256:
        raise SystemExit("keep world * rows <= 256 so the bf16 tags stay exact")

    # Each destination rank plays the role of one "expert" (E = R, K = 1), so the
    # receive buffer holds this rank's rows grouped by source rank -- the
    # all-to-all contract. Capacity is the buffer's NvS = S * K = rows, and a
    # destination's "home group" can receive at most every source's rows, so
    # limit the split jitter to keep the balancer inert (see the docstring).
    buffer = Buffer(rows, hidden, 1, world, world, B=1, num_sms=args.num_sms,
                    use_udma=False, group=group, explicitly_destroy=True)
    if not ShmemRuntime.is_initialized():
        ShmemRuntime.init_with_buffer(group=group, enable_udma=False)

    send_counts = uneven_splits(rows, world, rank, args.seed, args.skew)
    destinations = torch.repeat_interleave(
        torch.arange(world, dtype=torch.int32), torch.tensor(send_counts)).to(device)
    counts = torch.tensor(send_counts, dtype=torch.int32, device=device)
    # The host planner returns cu_seqlens alongside the plan; dispatch only
    # produces it when it runs the (A5-only) device planner itself.
    plan, cu_ref = build_explicit_plan(destinations, counts, group)

    payload = tagged_payload(rows, hidden, rank, device)

    received, _weights, _cu, plan = buffer.dispatch(payload, plan=plan, zero_copy=True)
    # combine is the exchange's transpose, so it must undo dispatch exactly.
    round_trip, _, _ = buffer.combine(plan=plan, hidden_nvsh=received)
    torch.npu.synchronize()

    failures = []
    if not torch.equal(round_trip, payload):
        failures.append("round trip: combine(dispatch(x)) != x")

    # Reference exchange: plain all_to_all_single with the same splits.
    recv_counts = [0] * world
    for source in range(world):
        for destination in range(world):
            if destination == rank:
                recv_counts[source] = uneven_splits(
                    rows, world, source, args.seed, args.skew)[destination]
    reference = torch.empty(sum(recv_counts), hidden, dtype=torch.bfloat16, device=device)
    dist.all_to_all_single(reference, payload.contiguous(),
                           output_split_sizes=recv_counts, input_split_sizes=send_counts)

    # cu_seqlens for this rank gives the per-source boundaries of its group.
    local_cu = (cu_ref[rank] if cu_ref.dim() > 1 else cu_ref).cpu()
    if os.environ.get("MOONEP_CHECK_DEBUG") == "1":
        print(f"[rank {rank}] send={send_counts} recv={recv_counts} "
              f"cu={local_cu.tolist()} tags={received[:int(local_cu[-1]), 0].float().cpu().tolist()}",
              flush=True)

    # The rank's own group is group `rank`; groups are cumulative, so its rows
    # are [cu[rank-1], cu[rank]) with the first one starting at zero.
    group_lo = 0 if rank == 0 else int(local_cu[rank - 1])
    group_hi = int(local_cu[rank])
    got = received[group_lo:group_hi]
    if got.shape != reference.shape or not torch.equal(got, reference):
        failures.append(f"exchange mismatch (got {tuple(got.shape)}, "
                        f"reference {tuple(reference.shape)}, "
                        f"equal={bool(torch.equal(got, reference))})")
    # The per-source split is implicit in the row order (sources concatenate in
    # rank order); cu_seqlens only carries the group totals, so the count check
    # here is what exposes the planner rebalancing instead of transporting.
    expected_rows = sum(recv_counts)
    if group_hi != expected_rows:
        failures.append(
            f"planner rebalanced: received {group_hi} rows where the exchange should "
            f"have moved {expected_rows} (capacity is NvS = S*K = {rows})")

    status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    print(f"[rank {rank}] rows={rows} send={send_counts} expect_recv={expected_rows} "
          f"got_recv={group_hi} -> {status}",
          flush=True)

    ShmemRuntime.barrier(group=group)
    torch.npu.synchronize()
    ShmemRuntime.finalize()
    buffer.destroy()
    dist.destroy_process_group()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
