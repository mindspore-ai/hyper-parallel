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
"""Four-process CPU/Gloo coverage for external producer-defined steps."""

from datetime import timedelta

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, WorkloadCost, build_distributed_dataloader,
    build_local_balancing_dataloader,
)


def _gather(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _run_local_steps(source_api: str, broadcast_plan: bool) -> list:
    """Exercise the shared codec through locality-scoped moved and retained steps."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
    steps = [
        [[{"id": step * world_size * 2 + rank * 2 + ordinal,
           "cost": 9 if step == 0 and rank in (0, 2) else 1} for ordinal in range(2)]]
        for step in range(2)
    ]
    config = DistributedDatasetConfig(
        seq_len=10, local_batch_size=1, communication_backend="gloo", balance_group_size=2,
        broadcast_plan=broadcast_plan,
    )

    def metadata_fn(sample: dict) -> SampleMetadata:
        """Return a fixed footprint with deliberately skewed per-sample costs.

        Args:
            sample: Raw sample carrying its synthetic cost.
        """
        return SampleMetadata(5, cost=WorkloadCost(llm=sample["cost"]))

    options = {"cost_model": lambda metadata: metadata.cost, "device": "cpu"}
    if source_api == "source":
        loader = build_distributed_dataloader(
            None, mesh, config, external_step_source=steps, metadata_fn=metadata_fn,
            pack_fn=lambda samples, _seq_len: tuple(samples), collate_fn=tuple, **options,
        )
    else:
        loader = build_local_balancing_dataloader(
            steps, mesh, config, metadata_fn=metadata_fn,
            pack_fn=lambda samples, _seq_len: tuple(samples), collate_fn=tuple, **options,
        )
    try:
        expected_group = (0, 1) if rank < 2 else (2, 3)
        assert loader.group_ranks == expected_group, (rank, loader.group_ranks)
        delivered = []
        for step, batch in enumerate(loader):
            outputs = _gather(batch)
            actual = sorted(sample["id"] for output in outputs for packing_bin in output for sample in packing_bin)
            expected = list(range(step * world_size * 2, (step + 1) * world_size * 2))
            assert actual == expected, f"Local balancing changed membership: {actual} != {expected}"
            assert all(len(output) == 1 and len(output[0]) == 2 for output in outputs), outputs
            stats = dict(loader.last_balance_stats)
            all_stats = _gather(stats)
            assert all_stats[0] == all_stats[1] and all_stats[2] == all_stats[3], (
                f"Group statistics differ: stats={all_stats}"
            )
            moved = stats["moved_samples"]
            assert (moved > 0) == (step == 0), f"Unexpected movement at step {step}: {moved}"
            delivered.append((batch, stats))
        assert len(delivered) == 2, f"Expected both source steps before exhaustion, got {len(delivered)}"
    finally:
        loader.close()
    return delivered


def test_dynamic_packing_dp4_gloo() -> None:
    """Run producer-defined steps through all node-local source entry points."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        for source_api in ("source", "local"):
            baseline = _run_local_steps(source_api, broadcast_plan=True)
            independent = _run_local_steps(source_api, broadcast_plan=False)
            assert independent == baseline, (
                f"Broadcast mode changed batches/stats: actual={independent}, expected={baseline}"
            )
    finally:
        dist.destroy_process_group()
