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
from typing import Any

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, WorkloadCost, build_distributed_dataloader,
    build_distributed_dataset, build_local_balancing_dataloader,
)
from hyper_parallel.distributed_data.schema import BufferedSampleMetadata, SampleKey


class _StepReader:
    """Produce variable sample counts without indexed payload access or read-ahead."""

    def __init__(self, rank: int, reader_idx: int) -> None:
        """Store ownership and initialize a replayable two-step stream."""
        self.rank = rank
        self.reader_idx = reader_idx
        self.epoch = 0
        self.batch_position = 0
        self.reference_bins = ()
        self.exhausted = False
        self.payloads = {}

    def prepare_next_step(self) -> None:
        """Expose exactly one producer-selected local step."""
        if self.reference_bins or self.exhausted:
            return
        if self.batch_position == 2:
            self.exhausted = True
            return
        lengths = (5, 5) if self.batch_position == 0 else (2, 3, 5)
        items = []
        for ordinal, tokens in enumerate(lengths):
            sample_id = self.batch_position * 6 + self.reader_idx * 3 + ordinal
            key = SampleKey(self.rank, sample_id, sample_id)
            metadata = SampleMetadata(tokens, cost=WorkloadCost(llm=9 if self.reader_idx == 0 else 1),
                                      sample_id=sample_id)
            items.append(BufferedSampleMetadata(key, metadata, ordinal))
            self.payloads[key] = {"id": sample_id, "tokens": tokens, "source": self.rank}
        self.reference_bins = (tuple(items),)

    def metadata(self) -> tuple[BufferedSampleMetadata, ...]:
        """Return only this step's metadata."""
        return tuple(item for packing_bin in self.reference_bins for item in packing_bin)

    def selected_payloads(self, keys: set[SampleKey]) -> tuple[tuple[SampleKey, Any], ...]:
        """Return already-buffered payloads.

        Args:
            keys: Selected sample occurrences owned by this Reader.
        """
        return tuple((key, self.payloads[key]) for key in keys)

    def commit(self, keys: set[SampleKey]) -> None:
        """Commit the whole selected local step.

        Args:
            keys: Occurrences consumed by the current step.
        """
        if keys != set(self.payloads):
            raise ValueError("A producer step must be committed in full.")
        self.batch_position += 1
        self.reference_bins = ()
        self.payloads = {}

    def state_dict(self) -> dict[str, int]:
        """Checkpoint committed progress, not a speculative fill."""
        return {"epoch": self.epoch, "position": self.batch_position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Recreate future payloads from the committed position.

        Args:
            state: Previously saved committed cursor.
        """
        self.set_epoch(state["epoch"])
        self.batch_position = state["position"]

    def set_epoch(self, epoch: int) -> None:
        """Reset the deterministic source.

        Args:
            epoch: Source epoch to replay.
        """
        self.epoch = epoch
        self.batch_position = 0
        self.reference_bins = ()
        self.exhausted = False
        self.payloads = {}


def _gather(value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _loader(mesh, reader_ranks):
    rank = dist.get_rank()
    reader = _StepReader(rank, reader_ranks.index(rank)) if rank in reader_ranks else None
    loader = build_distributed_dataloader(
        None, mesh,
        DistributedDatasetConfig(
            seq_len=10, local_batch_size=1, dataset_reader_ranks=reader_ranks,
            communication_backend="gloo",
        ),
        external_step_reader=reader,
        device="cpu", cost_model=lambda metadata: metadata.cost,
    )
    return loader


def _run_steps(mesh, reader_ranks):
    """Check step membership, routing, model peers, and checkpoint replay."""
    loader = _loader(mesh, reader_ranks)
    checkpoint = None
    second = None
    for step in range(2):
        batch = next(loader)
        outputs = _gather(batch)
        assert outputs[0] == outputs[1] and outputs[2] == outputs[3], (
            f"MP peers disagree: outputs={outputs}"
        )
        count = 2 if step == 0 else 3
        expected = sorted(step * 6 + reader_idx * 3 + ordinal
                          for reader_idx in range(2) for ordinal in range(count))
        actual = sorted(sample["id"] for rank in (0, 2) for packing_bin in outputs[rank] for sample in packing_bin)
        assert actual == expected, f"Step membership changed: actual={actual}, expected={expected}"
        for rank in (0, 2):
            lengths = [sum(sample["tokens"] for sample in packing_bin) for packing_bin in outputs[rank]]
            assert lengths == [10], f"Invalid packing lengths: rank={rank}, actual={lengths}, expected={[10]}"
        if step == 0:
            checkpoint = loader.state_dict()
        else:
            second = batch
    assert not list(loader), f"Expected exactly two producer steps, plan={loader.last_plan}"

    resumed = _loader(mesh, reader_ranks)
    resumed.load_state_dict(checkpoint)
    replay = list(resumed)
    assert replay == [second], f"Replay changed membership: actual={replay}, expected={[second]}"
    resumed.wait_for_prefetch()


def _run_local_steps(dataset_api: bool) -> None:
    """Exercise the shared codec through locality-scoped moved and retained steps."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
    steps = [
        [[{"id": step * world_size * 2 + rank * 2 + ordinal,
           "cost": 9 if step == 0 and rank == 0 else 1} for ordinal in range(2)]]
        for step in range(2)
    ]
    config = DistributedDatasetConfig(seq_len=10, local_batch_size=1, communication_backend="gloo")

    def metadata_fn(sample: dict) -> SampleMetadata:
        """Return a fixed footprint with deliberately skewed per-sample costs.

        Args:
            sample: Raw sample carrying its synthetic cost.
        """
        return SampleMetadata(5, cost=WorkloadCost(llm=sample["cost"]))

    options = {"cost_model": lambda metadata: metadata.cost, "device": "cpu", "max_steps": 2}
    if dataset_api:
        dataset = build_distributed_dataset(steps, metadata=metadata_fn, collate_fn=tuple)
        loader = build_distributed_dataloader(dataset, mesh, config, **options)
    else:
        loader = build_local_balancing_dataloader(
            steps, mesh, config, metadata_fn=metadata_fn,
            pack_fn=lambda samples, _seq_len: tuple(samples), collate_fn=tuple, **options,
        )
    try:
        for step in range(2):
            batch = next(loader)
            outputs = _gather(batch)
            actual = sorted(sample["id"] for output in outputs for packing_bin in output for sample in packing_bin)
            expected = list(range(step * world_size * 2, (step + 1) * world_size * 2))
            assert actual == expected, f"Local balancing changed membership: {actual} != {expected}"
            assert all(len(output) == 1 and len(output[0]) == 2 for output in outputs), outputs
            stats = dict(loader.last_balance_stats)
            moved = stats["moved_samples"]
            assert (moved > 0) == (step == 0), f"Unexpected movement at step {step}: {moved}"
    finally:
        loader.close()


def test_dynamic_packing_dp2_mp2_gloo() -> None:
    """Run producer-defined steps on Constructor-owned and separate Reader ranks."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "mp"))
        try:
            build_distributed_dataloader(
                [0], mesh, DistributedDatasetConfig(seq_len=10, local_batch_size=1, communication_backend="gloo"),
                device="cpu",
                metadata=[SampleMetadata(1)],
            )
        except ValueError as error:
            assert "Metadata mode requires batch_sampler" in str(error), f"Unexpected build error: {error}"
        else:
            raise AssertionError("Metadata streaming without a sampler must be rejected.")
        for reader_ranks in ((0, 2), (1, 3)):
            _run_steps(mesh, reader_ranks)
        for dataset_api in (False, True):
            _run_local_steps(dataset_api)
    finally:
        dist.destroy_process_group()
