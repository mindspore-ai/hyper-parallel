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
"""DP2/TP2 native BatchSampler ownership, metadata, and checkpoint coverage."""

from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import patch

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig, SampleMetadata, WorkloadCost, build_distributed_dataloader,
)


class _Dataset:
    """Track native index reads independently from metadata lookup."""

    def __init__(self) -> None:
        """Initialize the payload read log."""
        self.reads = []

    def __len__(self) -> int:
        """Return a size with an incomplete distributed tail."""
        return 18

    def __getitem__(self, index: int) -> dict:
        """Record a read and return original model fields."""
        self.reads.append(index)
        return {"id": index, "labels": [index + 1], "position_ids": [7], "loss_mask": [1]}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(1, cost=WorkloadCost(llm=9 if sample["id"] % 4 < 2 else 1), sample_id=sample["id"])


def _sampler(sampler_type: str, data_sharding: bool) -> object:
    return build_dataset_batch_sampler(
        total_samples=18, micro_batch_size=2, global_batch_size=8,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
        sampler_type=sampler_type, data_sharding=data_sharding, seed=17,
        index_mapping=[index // 2 for index in range(18)] if sampler_type == "single" else None,
    )


def _gather(value: object) -> list:
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _run_case(
        mesh: object, *, metadata_mode: bool, double_buffer: bool, sampler_type: str, data_sharding: bool,
) -> None:
    rank = dist.get_rank()
    dataset = _Dataset()
    reference = list(_sampler(sampler_type, data_sharding))
    source_sampler = _sampler(sampler_type, data_sharding)
    config = DistributedDatasetConfig(seq_len=8, local_batch_size=2, double_buffer=double_buffer)
    metadata_options = (
        {"metadata": [_metadata({"id": index}) for index in range(len(dataset))]}
        if metadata_mode else {"metadata_fn": _metadata}
    )
    no_a2a = patch.object(dist, "all_to_all_single", side_effect=AssertionError("metadata used A2A"))
    with no_a2a if metadata_mode else nullcontext():
        loader = build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config, batch_sampler=source_sampler, **metadata_options,
        )
        saved_state = None
        delivered = []
        for round_idx, expected_local_indices in enumerate(reference):
            batch = next(loader)
            delivered.append(batch)
            all_batches = _gather(batch)
            expected_indices = _gather(expected_local_indices)
            expected = Counter(expected_indices[0] + expected_indices[2])
            actual = Counter(sample["id"] for target in (0, 2) for sample in all_batches[target])
            assert actual == expected, f"Native round membership changed: actual={actual}, expected={expected}"
            assert all_batches[0] == all_batches[1] and all_batches[2] == all_batches[3], (
                f"TP peers received different batches: batches={all_batches}"
            )
            for sample in batch:
                assert sample["labels"] == [sample["id"] + 1] and sample["position_ids"] == [7], (
                    f"Original native fields changed: sample={sample}"
                )
            if rank % 2:
                assert source_sampler.consumed_samples == 0 and not dataset.reads, (
                    f"TP peer advanced sampler or read payloads: cursor={source_sampler.consumed_samples}, "
                    f"reads={dataset.reads}"
                )
            if round_idx == 0:
                saved_state = loader.state_dict()
                if rank % 2 == 0:
                    owner = "metadata_reader" if metadata_mode else "dataset_reader"
                    saved_cursor = saved_state[owner]["sampler"]["consumed_samples"]
                    assert saved_cursor == 4, f"Saved speculative sampler cursor: actual={saved_cursor}, expected=4"

        assert not list(loader), f"Native sampler should exhaust after {len(reference)} rounds"
        all_reads = _gather(dataset.reads)
        actual_reads = Counter(index for rank_reads in all_reads for index in rank_reads)
        expected_reads = Counter(index for refs in _gather(reference)[::2] for row in refs for index in row)
        # Checkpointing can replay a prepared metadata read, but never advances membership.
        if not (metadata_mode and double_buffer):
            assert actual_reads == expected_reads, (
                f"Read coverage differs: actual={actual_reads}, expected={expected_reads}"
            )
        resumed = build_distributed_dataloader(
            dataset if rank % 2 == 0 else None, mesh, config,
            batch_sampler=_sampler(sampler_type, data_sharding), **metadata_options,
        )
        resumed.load_state_dict(saved_state)
        remaining = list(resumed)
        assert remaining == delivered[1:], f"Native replay differs: actual={remaining}, expected={delivered[1:]}"


def _run_invalid_round(mesh: object, *, uneven_exhaustion: bool) -> None:
    """Fail every model peer together if native sampler rounds disagree."""
    sampler = _sampler("single", False)
    if not uneven_exhaustion and dist.get_rank() // 2 == 1:
        sampler.consumed_samples = 4
    stop_early = uneven_exhaustion and dist.get_rank() // 2 == 0
    changed_iterator = patch.object(type(sampler), "__iter__", return_value=iter(()))
    with changed_iterator if stop_early else nullcontext():
        loader = build_distributed_dataloader(
            _Dataset(), mesh, DistributedDatasetConfig(seq_len=8, local_batch_size=2),
            batch_sampler=sampler, metadata_fn=_metadata,
        )
        failure = None
        try:
            next(loader)
        except RuntimeError as exc:
            failure = str(exc)
    expected = "exhausted at different" if uneven_exhaustion else "inconsistent consumed_samples"
    errors = _gather(failure)
    assert all(error is not None and expected in error for error in errors), (
        f"Expected collective {expected!r} error on all ranks, got errors={errors}"
    )


def _run_cost_balance(mesh: object) -> None:
    """Redistribute unequal native samples while keeping the selected four outputs."""
    sampler = build_dataset_batch_sampler(
        total_samples=18, micro_batch_size=2, global_batch_size=8,
        dp_rank=dist.get_rank() // 2, dp_world_size=2,
    )
    loader = build_distributed_dataloader(
        _Dataset(), mesh, DistributedDatasetConfig(seq_len=8, local_batch_size=2),
        batch_sampler=sampler, metadata_fn=_metadata,
    )
    next(loader)
    plan = _gather(loader.last_plan)[0]
    actual_costs = [constructor.cost.llm for constructor in plan.constructors]
    assert actual_costs == [10, 10], f"Expected costs=[10, 10] instead of native [18, 2], got costs={actual_costs}"
    actual_indices = sorted(key.dataset_index for key in plan.selected_keys)
    assert actual_indices == [0, 1, 2, 3], f"Cost balancing changed the native round: indices={actual_indices}"


def test_native_batch_sampler_dp2_tp2_gloo() -> None:
    """Check native DP slicing and collective replay under both data paths."""
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        for metadata_mode in (False, True):
            for double_buffer in (False, True):
                for sampler_type, data_sharding in (("single", False), ("cyclic", False), ("cyclic", True)):
                    _run_case(mesh, metadata_mode=metadata_mode, double_buffer=double_buffer,
                              sampler_type=sampler_type, data_sharding=data_sharding)
        _run_invalid_round(mesh, uneven_exhaustion=False)
        _run_invalid_round(mesh, uneven_exhaustion=True)
        _run_cost_balance(mesh)
        dist.barrier()
    finally:
        dist.destroy_process_group()
