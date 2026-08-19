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
"""Tests for the Trainer-targeted dataloader component."""

import pytest
import torch
from torch.utils.data.distributed import DistributedSampler

from hyper_models.components.data import DataLoader


def test_dataloader_partitions_samples_across_dp_ranks() -> None:
    """Use a distributed sampler to give each DP rank disjoint samples."""
    dataset = list(range(8))
    rank_zero_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        dp_world_size=2,
        dp_rank=0,
    )
    rank_one_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        dp_world_size=2,
        dp_rank=1,
    )

    rank_zero_samples = torch.cat(list(rank_zero_loader)).tolist()
    rank_one_samples = torch.cat(list(rank_one_loader)).tolist()

    assert isinstance(rank_zero_loader.sampler, DistributedSampler)
    assert rank_zero_samples == [0, 2, 4, 6]
    assert rank_one_samples == [1, 3, 5, 7]


def test_dataloader_forwards_epoch_to_distributed_sampler() -> None:
    """Update deterministic shuffle state at every Trainer epoch."""
    dataloader = DataLoader(
        list(range(8)),
        batch_size=2,
        dp_world_size=2,
        dp_rank=0,
    )

    dataloader.set_epoch(3)

    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.epoch == 3


def _build_resumable_loader(dp_rank: int = 0) -> DataLoader:
    """Build a deterministic shuffled loader used by the resume tests."""
    return DataLoader(
        list(range(16)),
        batch_size=2,
        shuffle=True,
        drop_last=True,
        dp_world_size=2,
        dp_rank=dp_rank,
        seed=1234,
    )


def test_dataloader_state_dict_records_consumed_batches() -> None:
    """Track how far into the epoch this rank has read."""
    dataloader = _build_resumable_loader()
    dataloader.set_epoch(2)

    iterator = iter(dataloader)
    next(iterator)
    next(iterator)

    state = dataloader.state_dict()

    assert state["epoch"] == 2
    assert state["batches_consumed"] == 2
    assert state["batch_size"] == 2
    assert state["dp_world_size"] == 2
    assert "dp_rank" not in state


def test_dataloader_resumes_at_the_saved_position() -> None:
    """Replay exactly the batches the interrupted run had not consumed yet."""
    original = _build_resumable_loader()
    original.set_epoch(2)

    consumed = []
    iterator = iter(original)
    for _ in range(3):
        consumed.append(next(iterator).tolist())
    state = original.state_dict()
    remaining = [batch.tolist() for batch in iterator]

    resumed = _build_resumable_loader()
    resumed.load_state_dict(state)
    resumed.set_epoch(2)

    assert [batch.tolist() for batch in resumed] == remaining
    # The tail must not replay anything the first run already trained on.
    assert not set(sum(remaining, [])) & set(sum(consumed, []))


def test_dataloader_resumes_a_shuffled_epoch_without_a_distributed_sampler() -> None:
    """Reproduce the shuffle order across a restart at dp_world_size == 1."""
    original = DataLoader(list(range(16)), batch_size=2, shuffle=True, seed=1234)

    iterator = iter(original)
    for _ in range(3):
        next(iterator)
    state = original.state_dict()
    remaining = [batch.tolist() for batch in iterator]

    resumed = DataLoader(list(range(16)), batch_size=2, shuffle=True, seed=1234)
    resumed.load_state_dict(state)

    assert [batch.tolist() for batch in resumed] == remaining


def test_dataloader_restarts_counter_on_a_new_epoch() -> None:
    """Drop the restored position once training moves past that epoch."""
    dataloader = _build_resumable_loader()
    dataloader.load_state_dict({"epoch": 2, "batches_consumed": 3, "batch_size": 2, "dp_world_size": 2})

    dataloader.set_epoch(3)

    assert dataloader.state_dict()["batches_consumed"] == 0
    assert len(list(dataloader)) == len(list(_build_resumable_loader()))


def test_dataloader_discards_state_from_a_different_topology() -> None:
    """Ignore a saved position whose sample partition no longer applies."""
    dataloader = _build_resumable_loader()

    dataloader.load_state_dict(
        {"epoch": 1, "batches_consumed": 3, "batch_size": 2, "dp_world_size": 4}
    )

    assert dataloader.state_dict()["batches_consumed"] == 0
    assert dataloader.state_dict()["epoch"] == 0


@pytest.mark.parametrize(
    ("dp_world_size", "dp_rank"),
    [
        (0, 0),
        (2, -1),
        (2, 2),
    ],
)
def test_dataloader_rejects_invalid_dp_topology(
    dp_world_size: int,
    dp_rank: int,
) -> None:
    """Reject invalid data-parallel size and rank combinations."""
    with pytest.raises(ValueError):
        DataLoader(
            list(range(8)),
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
