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
