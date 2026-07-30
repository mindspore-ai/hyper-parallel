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
"""Tests for Trainer-compatible micro-batch collation."""

import pytest
import torch
from torch.utils.data import DataLoader

from hyper_models.components.data import (
    MakeMicroBatchCollator,
    calculate_num_micro_batches,
)


def _make_features(count: int) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.tensor([index, index + 1]),
            "labels": torch.tensor([index + 1, index + 2]),
        }
        for index in range(count)
    ]


def test_calculate_num_micro_batches() -> None:
    assert calculate_num_micro_batches(
        global_batch_size=32,
        micro_batch_size=2,
        dp_world_size=4,
    ) == 4


@pytest.mark.parametrize(
    ("global_batch_size", "micro_batch_size", "dp_world_size"),
    [
        (0, 1, 1),
        (8, 0, 1),
        (8, 1, 0),
        (10, 2, 4),
    ],
)
def test_calculate_num_micro_batches_rejects_invalid_sizes(
    global_batch_size: int,
    micro_batch_size: int,
    dp_world_size: int,
) -> None:
    with pytest.raises(ValueError):
        calculate_num_micro_batches(
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            dp_world_size=dp_world_size,
        )


def test_standard_dataloader_returns_micro_batch_list() -> None:
    global_batch_size = 8
    micro_batch_size = 2
    dp_world_size = 1
    num_micro_batch = calculate_num_micro_batches(
        global_batch_size,
        micro_batch_size,
        dp_world_size,
    )
    dataloader = DataLoader(
        _make_features(global_batch_size),
        batch_size=global_batch_size // dp_world_size,
        collate_fn=MakeMicroBatchCollator(num_micro_batch),
    )

    micro_batches = next(iter(dataloader))

    assert isinstance(micro_batches, list)
    assert len(micro_batches) == num_micro_batch
    assert all(isinstance(micro_batch, dict) for micro_batch in micro_batches)
    assert all(micro_batch["input_ids"].shape == (micro_batch_size, 2) for micro_batch in micro_batches)
    assert micro_batches[0]["input_ids"].tolist() == [[0, 1], [1, 2]]


def test_single_micro_batch_is_still_wrapped_in_list() -> None:
    collator = MakeMicroBatchCollator(num_micro_batch=1)

    micro_batches = collator(_make_features(2))

    assert len(micro_batches) == 1
    assert micro_batches[0]["input_ids"].shape == (2, 2)


def test_collator_accepts_single_item_transform_wrappers() -> None:
    collator = MakeMicroBatchCollator(num_micro_batch=2)
    wrapped_features = [(feature,) for feature in _make_features(4)]

    micro_batches = collator(wrapped_features)

    assert len(micro_batches) == 2
    assert all(micro_batch["labels"].shape == (2, 2) for micro_batch in micro_batches)


def test_collator_rejects_incomplete_micro_batch_group() -> None:
    collator = MakeMicroBatchCollator(num_micro_batch=2)

    with pytest.raises(ValueError, match="must be divisible"):
        collator(_make_features(3))


def test_collator_rejects_non_mapping_internal_output() -> None:
    collator = MakeMicroBatchCollator(
        num_micro_batch=1,
        internal_data_collator=lambda features: list(features),
    )

    with pytest.raises(ValueError, match="must return a mapping"):
        collator(_make_features(2))
