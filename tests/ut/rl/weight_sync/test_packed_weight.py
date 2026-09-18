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
"""CPU tests for whole-parameter packed full-gather buckets."""
# pylint: disable=forbidden-backend-import,missing-public-docstring

import pytest
import torch

import rl.roles.weight_sync.packed_weight as packed_module
from rl.roles.weight_sync.packed_weight import (
    build_packed_weight_buckets,
    materialize_packed_weight_bucket,
    unpack_packed_weights,
)


def test_bucket_metadata_reads_local_tensor_without_dtensor_dispatch() -> None:
    """Global shape and local scalar metadata must not dispatch parallel ops."""
    class LocalMetadata:
        """Expose distributed shape metadata with a known local tensor."""
        shape = (8, 4)
        dtype = torch.float32

        @staticmethod
        def to_local() -> torch.Tensor:
            return torch.empty(4, 4)

        @staticmethod
        def element_size() -> int:
            raise AssertionError("element_size must be read from the local tensor")

        @staticmethod
        def is_floating_point() -> bool:
            raise AssertionError("is_floating_point must be read from the local tensor")

    buckets = build_packed_weight_buckets({"weight": LocalMetadata()}, 256)
    assert buckets[0].entries[0].shape == (8, 4)
    assert buckets[0].total_bytes == 128


def test_packed_plan_batches_small_weights_and_isolates_oversized_weight() -> None:
    """Verify bucket boundaries, oversized weights and tied-parameter deduplication."""
    state = {
        "a.weight": torch.arange(2, dtype=torch.float32),
        "b.weight": torch.arange(2, dtype=torch.float32),
        "large.weight": torch.arange(6, dtype=torch.float32),
        "lm_head.weight": torch.arange(2, dtype=torch.float32),
        "position_ids": torch.arange(2, dtype=torch.int64),
    }

    buckets = build_packed_weight_buckets(
        state,
        16,
        skip_names=frozenset(("lm_head.weight",)),
    )

    assert [[entry.name for entry in bucket.entries] for bucket in buckets] == [
        ["a.weight", "b.weight"],
        ["large.weight"],
    ]
    assert [bucket.total_bytes for bucket in buckets] == [16, 24]
    assert buckets[1].total_bytes > 16


def test_only_producer_packs_after_every_rank_materializes_full_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify collective materialization on all ranks and packing only on the producer."""
    calls = []

    class DistributedValue:
        """Record full-tensor materialization on each participating rank."""
        shape = (2,)
        dtype = torch.float32

        @staticmethod
        def element_size() -> int:
            return 4

        def __init__(self, name: str, values: list[float]) -> None:
            self.name = name
            self.values = values

        def full_tensor(self) -> torch.Tensor:
            calls.append(self.name)
            return torch.tensor(self.values)

    state = {
        "a.weight": DistributedValue("a", [1.0, 2.0]),
        "b.weight": DistributedValue("b", [3.0, 4.0]),
    }
    bucket = build_packed_weight_buckets(state, 32)[0]

    monkeypatch.setattr(packed_module.dist, "get_rank", lambda: 1)
    assert materialize_packed_weight_bucket(state, bucket) is None
    assert calls == ["a", "b"]

    calls.clear()
    monkeypatch.setattr(packed_module.dist, "get_rank", lambda: 0)
    packed = materialize_packed_weight_bucket(state, bucket)
    assert calls == ["a", "b"]
    assert packed is not None
    weights = unpack_packed_weights(packed, bucket.worker_metadata())
    assert [name for name, unused_tensor in weights] == ["a.weight", "b.weight"]
    torch.testing.assert_close(weights[0][1], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(weights[1][1], torch.tensor([3.0, 4.0]))


def test_packed_metadata_rejects_out_of_bounds_parameter() -> None:
    packed = torch.zeros(8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="exceeds its buffer"):
        unpack_packed_weights(
            packed,
            [{
                "name": "weight",
                "dtype_name": "float32",
                "shape": [2],
                "buffer_offset": 4,
                "num_bytes": 8,
            }],
        )
