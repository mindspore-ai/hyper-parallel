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
"""Standalone end-to-end tests for Source Loader to Data Constructor flow."""

import unittest
from typing import Any

from hyper_parallel.distributed_data import (
    DistributedDataLoader,
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
    default_collate_fn,
    default_pack_fn,
)


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _build_standard_loader(
        samples: list[dict[str, Any]],
        config: DistributedDatasetConfig,
        events: list[tuple[str, Any]],
) -> DistributedDataLoader:
    def metadata_fn(sample: dict[str, Any]) -> SampleMetadata:
        """Derive planner metadata from a test sample."""
        events.append(("metadata", sample["id"]))
        return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

    def pack_fn(raw_samples: list[dict[str, Any]], seq_len: int) -> dict[str, Any]:
        """Record and combine samples in one planned bin."""
        sample_ids = tuple(sample["id"] for sample in raw_samples)
        token_count = sum(sample["tokens"] for sample in raw_samples)
        if token_count > seq_len:
            raise ValueError(f"test pack received {token_count} tokens for seq_len={seq_len}")
        events.append(("pack", sample_ids))
        return {"sample_ids": sample_ids, "tokens": token_count}

    def collate_fn(packed_sequences: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        """Record and return one immutable local batch."""
        events.append(("collate", len(packed_sequences)))
        return tuple(packed_sequences)

    return build_distributed_dataloader(
        samples,
        _StandaloneMesh(),
        config,
        metadata_fn=metadata_fn,
        pack_fn=pack_fn,
        collate_fn=collate_fn,
    )


def _drain(loader: DistributedDataLoader) -> tuple[list[Any], list[str]]:
    batches = []
    plan_ids = []
    while True:
        try:
            batches.append(next(loader))
        except StopIteration:
            break
        plan_ids.append(loader.last_plan_id)
    return batches, plan_ids


class TestDistributedDataLoaderEndToEnd(unittest.TestCase):
    """Verify the collective orchestration in its one-rank reference mode."""

    def test_default_callbacks_return_planned_bins_of_raw_samples(self) -> None:
        """Omitted callbacks preserve raw samples while respecting Planner bin capacity."""
        samples = [
            {"id": 0, "tokens": 7},
            {"id": 1, "tokens": 3},
            {"id": 2, "tokens": 6},
            {"id": 3, "tokens": 4},
        ]

        def metadata_fn(sample: dict[str, Any]) -> SampleMetadata:
            """Expose the raw sample's packing footprint."""
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        loader = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            DistributedDatasetConfig(seq_len=10, local_batch_size=2, buffer_size_multiplier=1.0),
            metadata_fn=metadata_fn,
        )

        batch = next(loader)

        self.assertEqual(batch, ((samples[0], samples[1]), (samples[2], samples[3])))
        self.assertIsInstance(batch, tuple)
        self.assertTrue(all(isinstance(packing_bin, tuple) for packing_bin in batch))
        self.assertTrue(all(sum(sample["tokens"] for sample in packing_bin) <= 10 for packing_bin in batch))

    def test_dynamically_packs_samples_then_collates_local_sequences(self) -> None:
        """Explicit callbacks still run per planned bin and local batch."""
        samples = [
            {"id": 0, "tokens": 6},
            {"id": 1, "tokens": 4},
            {"id": 2, "tokens": 6},
            {"id": 3, "tokens": 4},
        ]
        events: list[tuple[str, Any]] = []
        config = DistributedDatasetConfig(
            seq_len=10,
            local_batch_size=2,
            buffer_size_multiplier=1.0,
        )
        loader = _build_standard_loader(samples, config, events)

        batch = next(loader)

        self.assertEqual(
            batch,
            (
                {"sample_ids": (0, 1), "tokens": 10},
                {"sample_ids": (2, 3), "tokens": 10},
            ),
        )
        self.assertEqual(
            events,
            [
                ("metadata", 0),
                ("metadata", 1),
                ("metadata", 2),
                ("metadata", 3),
                ("pack", (0, 1)),
                ("pack", (2, 3)),
                ("collate", 2),
            ],
        )
        self.assertIsNotNone(loader.last_plan_id)
        with self.assertRaises(StopIteration):
            next(loader)

    def test_prepacked_sample_can_be_one_complete_local_batch(self) -> None:
        """A full-length raw sample naturally remains a singleton constructor bin."""
        samples = [
            {"id": 0, "tokens": 8, "batch": {"input_ids": [10, 11]}},
            {"id": 1, "tokens": 8, "batch": {"input_ids": [20, 21]}},
        ]
        pack_inputs: list[tuple[int, ...]] = []

        def metadata_fn(sample: dict[str, Any]) -> SampleMetadata:
            """Describe an already packed sample as one full sequence."""
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        def pack_fn(raw_samples: list[dict[str, Any]], seq_len: int) -> dict[str, Any]:
            """Unwrap the singleton prepacked batch."""
            self.assertEqual(seq_len, 8)
            pack_inputs.append(tuple(sample["id"] for sample in raw_samples))
            self.assertEqual(len(raw_samples), 1)
            return raw_samples[0]["batch"]

        def collate_fn(packed_sequences: list[dict[str, Any]]) -> dict[str, Any]:
            """Return the one already complete local batch."""
            self.assertEqual(len(packed_sequences), 1)
            return packed_sequences[0]

        loader = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            DistributedDatasetConfig(seq_len=8, local_batch_size=1, buffer_size_multiplier=1.0),
            metadata_fn=metadata_fn,
            pack_fn=pack_fn,
            collate_fn=collate_fn,
        )

        self.assertEqual(list(loader), [{"input_ids": [10, 11]}, {"input_ids": [20, 21]}])
        self.assertEqual(pack_inputs, [(0,), (1,)])

    def test_checkpoint_resume_matches_uninterrupted_batches_and_plan_ids(self) -> None:
        """Read-ahead samples resume exactly without duplication or replanning drift."""
        samples = [{"id": index, "tokens": 6 if index % 2 == 0 else 4} for index in range(6)]
        config = DistributedDatasetConfig(
            seq_len=10,
            local_batch_size=1,
            buffer_size_multiplier=2.0,
        )
        baseline_events: list[tuple[str, Any]] = []
        baseline = _build_standard_loader(samples, config, baseline_events)
        first_batch = next(baseline)
        checkpoint = baseline.state_dict()
        expected_batches, expected_plan_ids = _drain(baseline)

        resumed_events: list[tuple[str, Any]] = []
        resumed = _build_standard_loader(samples, config, resumed_events)
        resumed.load_state_dict(checkpoint)
        actual_batches, actual_plan_ids = _drain(resumed)

        self.assertEqual(first_batch, ({"sample_ids": (0, 1), "tokens": 10},))
        self.assertEqual(actual_batches, expected_batches)
        self.assertEqual(actual_plan_ids, expected_plan_ids)
        remaining_ids = [
            sample_id
            for batch in actual_batches
            for packed in batch
            for sample_id in packed["sample_ids"]
        ]
        self.assertEqual(remaining_ids, [2, 3, 4, 5])

    def test_checkpoint_fingerprint_canonicalizes_default_callback_mode(self) -> None:
        """Explicit defaults resume default state, while custom construction is incompatible."""
        samples = [{"id": index, "tokens": 6 if index % 2 == 0 else 4} for index in range(4)]
        config = DistributedDatasetConfig(seq_len=10, local_batch_size=1, buffer_size_multiplier=1.0)

        def metadata_fn(sample: dict[str, Any]) -> SampleMetadata:
            """Expose the sample token footprint for all three loaders."""
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        omitted_defaults = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            config,
            metadata_fn=metadata_fn,
        )
        self.assertEqual(next(omitted_defaults), ((samples[0], samples[1]),))
        checkpoint = omitted_defaults.state_dict()

        explicit_defaults = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            config,
            metadata_fn=metadata_fn,
            pack_fn=default_pack_fn,
            collate_fn=default_collate_fn,
        )
        explicit_defaults.load_state_dict(checkpoint)
        self.assertEqual(next(explicit_defaults), ((samples[2], samples[3]),))

        custom_callbacks = _build_standard_loader(samples, config, [])
        with self.assertRaisesRegex(ValueError, "config_fingerprint"):
            custom_callbacks.load_state_dict(checkpoint)

    def test_eof_drops_a_tail_that_cannot_fill_every_local_bin(self) -> None:
        """The default drop-last contract stops before invoking constructor callbacks."""
        events: list[tuple[str, Any]] = []
        loader = _build_standard_loader(
            [{"id": 0, "tokens": 4}],
            DistributedDatasetConfig(seq_len=10, local_batch_size=2, buffer_size_multiplier=1.0),
            events,
        )

        self.assertEqual(list(loader), [])
        self.assertEqual(events, [("metadata", 0)])
        self.assertIsNone(loader.last_plan)
        with self.assertRaises(StopIteration):
            next(loader)

    def test_extreme_finite_buffer_multiplier_does_not_overflow_fill_targets(self) -> None:
        """Read-ahead target arithmetic must remain bounded for a finite multiplier."""
        events: list[tuple[str, Any]] = []
        loader = _build_standard_loader(
            [{"id": 0, "tokens": 8}],
            DistributedDatasetConfig(
                seq_len=8,
                local_batch_size=1,
                buffer_size_multiplier=1e308,
                max_buffered_samples=1,
            ),
            events,
        )

        self.assertEqual(next(loader), ({"sample_ids": (0,), "tokens": 8},))
        self.assertEqual(events, [("metadata", 0), ("pack", (0,)), ("collate", 1)])


if __name__ == "__main__":
    unittest.main()
