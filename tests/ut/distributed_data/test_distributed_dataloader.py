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
"""Standalone end-to-end tests for Dataset Reader to Data Constructor flow."""

import unittest
from threading import Event
from typing import Any
from unittest.mock import patch

from hyper_parallel.distributed_data import (
    DistributedDataLoader,
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
    default_collate_fn,
    default_pack_fn,
)
from tests.common.mark_utils import arg_mark


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _IterOnlyDataset:
    """Provide a finite online stream without index access."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        """Store samples yielded by ``__iter__``."""
        self._samples = samples

    def __len__(self) -> int:
        """Return the finite stream length."""
        return len(self._samples)

    def __iter__(self) -> Any:
        """Yield raw samples in source order."""
        return iter(self._samples)


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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_default_callbacks_return_planned_bins_of_raw_samples(self) -> None:
        """Feature: Default distributed data construction.
        Description: Omit packing and collation callbacks for raw samples.
        Expectation: Raw samples retain planned bin boundaries and capacity limits.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_online_iter_only_dataset_does_not_require_getitem(self) -> None:
        """Feature: Iterable online loading.
        Description: Build a distributed loader over a source without index access.
        Expectation: Planning consumes materialized iterator samples without rereading them.
        """
        samples = [{"id": index, "tokens": 4} for index in range(2)]

        def metadata_fn(sample: dict[str, Any]) -> SampleMetadata:
            """Expose the materialized sample's token footprint."""
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        loader = build_distributed_dataloader(
            _IterOnlyDataset(samples),
            _StandaloneMesh(),
            DistributedDatasetConfig(
                seq_len=8,
                local_batch_size=1,
                buffer_size_multiplier=1.0,
                dataset_already_sharded=True,
            ),
            metadata_fn=metadata_fn,
        )

        self.assertEqual(list(loader), [((samples[0], samples[1]),)])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_double_buffer_prepares_next_batch_while_trainer_consumes_current(self) -> None:
        """Feature: Host double buffering.
        Description: Hold the second read while the first batch is delivered.
        Expectation: The second transaction finishes before its foreground next call.
        """
        samples = [{"id": 0, "tokens": 8}, {"id": 1, "tokens": 8}]
        second_started = Event()
        release_second = Event()
        second_ready = Event()

        def metadata_fn(sample: dict[str, int]) -> SampleMetadata:
            """Hold the second sample so overlap is directly observable."""
            if sample["id"] == 1:
                second_started.set()
                if not release_second.wait(timeout=5):
                    raise ValueError("test timed out waiting to release the second sample")
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        def collate_fn(packed_sequences: list[tuple[dict[str, int], ...]]) -> tuple[Any, ...]:
            """Signal when background construction of the second batch completes."""
            if packed_sequences[0][0]["id"] == 1:
                second_ready.set()
            return tuple(packed_sequences)

        loader = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            DistributedDatasetConfig(
                seq_len=8,
                local_batch_size=1,
                buffer_size_multiplier=1.0,
                max_buffered_samples=1,
                double_buffer=True,
            ),
            metadata_fn=metadata_fn,
            collate_fn=collate_fn,
        )

        try:
            self.assertEqual(next(loader), ((samples[0],),))
            first_plan_id = loader.last_plan_id
            self.assertTrue(second_started.wait(timeout=2))
            release_second.set()
            self.assertTrue(second_ready.wait(timeout=2))
            self.assertEqual(loader.last_plan_id, first_plan_id)

            self.assertEqual(next(loader), ((samples[1],),))
            self.assertNotEqual(loader.last_plan_id, first_plan_id)
            with self.assertRaises(StopIteration):
                next(loader)
        finally:
            release_second.set()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_double_buffer_checkpoint_replays_discarded_prefetch(self) -> None:
        """Feature: Double-buffer checkpoint recovery.
        Description: Save state after speculative preparation has begun.
        Expectation: Uncommitted prefetched samples replay after restoration.
        """
        samples = [{"id": index, "tokens": 6 if index % 2 == 0 else 4} for index in range(6)]
        config = DistributedDatasetConfig(
            seq_len=10,
            local_batch_size=1,
            buffer_size_multiplier=2.0,
            double_buffer=True,
        )
        baseline = _build_standard_loader(samples, config, [])
        first_batch = next(baseline)
        checkpoint = baseline.state_dict()
        expected_batches, expected_plan_ids = _drain(baseline)

        resumed = _build_standard_loader(samples, config, [])
        resumed.load_state_dict(checkpoint)
        actual_batches, actual_plan_ids = _drain(resumed)

        self.assertEqual(first_batch, ({"sample_ids": (0, 1), "tokens": 10},))
        self.assertEqual(actual_batches, expected_batches)
        self.assertEqual(actual_plan_ids, expected_plan_ids)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_double_buffer_surfaces_background_reader_errors_on_next(self) -> None:
        """Feature: Double-buffer error propagation.
        Description: Inject a failure into the speculative background transaction.
        Expectation: The failure surfaces on the next call without corrupting the delivered batch.
        """
        samples = [{"id": 0, "tokens": 8}, {"id": 1, "tokens": 8}]

        def metadata_fn(sample: dict[str, int]) -> SampleMetadata:
            """Fail only in the second background transaction."""
            if sample["id"] == 1:
                raise ValueError("injected background metadata failure")
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        loader = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            DistributedDatasetConfig(
                seq_len=8,
                local_batch_size=1,
                buffer_size_multiplier=1.0,
                max_buffered_samples=1,
                double_buffer=True,
            ),
            metadata_fn=metadata_fn,
        )

        self.assertEqual(next(loader), ((samples[0],),))
        with self.assertRaisesRegex(RuntimeError, "injected background metadata failure"):
            next(loader)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_dynamically_packs_samples_then_collates_local_sequences(self) -> None:
        """Feature: Dynamic sample packing.
        Description: Apply explicit callbacks to each planned bin and local batch.
        Expectation: Samples pack to capacity before local sequences are collated.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_metadata_plans_before_direct_reads_and_skips_payload_a2a(self) -> None:
        """Feature: Metadata-guided direct sample reads.
        Description: Plan from metadata before materializing Dataset payloads.
        Expectation: Only assigned indices are read and payload A2A is skipped.
        """
        read_indices = []

        class _RecordingDataset:
            def __init__(self) -> None:
                """Store four deterministic raw samples."""
                self.samples = [
                    {"id": 0, "tokens": 6},
                    {"id": 1, "tokens": 4},
                    {"id": 2, "tokens": 6},
                    {"id": 3, "tokens": 4},
                ]

            def __len__(self) -> int:
                """Return the shared metadata index-space size."""
                return len(self.samples)

            def __getitem__(self, index: int) -> dict[str, int]:
                """Record each direct read performed after planning."""
                read_indices.append(index)
                return self.samples[index]

        dataset = _RecordingDataset()
        metadata = [
            SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])
            for sample in dataset.samples
        ]
        loader = build_distributed_dataloader(
            dataset,
            _StandaloneMesh(),
            DistributedDatasetConfig(seq_len=10, local_batch_size=2, buffer_size_multiplier=1.0),
            metadata=metadata,
        )
        self.assertEqual(read_indices, [])

        with (
                patch.object(loader._data_plane, "prepare_exchange", side_effect=AssertionError("unexpected A2A")),
                patch.object(loader._data_plane, "exchange_prepared", side_effect=AssertionError("unexpected A2A")),
        ):
            batch = next(loader)

        self.assertEqual(batch, ((dataset.samples[0], dataset.samples[1]), (dataset.samples[2], dataset.samples[3])))
        self.assertEqual(read_indices, [0, 1, 2, 3])
        planned_indices = [
            sample.key.dataset_index
            for packing_bin in loader.last_plan.constructor_for(0).bins
            for sample in packing_bin.samples
        ]
        self.assertEqual(read_indices, planned_indices)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_pre_sharded_metadata_reads_on_reader_then_uses_payload_exchange(self) -> None:
        """Feature: Pre-sharded metadata routing.
        Description: Plan from local metadata before loading selected payloads.
        Expectation: Selected local payloads pass through the payload exchange path.
        """
        samples = [{"id": 0, "tokens": 4}, {"id": 1, "tokens": 4}]
        metadata = [SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"]) for sample in samples]
        loader = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            DistributedDatasetConfig(
                seq_len=8,
                local_batch_size=1,
                buffer_size_multiplier=1.0,
                dataset_already_sharded=True,
            ),
            metadata=metadata,
        )

        with (
                patch.object(
                    loader._data_plane,
                    "prepare_exchange",
                    wraps=loader._data_plane.prepare_exchange,
                ) as prepare_exchange,
                patch.object(
                    loader._data_plane,
                    "exchange_prepared",
                    wraps=loader._data_plane.exchange_prepared,
                ) as exchange_prepared,
        ):
            batch = next(loader)

        self.assertEqual(batch, ((samples[0], samples[1]),))
        prepare_exchange.assert_called_once()
        exchange_prepared.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_lookahead_does_not_change_online_step_sample_ids(self) -> None:
        """Feature: Online step sample selection.
        Description: Compare step membership across different lookahead sizes.
        Expectation: Read-ahead buffers future payloads without admitting them early.
        """
        samples = [
            {"id": index, "tokens": tokens}
            for index, tokens in enumerate((6, 6, 4, 4, 2, 8))
        ]

        def metadata_fn(sample: dict[str, int]) -> SampleMetadata:
            """Expose deterministic streaming-packing footprints."""
            return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])

        def step_ids(multiplier: float) -> list[frozenset[int]]:
            """Drain one read-ahead configuration into per-step ID sets."""
            loader = build_distributed_dataloader(
                samples,
                _StandaloneMesh(),
                DistributedDatasetConfig(
                    seq_len=10,
                    local_batch_size=2,
                    buffer_size_multiplier=multiplier,
                ),
                metadata_fn=metadata_fn,
            )
            return [
                frozenset(sample["id"] for packing_bin in batch for sample in packing_bin)
                for batch in loader
            ]

        expected = [frozenset({0, 1, 2}), frozenset({3, 4, 5})]
        self.assertEqual(step_ids(1.0), expected)
        self.assertEqual(step_ids(3.0), expected)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_metadata_and_online_modes_select_the_same_canonical_steps(self) -> None:
        """Feature: Canonical step sample selection.
        Description: Compare ahead-of-fetch metadata planning with online planning.
        Expectation: Both modes produce identical stream step membership.
        """
        samples = [
            {"id": index, "tokens": tokens}
            for index, tokens in enumerate((6, 6, 4, 4, 2, 8))
        ]
        metadata = [
            SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])
            for sample in samples
        ]
        config = DistributedDatasetConfig(
            seq_len=10,
            local_batch_size=2,
            buffer_size_multiplier=3.0,
        )
        online = build_distributed_dataloader(
            samples,
            _StandaloneMesh(),
            config,
            metadata_fn=lambda sample: SampleMetadata(
                pack_tokens=sample["tokens"],
                sample_id=sample["id"],
            ),
        )
        metadata_loader = build_distributed_dataloader(samples, _StandaloneMesh(), config, metadata=metadata)

        def selected_step_ids(loader: DistributedDataLoader) -> list[set[int]]:
            """Return frozen sample IDs from every delivered packing plan."""
            result = []
            for _ in loader:
                result.append({
                    sample.metadata.sample_id
                    for constructor in loader.last_plan.constructors
                    for packing_bin in constructor.bins
                    for sample in packing_bin.samples
                })
            return result

        self.assertEqual(selected_step_ids(online), [{0, 1, 2}, {3, 4, 5}])
        self.assertEqual(selected_step_ids(metadata_loader), [{0, 1, 2}, {3, 4, 5}])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_metadata_checkpoint_replays_metadata_buffer_without_payloads(self) -> None:
        """Feature: Metadata checkpoint recovery.
        Description: Restore a metadata buffer without storing raw payloads.
        Expectation: Future plans remain stable and only selected indices are reread.
        """
        samples = [{"id": index, "tokens": 6 if index % 2 == 0 else 4} for index in range(6)]
        metadata = [
            SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])
            for sample in samples
        ]
        config = DistributedDatasetConfig(
            seq_len=10,
            local_batch_size=1,
            buffer_size_multiplier=2.0,
            shuffle=True,
        )

        baseline = build_distributed_dataloader(samples, _StandaloneMesh(), config, metadata=metadata)
        baseline.set_epoch(3)
        first_batch = next(baseline)
        self.assertEqual(len(first_batch), 1)
        checkpoint = baseline.state_dict()
        self.assertNotIn("sidecar_reader", checkpoint)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["metadata_reader"]["epoch"], 3)
        self.assertEqual(checkpoint["direct_sample_loader"]["epoch"], 3)
        buffered_metadata = checkpoint["metadata_reader"]["buffer"]
        selected_keys = set(baseline.last_plan.selected_keys)
        self.assertTrue(buffered_metadata)
        self.assertTrue(selected_keys.isdisjoint(item.key for item in buffered_metadata))
        self.assertEqual(min(item.global_sample_position for item in buffered_metadata), len(selected_keys))
        expected_batches, expected_plan_ids = _drain(baseline)

        for reader_key in ("metadata_reader", "sidecar_reader"):
            with self.subTest(reader_key=reader_key):
                restored_state = dict(checkpoint)
                restored_state[reader_key] = restored_state.pop("metadata_reader")
                resumed = build_distributed_dataloader(samples, _StandaloneMesh(), config, metadata=metadata)
                resumed.load_state_dict(restored_state)
                actual_batches, actual_plan_ids = _drain(resumed)

                self.assertEqual(actual_batches, expected_batches)
                self.assertEqual(actual_plan_ids, expected_plan_ids)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_builder_rejects_ambiguous_or_misaligned_metadata(self) -> None:
        """Feature: Builder metadata validation.
        Description: Provide ambiguous or length-misaligned metadata inputs.
        Expectation: Online and metadata modes remain exclusive and index-aligned.
        """
        samples = [{"id": 0, "tokens": 4}]
        metadata = [SampleMetadata(pack_tokens=4, sample_id=0)]
        config = DistributedDatasetConfig(seq_len=4, local_batch_size=1)

        with self.assertRaisesRegex(ValueError, "either online metadata_fn or metadata"):
            build_distributed_dataloader(
                samples,
                _StandaloneMesh(),
                config,
                metadata_fn=lambda sample: metadata[0],
                metadata=metadata,
            )
        with self.assertRaisesRegex(ValueError, "must provide metadata"):
            build_distributed_dataloader(samples, _StandaloneMesh(), config)
        with self.assertRaisesRegex(ValueError, "does not match Dataset length"):
            build_distributed_dataloader(samples, _StandaloneMesh(), config, metadata=metadata * 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_prepacked_sample_can_be_one_complete_local_batch(self) -> None:
        """Feature: Prepacked sample construction.
        Description: Treat one full-length raw sample as a complete local batch.
        Expectation: The sample remains in a singleton constructor bin.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_resume_matches_uninterrupted_batches_and_plan_ids(self) -> None:
        """Feature: Distributed loader checkpoint recovery.
        Description: Resume buffered iteration and compare with uninterrupted loading.
        Expectation: Batches and plan IDs continue without duplication or drift.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_fingerprint_canonicalizes_default_callback_mode(self) -> None:
        """Feature: Constructor callback checkpoint identity.
        Description: Restore state with omitted, explicit-default, and custom callbacks.
        Expectation: Equivalent defaults match while custom construction is incompatible.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_eof_drops_a_tail_that_cannot_fill_every_local_bin(self) -> None:
        """Feature: End-of-stream tail handling.
        Description: Exhaust the source before every local bin can be populated.
        Expectation: The default drop-last contract stops before construction callbacks.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_extreme_finite_buffer_multiplier_does_not_overflow_fill_targets(self) -> None:
        """Feature: Bounded read-ahead target arithmetic.
        Description: Use an extreme finite multiplier with a sample-count cap.
        Expectation: Fill targets remain bounded and the batch is delivered.
        """
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
