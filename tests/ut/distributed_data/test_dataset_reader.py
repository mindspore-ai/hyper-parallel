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
"""Tests for rank-strided Dataset Reader materialization and recovery."""

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from torch.utils.data._utils.pin_memory import (  # pylint: disable=forbidden-backend-import
    pin_memory as torch_pin_memory,
)
from torch.utils.data import Dataset  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.schema import SampleKey, SampleMetadata
from hyper_parallel.distributed_data.sidecar import SidecarMetadataReader
from hyper_parallel.distributed_data.dataset_reader import DatasetReader, _IndexedPayload
from tests.common.mark_utils import arg_mark


class _RecordingDataset:
    def __init__(self, tokens: list[int], events: list[tuple[str, int]]) -> None:
        """Store token counts and a shared callback-order log."""
        self._tokens = tokens
        self._events = events

    def __len__(self) -> int:
        """Return the mapping Dataset size."""
        return len(self._tokens)

    def __getitem__(self, index: int) -> dict[str, int]:
        """Record materialization before returning one sample."""
        self._events.append(("getitem", index))
        return {"index": index, "tokens": self._tokens[index]}


class _IterOnlyDataset(Dataset):
    """Match a Dataset subclass that implements ``__iter__`` but not ``__getitem__``."""

    def __init__(self, tokens: list[int], events: list[tuple[str, int]]) -> None:
        """Store the finite local stream and materialization log."""
        self._tokens = tokens
        self._events = events

    def __len__(self) -> int:
        """Return the local stream size."""
        return len(self._tokens)

    def __iter__(self) -> Any:
        """Yield every local sample without supporting index access."""
        for index, tokens in enumerate(self._tokens):
            self._events.append(("iter", index))
            yield {"index": index, "tokens": tokens}


class _PinnablePayload:
    def __init__(self, value: int) -> None:
        """Store a value and expose whether the walker visited it."""
        self.value = value
        self.pin_calls = 0

    def pin_memory(self) -> str:
        """Record recursive pinning without requiring accelerator hardware."""
        self.pin_calls += 1
        return f"pinned-{self.value}"


def _metadata_callback(events: list[tuple[str, int]]) -> Callable[[dict[str, int]], SampleMetadata]:
    def metadata_fn(sample: dict[str, int]) -> SampleMetadata:
        """Record metadata derivation and return the sample token cost."""
        events.append(("metadata", sample["index"]))
        return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["index"])

    return metadata_fn


def _dataset_reader(
        dataset: Any,
        metadata_fn: Callable[[dict[str, int]], SampleMetadata],
        *,
        reader_rank: int = 4,
        reader_idx: int = 0,
        reader_count: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        dataset_already_sharded: bool = False,
) -> DatasetReader:
    return DatasetReader(
        dataset,
        metadata_fn,
        reader_rank=reader_rank,
        reader_idx=reader_idx,
        reader_count=reader_count,
        seq_len=16,
        shuffle=False,
        seed=17,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=None,
        persistent_workers=persistent_workers,
        dataset_already_sharded=dataset_already_sharded,
    )


class TestDatasetReader(unittest.TestCase):
    """Verify samples are materialized once, then exposed as lightweight metadata."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_strides_dataset_indices_and_derives_metadata_after_getitem(self) -> None:
        """Feature: Mapping Dataset Reader stride.
        Description: Materialize one reader partition and derive metadata afterward.
        Expectation: Only the reader's global stride is read in source order.
        """
        events: list[tuple[str, int]] = []
        dataset = _RecordingDataset(list(range(1, 9)), events)
        reader = _dataset_reader(
            dataset,
            _metadata_callback(events),
            reader_rank=6,
            reader_idx=1,
            reader_count=3,
        )

        error = reader.fill(min_samples=4, min_tokens=100, max_samples=10)

        self.assertIsNone(error)
        self.assertTrue(reader.exhausted)
        self.assertEqual(
            events,
            [
                ("getitem", 1),
                ("metadata", 1),
                ("getitem", 4),
                ("metadata", 4),
                ("getitem", 7),
                ("metadata", 7),
            ],
        )
        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(6, 1), SampleKey(6, 4), SampleKey(6, 7)])
        self.assertEqual([item.global_sample_position for item in reader.metadata()], [1, 4, 7])
        self.assertEqual([item.metadata.pack_tokens for item in reader.metadata()], [2, 5, 8])
        self.assertEqual(reader.effective_buffer_tokens, 15)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_pre_sharded_mapping_dataset_disables_second_reader_stride(self) -> None:
        """Feature: Pre-sharded mapping Dataset Reader.
        Description: Read a local mapping Dataset with secondary striding disabled.
        Expectation: Every local sample is exposed to its Dataset Reader.
        """
        events: list[tuple[str, int]] = []
        reader = _dataset_reader(
            _RecordingDataset([1, 2, 3], events),
            _metadata_callback(events),
            reader_rank=6,
            reader_idx=1,
            reader_count=3,
            dataset_already_sharded=True,
        )

        self.assertIsNone(reader.fill(min_samples=3, min_tokens=6, max_samples=3))

        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(6, 0), SampleKey(6, 1), SampleKey(6, 2)])
        self.assertEqual([item.global_sample_position for item in reader.metadata()], [1, 4, 7])
        self.assertEqual(
            events,
            [
                ("getitem", 0),
                ("metadata", 0),
                ("getitem", 1),
                ("metadata", 1),
                ("getitem", 2),
                ("metadata", 2),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_iter_only_dataset_is_consumed_without_getitem(self) -> None:
        """Feature: Iterable online Dataset Reader.
        Description: Consume an already-sharded iterator without index access.
        Expectation: Samples and metadata are produced directly from iteration.
        """
        events: list[tuple[str, int]] = []
        reader = _dataset_reader(
            _IterOnlyDataset([1, 2, 3], events),
            _metadata_callback(events),
            reader_rank=6,
            reader_idx=1,
            reader_count=3,
            dataset_already_sharded=True,
        )

        self.assertIsNone(reader.fill(min_samples=3, min_tokens=6, max_samples=3))

        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(6, 0), SampleKey(6, 1), SampleKey(6, 2)])
        self.assertEqual([item.global_sample_position for item in reader.metadata()], [1, 4, 7])
        self.assertEqual(
            events,
            [
                ("iter", 0),
                ("metadata", 0),
                ("iter", 1),
                ("metadata", 1),
                ("iter", 2),
                ("metadata", 2),
            ],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_unsharded_iter_only_dataset_applies_reader_stride(self) -> None:
        """Feature: Iterable Dataset Reader stride.
        Description: Consume a shared iterator in the default unsharded mode.
        Expectation: The Dataset Reader applies its configured global stride.
        """
        events: list[tuple[str, int]] = []
        reader = _dataset_reader(
            _IterOnlyDataset([1] * 8, events),
            _metadata_callback(events),
            reader_rank=6,
            reader_idx=1,
            reader_count=3,
        )

        self.assertIsNone(reader.fill(min_samples=4, min_tokens=100, max_samples=10))

        self.assertEqual([item.metadata.sample_id for item in reader.metadata()], [1, 4, 7])
        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(6, 0), SampleKey(6, 1), SampleKey(6, 2)])
        self.assertEqual([item.global_sample_position for item in reader.metadata()], [1, 4, 7])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_iter_only_checkpoint_replays_source_to_next_local_position(self) -> None:
        """Feature: Iterable Dataset Reader recovery.
        Description: Restore buffered payloads and rebuild the source iterator.
        Expectation: Iteration advances to the next unread local position.
        """
        original_events: list[tuple[str, int]] = []
        original = _dataset_reader(
            _IterOnlyDataset([1] * 5, original_events),
            _metadata_callback(original_events),
            dataset_already_sharded=True,
        )
        self.assertIsNone(original.fill(min_samples=2, min_tokens=2, max_samples=2))
        state = original.state_dict()

        restored_events: list[tuple[str, int]] = []
        restored = _dataset_reader(
            _IterOnlyDataset([1] * 5, restored_events),
            _metadata_callback(restored_events),
            dataset_already_sharded=True,
        )
        restored.load_state_dict(state)
        self.assertIsNone(restored.fill(min_samples=4, min_tokens=4, max_samples=4))

        self.assertEqual([item.metadata.sample_id for item in restored.metadata()], [0, 1, 2, 3])
        self.assertEqual(
            restored_events,
            [("iter", 0), ("iter", 1), ("iter", 2), ("metadata", 2), ("iter", 3), ("metadata", 3)],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_selected_payloads_are_non_destructive_until_commit(self) -> None:
        """Feature: Transactional Dataset Reader routing.
        Description: Read selected payloads repeatedly before committing their keys.
        Expectation: Reads are non-destructive and commit removes only selected samples.
        """
        events: list[tuple[str, int]] = []
        reader = _dataset_reader(_RecordingDataset([2, 3, 4], events), _metadata_callback(events), reader_rank=2)
        self.assertIsNone(reader.fill(min_samples=3, min_tokens=9, max_samples=3))
        selected_key = SampleKey(2, 1)

        first = reader.selected_payloads({selected_key})
        second = reader.selected_payloads({selected_key})

        self.assertEqual(first, second)
        self.assertEqual(first[0][1], {"index": 1, "tokens": 3})
        self.assertEqual(reader.buffer_size, 3)
        reader.commit({selected_key})
        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(2, 0), SampleKey(2, 2)])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_restores_buffer_and_continues_without_replaying_indices(self) -> None:
        """Feature: Mapping Dataset Reader recovery.
        Description: Restore buffered payloads and continue filling the reader.
        Expectation: Reading resumes at the next ordinal without replaying indices.
        """
        original_events: list[tuple[str, int]] = []
        original = _dataset_reader(
            _RecordingDataset([1] * 8, original_events),
            _metadata_callback(original_events),
            reader_rank=5,
            reader_idx=0,
            reader_count=2,
        )
        self.assertIsNone(original.fill(min_samples=2, min_tokens=2, max_samples=2))
        state = original.state_dict()

        restored_events: list[tuple[str, int]] = []
        restored = _dataset_reader(
            _RecordingDataset([1] * 8, restored_events),
            _metadata_callback(restored_events),
            reader_rank=5,
            reader_idx=0,
            reader_count=2,
        )
        restored.load_state_dict(state)

        self.assertEqual(restored.metadata(), original.metadata())
        self.assertEqual(restored_events, [])
        self.assertIsNone(original.fill(min_samples=4, min_tokens=4, max_samples=4))
        self.assertIsNone(restored.fill(min_samples=4, min_tokens=4, max_samples=4))
        self.assertEqual(restored.metadata(), original.metadata())
        self.assertEqual(
            restored_events,
            [("getitem", 4), ("metadata", 4), ("getitem", 6), ("metadata", 6)],
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_metadata_callback_failure_is_returned_with_reader_context(self) -> None:
        """Feature: Metadata callback error handling.
        Description: Return invalid metadata from the online callback.
        Expectation: The error includes Dataset Reader context for collective propagation.
        """
        events: list[tuple[str, int]] = []
        dataset = _RecordingDataset([3], events)

        def invalid_metadata(sample: dict[str, int]) -> dict[str, int]:
            """Return the wrong callback type after recording invocation."""
            events.append(("metadata", sample["index"]))
            return sample

        reader = _dataset_reader(dataset, invalid_metadata, reader_rank=7)
        error = reader.fill(min_samples=1, min_tokens=1, max_samples=1)

        self.assertIn("Dataset Reader rank 7 failed", error)
        self.assertIn("metadata_fn must return SampleMetadata", error)
        self.assertEqual(events, [("getitem", 0), ("metadata", 0)])
        self.assertEqual(reader.buffer_size, 0)

        repeated_error = reader.fill(min_samples=1, min_tokens=1, max_samples=1)
        self.assertEqual(repeated_error, error)
        self.assertEqual(events, [("getitem", 0), ("metadata", 0)])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_indexed_payload_participates_in_recursive_pin_memory_walk(self) -> None:
        """Feature: Indexed payload memory pinning.
        Description: Apply recursive pinning to an index-preserving payload wrapper.
        Expectation: Recursive pinning reaches the payload while preserving its index.
        """
        pinnable = _PinnablePayload(9)
        indexed = _IndexedPayload(dataset_index=17, payload={"nested": [pinnable]})

        pinned = torch_pin_memory(indexed)

        self.assertIsInstance(pinned, _IndexedPayload)
        self.assertEqual(pinned.dataset_index, 17)
        self.assertEqual(pinned.payload, {"nested": ["pinned-9"]})
        self.assertEqual(pinnable.pin_calls, 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_iterator_rebuild_reuses_data_loader_for_persistent_workers(self) -> None:
        """Feature: Persistent Dataset Reader workers.
        Description: Rebuild iterators during checkpoint restoration.
        Expectation: The worker-owning native DataLoader is reused.
        """
        created_loaders = []

        class _FakeDataLoader:
            def __init__(
                    self,
                    dataset: Any,
                    *,
                    collate_fn: Callable[[Any], Any],
                    **worker_options: Any,
            ) -> None:
                """Capture construction options and provide deterministic iterator generations."""
                self.dataset = dataset
                self.collate_fn = collate_fn
                self.worker_options = worker_options
                self.iter_calls = 0
                created_loaders.append(self)

            def __iter__(self) -> Any:
                """Return one successive sample for each iterator rebuild."""
                dataset_index = self.iter_calls
                self.iter_calls += 1
                return iter((self.collate_fn(self.dataset[dataset_index]),))

        events: list[tuple[str, int]] = []
        with patch("hyper_parallel.distributed_data.dataset_reader.DataLoader", _FakeDataLoader):
            reader = _dataset_reader(
                _RecordingDataset([2, 3, 4], events),
                _metadata_callback(events),
                reader_rank=3,
                num_workers=1,
                persistent_workers=True,
            )
            self.assertIsNone(reader.fill(min_samples=1, min_tokens=1, max_samples=1))
            state = reader.state_dict()
            reader.load_state_dict(state)
            self.assertIsNone(reader.fill(min_samples=2, min_tokens=2, max_samples=2))

        self.assertEqual(len(created_loaders), 1)
        self.assertEqual(created_loaders[0].iter_calls, 2)
        self.assertEqual(created_loaders[0].worker_options["num_workers"], 1)
        self.assertTrue(created_loaders[0].worker_options["persistent_workers"])
        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(3, 0), SampleKey(3, 1)])
        self.assertEqual(
            events,
            [("getitem", 0), ("metadata", 0), ("getitem", 1), ("metadata", 1)],
        )


class TestSidecarMetadataReader(unittest.TestCase):
    """Verify sidecar reader partitions never materialize Dataset payloads."""

    @staticmethod
    def _reader(
            metadata: list[SampleMetadata],
            reader_idx: int,
            *,
            dataset_already_sharded: bool = False,
    ) -> SidecarMetadataReader:
        return SidecarMetadataReader(
            metadata,
            reader_rank=reader_idx + 4,
            reader_idx=reader_idx,
            reader_count=2,
            seq_len=16,
            shuffle=False,
            seed=23,
            dataset_already_sharded=dataset_already_sharded,
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_reader_idxs_cover_sidecar_indices_without_payload_reads(self) -> None:
        """Feature: Sidecar metadata partitioning.
        Description: Read sidecar entries through complementary Dataset Reader strides.
        Expectation: The strides form a complete disjoint index partition without payload reads.
        """
        metadata = [SampleMetadata(pack_tokens=index + 1, sample_id=f"sample-{index}") for index in range(6)]
        first = self._reader(metadata, 0)
        second = self._reader(metadata, 1)

        self.assertIsNone(first.fill(min_samples=6, min_tokens=100, max_samples=6))
        self.assertIsNone(second.fill(min_samples=6, min_tokens=100, max_samples=6))

        self.assertEqual([item.key.dataset_index for item in first.metadata()], [0, 2, 4])
        self.assertEqual([item.key.dataset_index for item in second.metadata()], [1, 3, 5])
        self.assertEqual([item.global_sample_position for item in first.metadata()], [0, 2, 4])
        self.assertEqual([item.global_sample_position for item in second.metadata()], [1, 3, 5])
        self.assertEqual(
            {item.key.dataset_index for item in first.metadata() + second.metadata()},
            set(range(6)),
        )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_checkpoint_preserves_unselected_metadata_buffer(self) -> None:
        """Feature: Sidecar metadata recovery.
        Description: Commit one candidate and restore the remaining reader buffer.
        Expectation: Skipped candidates and the reader cursor are preserved.
        """
        metadata = [SampleMetadata(pack_tokens=1, sample_id=index) for index in range(8)]
        original = self._reader(metadata, 0)
        self.assertIsNone(original.fill(min_samples=3, min_tokens=3, max_samples=3))
        selected_key = original.metadata()[1].key
        original.commit({selected_key})
        state = original.state_dict()

        restored = self._reader(metadata, 0)
        restored.load_state_dict(state)
        self.assertEqual(restored.metadata(), original.metadata())

        self.assertIsNone(original.fill(min_samples=4, min_tokens=4, max_samples=4))
        self.assertIsNone(restored.fill(min_samples=4, min_tokens=4, max_samples=4))
        self.assertEqual(restored.metadata(), original.metadata())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_pre_sharded_reader_exposes_every_local_sidecar_entry(self) -> None:
        """Feature: Pre-sharded sidecar metadata.
        Description: Read a local sidecar with secondary striding disabled.
        Expectation: Every local metadata entry is exposed exactly once.
        """
        metadata = [SampleMetadata(pack_tokens=index + 1, sample_id=index) for index in range(3)]
        reader = self._reader(metadata, 1, dataset_already_sharded=True)

        self.assertIsNone(reader.fill(min_samples=3, min_tokens=6, max_samples=3))

        self.assertEqual([item.key for item in reader.metadata()], [SampleKey(5, 0), SampleKey(5, 1), SampleKey(5, 2)])
        self.assertEqual([item.global_sample_position for item in reader.metadata()], [1, 3, 5])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_active_sidecar_buffer_rejects_epoch_change(self) -> None:
        """Feature: Sidecar epoch transitions.
        Description: Change epoch while the metadata planning buffer is active.
        Expectation: The reader rejects silently discarding in-progress candidates.
        """
        reader = self._reader([SampleMetadata(pack_tokens=1, sample_id=index) for index in range(4)], 0)
        self.assertIsNone(reader.fill(min_samples=1, min_tokens=1, max_samples=1))

        with self.assertRaisesRegex(ValueError, "active sidecar metadata buffer"):
            reader.set_epoch(1)


class TestSharedReaderState(unittest.TestCase):
    """Protect buffer transactions and shared shuffle/stride semantics."""

    @staticmethod
    def _readers(sharded: bool) -> tuple[DatasetReader, SidecarMetadataReader]:
        """Create online and metadata readers over the same logical sample order."""
        events: list[tuple[str, int]] = []
        tokens = list(range(1, 10))
        options = {
            "reader_rank": 5, "reader_idx": 1, "reader_count": 2, "seq_len": 16,
            "shuffle": True, "seed": 17, "dataset_already_sharded": sharded,
        }
        online = DatasetReader(
            _RecordingDataset(tokens, events), _metadata_callback(events),
            num_workers=0, pin_memory=False, prefetch_factor=None, persistent_workers=False, **options,
        )
        sidecar = SidecarMetadataReader(
            [SampleMetadata(pack_tokens=value, sample_id=index) for index, value in enumerate(tokens)], **options,
        )
        return online, sidecar

    def test_failed_restore_and_commit_leave_buffer_unchanged(self) -> None:
        """Both formats reject corrupt state and missing keys before mutating live data."""
        for reader in self._readers(False):
            with self.subTest(reader=type(reader).__name__):
                self.assertIsNone(reader.fill(min_samples=2, min_tokens=1, max_samples=2))
                baseline = reader.state_dict()
                corruptions = (
                    {"version": 0}, {"reader_idx": 0}, {"epoch": True}, {"next_ordinal": -1},
                    {"exhausted": 1}, {"error": ""}, {"buffer": ()}, {"buffer": [object()]},
                    {"buffer": baseline["buffer"] * 2},
                )
                for changes in corruptions:
                    with self.subTest(field=tuple(changes)), self.assertRaises(ValueError):
                        reader.load_state_dict({**baseline, **changes})
                    self.assertEqual(reader.state_dict(), baseline)
                keys = {reader.metadata()[0].key, SampleKey(5, 100)}
                with self.assertRaisesRegex(ValueError, "Cannot commit missing"):
                    reader.commit(keys)
                self.assertEqual(reader.state_dict(), baseline)
                baseline["buffer"].clear()
                self.assertEqual(reader.buffer_size, 2)

    def test_shuffle_and_resume_match_between_reader_modes(self) -> None:
        """Sharded and unsharded streams retain identical metadata order across recovery."""
        for sharded in (False, True):
            with self.subTest(sharded=sharded):
                readers = self._readers(sharded)
                restored = self._readers(sharded)
                for reader, resumed in zip(readers, restored):
                    reader.set_epoch(2)
                    self.assertIsNone(reader.fill(min_samples=2, min_tokens=1, max_samples=2))
                    reader.commit({reader.metadata()[0].key})
                    resumed.load_state_dict(reader.state_dict())
                    self.assertIsNone(reader.fill(min_samples=9, min_tokens=100, max_samples=9))
                    self.assertIsNone(resumed.fill(min_samples=9, min_tokens=100, max_samples=9))
                    self.assertEqual(resumed.state_dict(), reader.state_dict())
                self.assertEqual(readers[0].metadata(), readers[1].metadata())


if __name__ == "__main__":
    unittest.main()
