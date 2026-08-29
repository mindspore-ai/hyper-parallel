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
"""Tests for rank-strided Source Loader materialization and recovery."""

import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

from torch.utils.data._utils.pin_memory import (  # pylint: disable=forbidden-backend-import
    pin_memory as torch_pin_memory,
)

from hyper_parallel.distributed_data.schema import SampleKey, SampleMetadata
from hyper_parallel.distributed_data.source_loader import SourceLoader, _IndexedPayload


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


def _source_loader(
        dataset: _RecordingDataset,
        metadata_fn: Callable[[dict[str, int]], SampleMetadata],
        *,
        source_rank: int = 4,
        source_lane: int = 0,
        source_lane_count: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
) -> SourceLoader:
    return SourceLoader(
        dataset,
        metadata_fn,
        source_rank=source_rank,
        source_lane=source_lane,
        source_lane_count=source_lane_count,
        seq_len=16,
        shuffle=False,
        seed=17,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=None,
        persistent_workers=persistent_workers,
    )


class TestSourceLoader(unittest.TestCase):
    """Verify samples are materialized once, then exposed as lightweight metadata."""

    def test_strides_dataset_indices_and_derives_metadata_after_getitem(self) -> None:
        """A source lane reads its global stride and invokes metadata afterward."""
        events: list[tuple[str, int]] = []
        dataset = _RecordingDataset(list(range(1, 9)), events)
        source = _source_loader(
            dataset,
            _metadata_callback(events),
            source_rank=6,
            source_lane=1,
            source_lane_count=3,
        )

        error = source.fill(min_samples=4, min_tokens=100, max_samples=10)

        self.assertIsNone(error)
        self.assertTrue(source.exhausted)
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
        self.assertEqual([item.key for item in source.metadata()], [SampleKey(6, 1), SampleKey(6, 4), SampleKey(6, 7)])
        self.assertEqual([item.metadata.pack_tokens for item in source.metadata()], [2, 5, 8])
        self.assertEqual(source.effective_buffer_tokens, 15)

    def test_selected_payloads_are_non_destructive_until_commit(self) -> None:
        """Routing reads payloads transactionally and commit removes only selected keys."""
        events: list[tuple[str, int]] = []
        source = _source_loader(_RecordingDataset([2, 3, 4], events), _metadata_callback(events), source_rank=2)
        self.assertIsNone(source.fill(min_samples=3, min_tokens=9, max_samples=3))
        selected_key = SampleKey(2, 1)

        first = source.selected_payloads({selected_key})
        second = source.selected_payloads({selected_key})

        self.assertEqual(first, second)
        self.assertEqual(first[0][1], {"index": 1, "tokens": 3})
        self.assertEqual(source.buffer_size, 3)
        source.commit({selected_key})
        self.assertEqual([item.key for item in source.metadata()], [SampleKey(2, 0), SampleKey(2, 2)])

    def test_checkpoint_restores_buffer_and_continues_without_replaying_indices(self) -> None:
        """Restoration preserves read-ahead payloads and resumes at the next lane ordinal."""
        original_events: list[tuple[str, int]] = []
        original = _source_loader(
            _RecordingDataset([1] * 8, original_events),
            _metadata_callback(original_events),
            source_rank=5,
            source_lane=0,
            source_lane_count=2,
        )
        self.assertIsNone(original.fill(min_samples=2, min_tokens=2, max_samples=2))
        state = original.state_dict()

        restored_events: list[tuple[str, int]] = []
        restored = _source_loader(
            _RecordingDataset([1] * 8, restored_events),
            _metadata_callback(restored_events),
            source_rank=5,
            source_lane=0,
            source_lane_count=2,
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

    def test_metadata_callback_failure_is_returned_with_source_context(self) -> None:
        """A bad metadata callback is surfaced for collective error propagation."""
        events: list[tuple[str, int]] = []
        dataset = _RecordingDataset([3], events)

        def invalid_metadata(sample: dict[str, int]) -> dict[str, int]:
            """Return the wrong callback type after recording invocation."""
            events.append(("metadata", sample["index"]))
            return sample

        source = _source_loader(dataset, invalid_metadata, source_rank=7)
        error = source.fill(min_samples=1, min_tokens=1, max_samples=1)

        self.assertIn("Source Loader rank 7 failed", error)
        self.assertIn("metadata_fn must return SampleMetadata", error)
        self.assertEqual(events, [("getitem", 0), ("metadata", 0)])
        self.assertEqual(source.buffer_size, 0)

        repeated_error = source.fill(min_samples=1, min_tokens=1, max_samples=1)
        self.assertEqual(repeated_error, error)
        self.assertEqual(events, [("getitem", 0), ("metadata", 0)])

    def test_indexed_payload_participates_in_recursive_pin_memory_walk(self) -> None:
        """The index wrapper must preserve identity while recursively pinning its payload."""
        pinnable = _PinnablePayload(9)
        indexed = _IndexedPayload(dataset_index=17, payload={"nested": [pinnable]})

        pinned = torch_pin_memory(indexed)

        self.assertIsInstance(pinned, _IndexedPayload)
        self.assertEqual(pinned.dataset_index, 17)
        self.assertEqual(pinned.payload, {"nested": ["pinned-9"]})
        self.assertEqual(pinnable.pin_calls, 1)

    def test_iterator_rebuild_reuses_data_loader_for_persistent_workers(self) -> None:
        """Checkpoint restoration must call iter again without replacing the worker-owning DataLoader."""
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
        with patch("hyper_parallel.distributed_data.source_loader.DataLoader", _FakeDataLoader):
            source = _source_loader(
                _RecordingDataset([2, 3, 4], events),
                _metadata_callback(events),
                source_rank=3,
                num_workers=1,
                persistent_workers=True,
            )
            self.assertIsNone(source.fill(min_samples=1, min_tokens=1, max_samples=1))
            state = source.state_dict()
            source.load_state_dict(state)
            self.assertIsNone(source.fill(min_samples=2, min_tokens=2, max_samples=2))

        self.assertEqual(len(created_loaders), 1)
        self.assertEqual(created_loaders[0].iter_calls, 2)
        self.assertEqual(created_loaders[0].worker_options["num_workers"], 1)
        self.assertTrue(created_loaders[0].worker_options["persistent_workers"])
        self.assertEqual([item.key for item in source.metadata()], [SampleKey(3, 0), SampleKey(3, 1)])
        self.assertEqual(
            events,
            [("getitem", 0), ("metadata", 0), ("getitem", 1), ("metadata", 1)],
        )


if __name__ == "__main__":
    unittest.main()
