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
"""Unit tests for HF streaming dataset helpers."""
# pylint: disable=protected-access
import os
import sys
import types
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.data.hf import StreamingTokenizedDataset, build_hf_datasets


class _FakeHFIterable:
    """Tiny iterable dataset double with HF-like streaming methods."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(rows[0].keys()) if rows else []
        self.epoch = 0
        self.loaded_state = None
        self.shuffle_seed = None
        self.shuffle_buffer = None

    def map(self, fn, batched=False, remove_columns=None, desc=None):
        """Apply a batched map function and return a cloned iterable."""
        del remove_columns, desc
        if not batched:
            raise AssertionError("Streaming builder is expected to use batched=True")
        batch = {key: [row[key] for row in self.rows] for key in self.column_names}
        output = fn(batch)
        keys = list(output.keys())
        mapped_rows = []
        for idx in range(len(output[keys[0]])):
            mapped_rows.append({key: output[key][idx] for key in keys})
        return _FakeHFIterable(mapped_rows)

    def filter(self, fn):
        return _FakeHFIterable([row for row in self.rows if fn(row)])

    def shuffle(self, seed, buffer_size):
        clone = _FakeHFIterable(self.rows)
        clone.shuffle_seed = seed
        clone.shuffle_buffer = buffer_size
        return clone

    def with_format(self, fmt):
        if fmt != "torch":
            raise AssertionError(f"Unexpected format request: {fmt}")
        return self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state):
        self.loaded_state = dict(state)
        self.epoch = int(state["epoch"])

    def __iter__(self):
        rows = list(self.rows)
        if self.epoch % 2 == 1:
            rows.reverse()
        return iter(rows)


class _FakeSplitIterable:
    """Simple DP-rank sharding wrapper."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.column_names = getattr(dataset, "column_names", [])

    def with_format(self, fmt):
        self.dataset = self.dataset.with_format(fmt)
        return self

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state):
        self.dataset.load_state_dict(state)

    def __iter__(self):
        for idx, row in enumerate(self.dataset):
            if idx % self.world_size == self.rank:
                yield row


def _build_fake_datasets_modules(rows):
    """Create fake ``datasets`` modules for the lazy imports in ``hf.py``."""
    datasets_module = types.ModuleType("datasets")

    def _load_dataset(*args, **kwargs):
        """Return a fixed iterable dataset for builder tests."""
        del args, kwargs
        return _FakeHFIterable(rows)

    def _split_dataset_by_node(dataset, rank, world_size):
        """Return a simple DP-rank shard wrapper."""
        return _FakeSplitIterable(dataset, rank, world_size)

    datasets_module.load_dataset = _load_dataset
    distributed_module = types.ModuleType("datasets.distributed")
    distributed_module.split_dataset_by_node = _split_dataset_by_node
    return datasets_module, distributed_module


class _BaseState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class _BaseStub:
    def __init__(self, max_steps):
        self.state = _BaseState(max_steps)


def _make_args(train_size=5):
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            train_path="dummy/path",
            subset=None,
            train_size=train_size,
            streaming=True,
            shuffle=True,
        ),
        train=types.SimpleNamespace(
            seed=17,
            max_steps=99,
            num_train_epochs=2,
            global_batch_size=2,
        ),
    )


class TestHFStreamingBuilder(unittest.TestCase):
    """Coverage for the new HF streaming builder path."""

    def test_streaming_builder_self_shards_and_clips_steps(self):
        """Verify streaming builder self-shards rows and updates step budget."""
        rows = [
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
            {"text": "d"},
            {"text": "e"},
            {"text": "f"},
            {"text": "g"},
        ]
        datasets_module, distributed_module = _build_fake_datasets_modules(rows)
        args = _make_args(train_size=5)
        base = _BaseStub(max_steps=99)

        def _transform(batch):
            return {
                "input_ids": [[ord(text)] for text in batch["text"]],
                "labels": [[ord(text)] for text in batch["text"]],
            }

        with unittest.mock.patch.dict(
            sys.modules,
            {"datasets": datasets_module, "datasets.distributed": distributed_module},
        ):
            dataset = build_hf_datasets(
                base=base,
                args=args,
                data_transform=_transform,
                dp_rank=0,
                dp_size=2,
            )

        self.assertIsInstance(dataset, StreamingTokenizedDataset)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(base.state.max_steps, 4)

        dataset.set_epoch(0)
        epoch0 = [item["input_ids"].tolist() for item in dataset]
        dataset.set_epoch(1)
        epoch1 = [item["input_ids"].tolist() for item in dataset]
        self.assertEqual(epoch0, [[97], [99], [101]])
        self.assertEqual(epoch1, [[103], [101], [99]])

    def test_streaming_dataset_state_roundtrip(self):
        """Verify streaming dataset state is restored into a fresh wrapper."""
        wrapped = StreamingTokenizedDataset(_FakeHFIterable([{"input_ids": [1], "labels": [1]}]), logical_length=1)
        wrapped.set_epoch(7)
        state = wrapped.state_dict()

        clone_source = _FakeHFIterable([{"input_ids": [1], "labels": [1]}])
        clone = StreamingTokenizedDataset(clone_source, logical_length=1)
        clone.load_state_dict(state)

        self.assertEqual(clone._epoch, 7)
        self.assertEqual(clone_source.loaded_state, {"epoch": 7})

    def test_streaming_builder_repeats_finite_rank_shard_to_logical_budget(self):
        """Verify a finite rank shard repeats rows until the logical length is met."""
        rows = [
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
        ]
        datasets_module, distributed_module = _build_fake_datasets_modules(rows)
        args = _make_args(train_size=4)
        base = _BaseStub(max_steps=99)

        def _transform(batch):
            return {
                "input_ids": [[ord(text)] for text in batch["text"]],
                "labels": [[ord(text)] for text in batch["text"]],
            }

        with unittest.mock.patch.dict(
            sys.modules,
            {"datasets": datasets_module, "datasets.distributed": distributed_module},
        ):
            dataset = build_hf_datasets(
                base=base,
                args=args,
                data_transform=_transform,
                dp_rank=1,
                dp_size=2,
            )

        self.assertIsInstance(dataset, StreamingTokenizedDataset)
        self.assertEqual(len(dataset), 2)
        self.assertEqual([item["input_ids"].tolist() for item in dataset], [[98], [98]])


if __name__ == "__main__":
    unittest.main()
