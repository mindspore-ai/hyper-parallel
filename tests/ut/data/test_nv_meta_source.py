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
"""Two source scenarios: selective storage I/O, then scheduling and worker replay."""

import io
import itertools
import json
import multiprocessing
import pickle
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fsspec
import numpy as np
import torch

from hyper_parallel.data.batching.build_collate_fn import TextPackingCollator
from hyper_parallel.data.batching.build_dataloader import FixedBatchDataLoader, TokenBatchLoader
from hyper_parallel.data.nv_meta.build_dataset import build_nv_meta_dataset
from hyper_parallel.data.nv_meta.provider import NvMetaSource
from hyper_parallel.data.nv_meta.reader import NvMetaDataset, _TarHandlePool
from hyper_parallel.data.nv_meta.sample_adapter import NvMetaSampleAdapter
from hyper_parallel.data.online.provider import SourceBuildContext
from hyper_parallel.data.online.source_views import (
    IndexedBlendMappingView, IndexedIterableView, IndexedMappingView,
    build_indexed_view, validate_indexed_dataloader,
)
from hyper_parallel.data.parallel import DataLoaderParallelContext
from hyper_parallel.data.text.build_dataset import build_online_iterable_dataset
from hyper_parallel.data.text.text_transform import PretokenizedTextTransform
from tests.fixtures.data.nv_meta import make_nv_meta


def _open_reader(root: str) -> tuple[str, bytes]:
    """Build or reuse the same cache in an independent spawn worker."""
    reader = NvMetaDataset(root, cache_dir=Path(root) / "cache")
    try:
        return reader.fingerprint, reader[0]["txt"]
    finally:
        reader.close()


class _Records:
    """Virtual indexed source for ownership and very large plan bounds."""

    def __init__(self, length: int, fingerprint: str = "fixture") -> None:
        """Store cardinality and a source identity without allocating samples."""
        self.length, self.fingerprint, self.epoch = length, fingerprint, -1

    def __len__(self) -> int:
        """Return the virtual source length."""
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        """Materialize one small record."""
        if not 0 <= index < self.length:
            raise IndexError(index)
        return {"id": index, "source": self.fingerprint}

    def set_epoch(self, epoch: int) -> None:
        """Observe the legacy single-sampler epoch convention."""
        self.epoch = epoch

    @staticmethod
    def cost_for_index(index: int, metric: object) -> float:
        """Provide heterogeneous costs without payload I/O."""
        del metric
        return float((index * 19) % 37 + 1)


class TestNvMetaSource(unittest.TestCase):
    """Exercise real prepared files and observable access/restore contracts."""

    def setUp(self) -> None:
        """Close readers before removing temporary files, including on Windows."""
        resources = ExitStack()
        self.addCleanup(resources.close)
        self.root = Path(resources.enter_context(tempfile.TemporaryDirectory()))

    def _reader(self, root: Path, **options: object) -> NvMetaDataset:
        reader = NvMetaDataset(root, cache_dir=root / "cache", **options)
        self.addCleanup(reader.close)
        return reader

    def test_selective_io_and_cache_lifecycle(self) -> None:
        """Read only indexed samples/parts locally and remotely; publish/reuse safe caches."""
        for remote in (False, True):
            with self.subTest(remote=remote):
                self._selective_io(self.root / str(remote), remote)
        root = make_nv_meta(self.root / "shared", [{"txt": b"hello"}] * 20)
        with patch.object(NvMetaDataset, "_compile_cache", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self._reader(root)
        self.assertEqual(list((root / "cache").glob("*/manifest.json")), [])
        with multiprocessing.get_context("spawn").Pool(2) as pool:
            results = pool.map(_open_reader, [str(root)] * 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][1], b"hello")
        self.assertEqual(len(list((root / "cache").glob("*/manifest.json"))), 1)
        self._default_records()
        for extension in ("yaml", "json"):
            with self.subTest(split_format=extension):
                root = make_nv_meta(self.root / extension, [{"txt": b"x"}] * 3,
                                    keys=["a", "b", "c"], shard_name="shard-000001.tar")
                meta = root / ".nv-meta"
                (meta / "split.yaml").unlink()
                (meta / f"split.{extension}").write_text(json.dumps({
                    "split_parts": {"val": ["shard-{000000..000009}.tar"]},
                    "exclude": {"val": ["shard-000001.tar/{a,b}"]},
                }), encoding="utf-8")
                (meta / ".info.json").rename(meta / ".info.yaml")
                self.assertEqual(self._reader(root, split="valid")[0]["sample_key"], "c")
                self.assertEqual(len(self._reader(root, split="train")), 0)
        for failure in ("offset", "cache", "truncated", "compressed"):
            with self.subTest(failure=failure):
                self._corrupt_input(self.root / failure, failure)

    def _default_records(self) -> None:
        """Built-in record selection stays lazy and reads only fixed payload fields."""
        tokens, tensors = io.BytesIO(), io.BytesIO()
        np.save(tokens, np.array([3, 5], dtype=np.int32))
        torch.save({"input_ids": torch.tensor([3, 5])}, tensors)
        cases = (("txt", b"hello", "text", "hello"), ("tokens.npy", tokens.getvalue(), "input_ids", [3, 5]),
                 ("pt", tensors.getvalue(), "input_ids", [3, 5]))
        for part, payload, field, expected in cases:
            with self.subTest(record_part=part):
                root = make_nv_meta(self.root / part, [{part: payload, "unused.bin": b"unselected"}])
                with patch.object(_TarHandlePool, "read", side_effect=AssertionError("eager payload read")):
                    dataset = build_nv_meta_dataset(data_path=root, record_part=part,
                                                    data_config={"cache_dir": root / "cache"})
                reader = dataset.source.source
                self.addCleanup(reader.close)
                with patch.object(reader._handles, "read", wraps=reader._handles.read) as read:
                    result = dataset[0][field]
                    self.assertEqual(result if isinstance(result, str) else result.tolist(), expected)
                    self.assertEqual(sum(call.args[2] for call in read.call_args_list), len(payload))
                raw = self._reader(root, required_parts=[part])[0]
                self.assertEqual(raw["parts"], {part: payload})
                adapter = NvMetaSampleAdapter(record_part=None, field_map={"renamed": part})
                custom = build_nv_meta_dataset(data_path=root, sample_adapter=adapter,
                                               data_config={"cache_dir": root / "cache"})
                self.addCleanup(custom.source.source.close)
                result = custom[0]["renamed"]
                if isinstance(result, dict):
                    result = result[field]
                self.assertEqual(result if isinstance(result, str) else result.tolist(), expected)

    def _selective_io(self, root: Path, remote: bool) -> None:
        parts = {"txt": b"caption", "video.mp4": bytes(256 * 1024), "img1.jpg": b"selected image"}
        make_nv_meta(root, [parts, parts] + [{"txt": b"unused"}] * 1998,
                     metadata={"sample_000001.img1.jpg": {"width": 30, "height": 20}})
        connection = sqlite3.connect(root / ".nv-meta/index.sqlite")
        try:
            if remote:
                filesystem = fsspec.filesystem("memory")
                uri = f"memory://{self.root.name}/samples.tar"
                filesystem.pipe(uri, (root / "samples.tar").read_bytes())
                self.addCleanup(filesystem.rm, uri)
                connection.execute("UPDATE tar_files SET path=?", (uri,))
                connection.execute("INSERT INTO tar_files VALUES (1, 'not-downloaded.tar')")
            else:
                connection.execute("DROP TABLE tar_files")
            connection.commit()
        finally:
            connection.close()
        selected = {"txt": parts["txt"], "img1.jpg": parts["img1.jpg"]}
        with patch.object(_TarHandlePool, "read", side_effect=AssertionError("payload read during indexing")):
            reader = self._reader(root, required_parts=list(selected))
            self.assertIsNone(reader._samples)
            self.assertEqual(len(reader), 2000)
            metadata = reader.index_metadata(1)
            self.assertEqual(reader.cost_for_index(1, "pixels"), 600)
        with patch.object(reader._handles, "read", wraps=reader._handles.read) as read:
            self.assertEqual(reader[1]["parts"], selected)
            self.assertEqual([(call.args[1], call.args[2]) for call in read.call_args_list],
                             [(part.offset, part.size) for part in metadata.parts])
            self.assertEqual(sum(call.args[2] for call in read.call_args_list), sum(map(len, selected.values())))
        self.assertIsInstance(reader._samples, np.memmap)
        self.assertFalse(reader._samples.flags.writeable)
        serialized = pickle.dumps(reader)
        self.assertLess(len(serialized), 16000)
        restored = pickle.loads(serialized)
        self.addCleanup(restored.close)
        self.assertIsNone(restored._samples)
        self.assertEqual(restored[1]["parts"], selected)
        with patch.object(NvMetaDataset, "_compile_cache", side_effect=AssertionError("warm cache rebuilt")):
            self.assertEqual(self._reader(root, required_parts=list(selected)).fingerprint, reader.fingerprint)
        (root / ".nv-meta/index.uuid").write_text("replacement", encoding="utf-8")
        self.assertNotEqual(self._reader(root, required_parts=list(selected)).fingerprint, reader.fingerprint)

    def _corrupt_input(self, root: Path, failure: str) -> None:
        make_nv_meta(root, [{"txt": b"hello"}])
        if failure == "offset":
            connection = sqlite3.connect(root / ".nv-meta/index.sqlite")
            try:
                connection.execute("UPDATE sample_parts SET content_byte_offset=-1")
                connection.commit()
            finally:
                connection.close()
            with self.assertRaisesRegex(ValueError, "negative"):
                self._reader(root)
            return
        reader = self._reader(root)
        if failure == "cache":
            (reader._cache_path / "samples.bin").write_bytes(b"invalid")
        else:
            payload = b"short" if failure == "truncated" else b"\x1f\x8b" + bytes(10000)
            (root / "samples.tar").write_bytes(payload)
        with self.assertRaisesRegex((ValueError, OSError), "Corrupt|range exceeds|uncompressed"):
            _ = reader[0]

    def test_access_plans_worker_resume_and_legacy_compatibility(self) -> None:
        """Keep bounded ownership plans, exact prefetched replay and unchanged old defaults."""
        for length, policy in itertools.product((0, 1, 2, 7, 31, 65), ("none", "greedy", "lpt")):
            with self.subTest(length=length, policy=policy):
                self._ownership(length, policy)
        huge = IndexedBlendMappingView([_Records(10**9, "a"), _Records(10**9, "b")],
                                       [1e300, 1e300], 10**18 + 1)
        self.assertLess(len(pickle.dumps(huge)), 4096)
        self.assertEqual(huge._ends, [5 * 10**17 + 1, 10**18 + 1])
        self.assertIn(huge[10**18]["source"], ("a", "b"))
        sources = []
        for source_index, length in enumerate((2, 3)):
            root = make_nv_meta(self.root / f"mapping-{source_index}", [
                {"json": json.dumps({"id": index, "source": source_index}).encode()} for index in range(length)
            ])
            (root / ".nv-meta/split.yaml").write_text(json.dumps({
                "split_parts": {"train": ["samples.tar"], "val": ["samples.tar"]},
            }), encoding="utf-8")
            sources.append({"data_path": str(root), "weight": 2 - source_index})
        for split, sizes, total, first_count in (("train", None, 5, 3),
                                                 ("train", (12, 6, 0), 12, 8),
                                                 ("validation", (12, 6, 0), 6, 4)):
            with self.subTest(mapping_split=split, split_sizes=sizes):
                provider = NvMetaSource(data_config={"sources": sources, "split": split,
                                                   "cache_dir": str(self.root / "mapping-cache")})
                blend = provider.build(access_mode="mapping", context=SourceBuildContext(split_sizes=sizes))
                for child in blend.source_datasets:
                    self.addCleanup(child.source.source.close)
                samples = [blend[index] for index in range(len(blend))]
                self.assertEqual(len(samples), total)
                self.assertEqual(sum(item["source"] == 0 for item in samples), first_count)
                self.assertEqual({item["source"] for item in samples}, {0, 1})
                self.assertTrue(all(0 <= item["id"] < 2 + item["source"] for item in samples))
        legacy = _Records(3)
        FixedBatchDataLoader(legacy, batch_size=1, sampler_type="single").set_epoch(2)
        self.assertEqual(legacy.epoch, 0)
        self.assertEqual(validate_indexed_dataloader(legacy, sampler_type="cyclic"), ())
        self._invalid_plans()
        self._packed_worker_resume()

    def _ownership(self, length: int, policy: str) -> None:
        source = _Records(length)
        mapping = IndexedMappingView(source, size=length * 2, shuffle=True)
        first = [mapping[index]["id"] for index in range(length)]
        second = [mapping[index]["id"] for index in range(length, length * 2)]
        self.assertEqual(sorted(first), list(range(length)))
        self.assertEqual(sorted(second), list(range(length)))
        ranks = []
        for rank in range(2):
            records = []
            for worker_id in range(3):
                view = IndexedIterableView(source, shuffle=True, balance_policy=policy, balance_by="bytes",
                                           output_index_for_resume=True,
                                           context=SimpleNamespace(dp_rank=rank, dp_world_size=2))
                with patch("hyper_parallel.data.online.source_views.torch_data.get_worker_info",
                           return_value=SimpleNamespace(id=worker_id, num_workers=3)):
                    emitted = list(view)
                view.set_epoch(5)
                for sample, key in emitted:
                    self.assertEqual(view.get_item(key), sample)
                records.extend(key for _, key in emitted)
            ranks.append(records)
        self.assertEqual(len(ranks[0]), len(ranks[1]))
        self.assertEqual(sum(map(len, ranks)), length - length % 2)
        self.assertEqual(len(set(ranks[0] + ranks[1])), length - length % 2)

    def _invalid_plans(self) -> None:
        view = IndexedMappingView(_Records(20), balance_policy="greedy", balance_slots=2, balance_group_size=4)
        for options in ({"sampler_type": "cyclic"}, {"dp_world_size": 3}, {"micro_batch_size": 3},
                        {"data_rearrange_map": {0: 1}}):
            with self.subTest(sampler=options), self.assertRaises(ValueError):
                validate_indexed_dataloader(SimpleNamespace(source_dataset=view), **options)
        cursor = IndexedIterableView(_Records(20))
        next(iter(cursor))
        state = cursor.state_dict()
        with self.assertRaisesRegex(ValueError, "does not match"):
            IndexedIterableView(_Records(20, "replacement")).load_state_dict(state)
        cursor.load_state_dict(state)
        with patch("hyper_parallel.data.online.source_views.torch_data.get_worker_info",
                   return_value=SimpleNamespace(id=0, num_workers=2)):
            with self.assertRaisesRegex(ValueError, "topology"):
                next(iter(cursor))
        rejected = build_indexed_view(
            _Records(3), access_mode="iterable", context=SourceBuildContext(),
            data_config={"repeat": True, "filter_samples": True}, sample_filter=lambda _: False,
        )
        with self.assertRaisesRegex(ValueError, "rejected-sample limit"):
            next(iter(rejected))
        provider = NvMetaSource(data_config={"sources": [{"data_path": "unopened", "read_balance": "bytes"}]})
        with patch("hyper_parallel.data.nv_meta.provider.NvMetaDataset") as reader:
            with self.assertRaisesRegex(ValueError, "cannot enable read_balance"):
                provider.build(access_mode="mapping", context=SourceBuildContext())
            reader.assert_not_called()
        for options in ({"record_part": "csv"}, {"record_part": None},
                        {"data_config": {"split_name": "train"}}, {"data_config": {"parts": ["txt"]}}):
            with self.subTest(source_options=options), self.assertRaises(ValueError):
                NvMetaSource(data_path="unopened", **options)
        with patch("hyper_parallel.data.nv_meta.provider.NvMetaDataset") as reader:
            with self.assertRaisesRegex(ValueError, "not both"):
                build_nv_meta_dataset(data_path="unopened", record_part="txt", sample_adapter=NvMetaSampleAdapter())
            reader.assert_not_called()
        provider = NvMetaSource(data_config={"sources": [
            {"data_path": "unopened-a", "weight": 1}, {"data_path": "unopened-b", "weight": 1},
        ]})
        with patch("hyper_parallel.data.nv_meta.provider.NvMetaDataset") as reader:
            with self.assertRaisesRegex(ValueError, "requires mapping access"):
                provider.build(access_mode="iterable", context=SourceBuildContext())
            reader.assert_not_called()

    def _packed_worker_resume(self) -> None:
        datasets, loaders, records = [], [], []
        for index in range(23):
            token, length = 10 + index, 3 + index % 3
            record = {"input_ids": [token] * length, "labels": [token + 1] * length}
            records.append({"json": json.dumps(record).encode()})
        root = make_nv_meta(self.root / "single", records)
        try:
            for _ in range(3):
                dataset = build_online_iterable_dataset(
                    data_config={}, source=NvMetaSource(data_path=root, data_config={
                        "repeat": True, "shuffle": True,
                        "cache_dir": str(self.root / "cache"), "output_index_for_resume": True,
                    }), transform=PretokenizedTextTransform(8),
                    dataloader_context=DataLoaderParallelContext(barrier=lambda: None),
                )
                datasets.append(dataset)
                loaders.append(TokenBatchLoader(
                    dataset, collate_fn=TextPackingCollator(), batch_size=1, dp_world_size=1,
                    max_seq_len=8, min_buffered_samples=7, save_by_idx=True,
                    num_workers=2, prefetch_factor=2, persistent_workers=True,
                ))
                loaders[-1].source_dataloader.multiprocessing_context = "spawn"
            original, restored = loaders[0], loaders[1]
            original.set_epoch(3)
            iterator = iter(original)
            list(itertools.islice(iterator, 4))
            state = original.state_dict()
            self.assertTrue(state["buffer"])
            self.assertTrue(all(isinstance(key, int) and 0 <= key < len(records) and position == 0
                                for key, position in state["buffer"]))
            expected = list(itertools.islice(iterator, 8))
            restored.load_state_dict(state)
            restored.set_epoch(3)
            self._equal_batches(list(itertools.islice(restored, 8)), expected)
            original.set_epoch(4)
            reference = loaders[2]
            reference.set_epoch(4)
            self._equal_batches(list(itertools.islice(original, 4)), list(itertools.islice(reference, 4)))
        finally:
            for loader in loaders:
                worker_iterator = loader.source_dataloader._iterator
                if worker_iterator is not None:
                    worker_iterator._shutdown_workers()
            for dataset in datasets:
                dataset.source_dataset.source.source.close()

    def _equal_batches(self, actual: list[dict], expected: list[dict]) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_batch, expected_batch in zip(actual, expected):
            self.assertEqual(set(actual_batch), set(expected_batch))
            for field in actual_batch:
                torch.testing.assert_close(actual_batch[field], expected_batch[field])
