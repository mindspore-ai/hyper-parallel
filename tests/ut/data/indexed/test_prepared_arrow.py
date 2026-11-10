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
"""Exercise prepared Indexed records through real Arrow, YAML and batch assembly."""

from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import gc
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import torch
import yaml
from datasets import Dataset

from hyper_parallel.data.batching import (
    FixedBatchDataLoader, TextParallelBatch, build_indexed_collate_fn, build_sft_collate_fn,
)
from hyper_parallel.data.batching.build_dataloader import build_dataloader
from hyper_parallel.data.batching.runtime_input import IndexedBoundaryResolver, RuntimeInputAdapter
from hyper_parallel.data.indexed.indexed_data_reader import ArrowDataReader
from hyper_parallel.data.indexed.indexed_data_config import (
    GPTDatasetConfig, build_gpt_dataset_config, resolve_data_paths,
)
from hyper_parallel.data.indexed.indexed_data_reader import _IndexReader
from hyper_parallel.data.indexed.indexed_pretrain_dataset import GPTDataset, GPTFromMRDataset, MockGPTDataset
from hyper_parallel.data.parallel import DataLoaderParallelContext
from hyper_parallel.data.text.build_dataset import build_indexed_text_dataset
from hyper_parallel.data.text.build_tokenizer import AutoTokenizer
from hyper_parallel.data.tools.io import IndexedDatasetBuilder
from hyper_parallel.distributed_data import WorkloadCost
from hyper_parallel.trainer.config.resolver import resolve_config


@contextmanager
def _test_directory() -> Iterator[Path]:
    """Release indexed mmap caches before deleting files on shared storage."""
    with TemporaryDirectory() as directory:
        try:
            yield Path(directory)
        finally:
            _IndexReader.__getitem__.cache_clear()
            gc.collect()


class _BoundaryAdapter(RuntimeInputAdapter):
    """Expose boundaries without importing a particular model."""

    def build_runtime_inputs(self, *, batch: Mapping[str, Any], context: Any) -> Mapping[str, Any]:
        """Forward global packed boundaries to a test model."""
        del context
        return {"cu_seq_lens": batch["cu_seq_lens"]}


class TestPreparedArrow(unittest.TestCase):
    """Integration checks use actual files, with no NPU or distributed setup."""

    def setUp(self) -> None:
        """Create isolated corpora and tokenizer metadata without network access."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.root = stack.enter_context(_test_directory())
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/unused/tokenizer", tokenizer_type="pretokenized", vocab_size=128, eod_token_id=2,
        )
        self.rows = [
            {"input_ids": [11, 2, 13, 21, 22], "labels": [2, 13, -100, 22, -100],
             "cu_seq_lens": [0, 3, 5], "loss_mask": [0, 1, 1, 1, 1]},
            {"input_ids": [31, 32, 41, 0, 0], "labels": [32, -100, 42, -100, -100],
             "cu_seq_lens": [0, 2, 3, 5], "loss_mask": [1, 1, 1, 0, 0]},
        ]
        self.options = {
            "random_seed": 42, "seq_length": 5, "simple_blend": "intra",
            "tokenizer": self.tokenizer, "labels_are_shifted": True,
            "low_level_dataset": {"_target_": "hyper_parallel.data.indexed.SFTFromMRDataset"},
        }

    def _write(self, name, rows):
        """Write the Arrow IPC stream format used by HF Dataset files."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows)
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return str(path)

    def _build(self, path, count=2, **options):
        """Call the public Indexed builder with explicit sample counts."""
        config = {**self.options, **options}
        return build_indexed_text_dataset(
            data_path=path, data_config=config, train_valid_test_num_samples=(count, 0, 0),
        )[0]

    def _runtime(self, *, cp_size=1, cp_rank=0, **options):
        """Mock only parallel ownership, retaining real sharding and masks."""
        context = DataLoaderParallelContext(cp_world_size=cp_size, cp_rank=cp_rank)
        with patch("hyper_parallel.data.batching.get_batch.create_dataloader_parallel_context", return_value=context):
            return TextParallelBatch(
                mesh_context=None, device=torch.device("cpu"), tokenizer=self.tokenizer,
                data_config=self.options, source_type="indexed", pp_shared_data=False, **options,
            )

    def test_arrow_to_model_preserves_labels_and_boundaries(self):
        """Fixed-length records retain masks and internal boundaries through get_batch."""
        dataset = self._build(self._write("data.arrow", self.rows))
        loader = FixedBatchDataLoader(dataset, batch_size=2, collate_fn=build_sft_collate_fn())
        runtime = self._runtime(attention_mode="dense", reset_position_ids=True)
        model, loss = runtime(iter(loader))
        self.assertEqual(model["input_ids"].tolist(), [[11, 2, 13, 21, 22], [31, 32, 41, 0, 0]])
        self.assertEqual(model["shift_labels"].tolist(), [[-100, 13, -100, 22, -100], [32, -100, 42, -100, -100]])
        self.assertEqual(loss["loss_mask"].sum().item(), 4)
        self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 0, 1], [0, 1, 0, 0, 1]])
        mask = model["attention_mask"][0, 0]
        self.assertTrue(mask[2, 1])
        self.assertFalse(mask[3, 2])
        self.assertFalse(model["attention_mask"][1, 0, 2, 1])
        self.assertFalse(model["attention_mask"][1, 0, 3, 2])
        self.assertEqual(dataset.dataset[0]["labels"], self.rows[0]["labels"])

    def test_record_length_matches_configured_sequence_length(self):
        """Reject both short and long records on access, including lazy sources."""
        path = self._write("length.arrow", self.rows)
        for lazy in (False, True):
            for extra in (False, True):
                for length in (4, 5, 6):
                    with self.subTest(lazy=lazy, extra=extra, length=length):
                        dataset = self._build(path, seq_length=length, data_lazy_load=lazy,
                                              add_extra_token_to_sequence=extra)
                        loader = FixedBatchDataLoader(dataset, batch_size=2,
                                                      collate_fn=build_sft_collate_fn())
                        if length == 5:
                            batch = next(iter(loader))
                            self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
                            self.assertEqual(batch["input_ids"][0].tolist(), self.rows[0]["input_ids"])
                        else:
                            with self.assertRaisesRegex(ValueError, f"expected {length}, got 5"):
                                next(iter(loader))

    def test_indexed_input_ids_through_balancing_and_repacking(self) -> None:
        """Feature: Unified Indexed text fields.
        Description: Feed real Arrow records through Trainer balancing with and without repacking.
        Expectation: Input IDs, shifted labels and document boundaries reach the runtime unchanged.
        """
        dataset = self._build(self._write("balanced.arrow", self.rows))
        context = SimpleNamespace(
            dp_rank=0, dp_size=1, pp_size=1,
            device_mesh=SimpleNamespace(mesh_shape=(1,), mesh_dim_names=("dp",), rank_list=(0,)),
        )
        for repack in (False, True):
            with self.subTest(repack=repack):
                loaders, samplers = build_dataloader(
                    SimpleNamespace(load_balance={"repack_sequences": repack, "pad_token_id": 0}),
                    datasets=(dataset, None, None), collate_fn=build_sft_collate_fn(),
                    mesh_context=context,
                    training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=2, seed=42),
                    data_config=self.options,
                    cost_model=lambda sample: WorkloadCost(llm=sample.pack_tokens ** 2),
                )
                self.assertIsNone(samplers[0])
                with loaders[0] as loader:
                    batch = next(loader)
                self.assertNotIn("tokens", batch)
                self.assertEqual(batch["input_ids"].tolist(), [row["input_ids"] for row in self.rows])
                self.assertEqual(batch["labels"].tolist(),
                                 [[-100, 13, -100, 22, -100], [32, -100, 42, -100, -100]])
                runtime = self._runtime(
                    attention_mode="compressed", runtime_input_adapter=_BoundaryAdapter(), reset_position_ids=True,
                )
                model, loss = runtime(iter([batch]))
                self.assertTrue(torch.equal(model["input_ids"], batch["input_ids"]))
                self.assertTrue(torch.equal(loss["shift_labels"], batch["labels"]))
                self.assertEqual(model["cu_seq_lens"].tolist(), [0, 3, 5, 7, 8, 10])
                self.assertEqual(model["position_ids"].tolist(), [[0, 1, 2, 0, 1], [0, 1, 0, 0, 1]])
                self.assertEqual(loss["loss_mask"].sum().item(), 4)
        self.assertEqual(dataset.dataset[0]["labels"], self.rows[0]["labels"])

    def test_skip_data_check_only_skips_record_length_check(self):
        """Explicit opt-out preserves variable rows without bypassing label validation."""
        rows = deepcopy(self.rows)
        rows[1] = {"input_ids": [31, 32, 41], "labels": [32, -100, 42],
                   "cu_seq_lens": [0, 2, 3], "loss_mask": [1, 1, 1]}
        path = self._write("variable.arrow", rows)
        checked = self._build(path)
        self.assertEqual(len(checked[0]["input_ids"]), 5)
        with self.assertRaisesRegex(ValueError, "record 1: expected 5, got 3"):
            _ = checked[1]
        skipped = self._build(path, seq_length=8, skip_data_check=True)
        batch = next(iter(FixedBatchDataLoader(skipped, batch_size=2, collate_fn=build_sft_collate_fn())))
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(batch["labels"][1].tolist(), [32, -100, 42, -100, -100])
        rows[1]["labels"] = [32]
        invalid = self._build(self._write("invalid-shape.arrow", rows), skip_data_check=True)
        with self.assertRaisesRegex(ValueError, "matching nonempty 1D"):
            _ = invalid[1]

    def test_multiple_files_and_hf_saved_dataset(self):
        """Each loose or HF-saved Arrow file remains an independent source."""
        b = self._write("loose/b.arrow", self.rows[1:])
        a = self._write("loose/a.arrow", self.rows[:1])
        paths = resolve_data_paths(str(self.root / "loose"), file_suffix=".arrow", distributed_walk=False)
        self.assertEqual(paths, ["1", a, "1", b])
        with self.assertRaisesRegex(ValueError, "one .arrow file"):
            ArrowDataReader(str(self.root / "loose"))
        restored = pickle.loads(pickle.dumps(ArrowDataReader(b)))
        self.assertEqual(restored[0], self.rows[1])
        self.assertTrue(restored.dataset.cache_files)
        dataset = self._build(str(self.root / "loose"))
        self.assertEqual(len(dataset.datasets), 2)
        self.assertEqual(len(dataset.datasets[0].dataset), 1)
        saved = self.root / "saved"
        Dataset.from_list(self.rows).save_to_disk(str(saved), num_shards=2)
        dataset = self._build(str(saved))
        self.assertEqual(len(dataset.datasets), 2)
        self.assertEqual(dataset[0]["input_ids"].tolist(), self.rows[0]["input_ids"])
        self.assertEqual(dataset[1]["input_ids"].tolist(), self.rows[1]["input_ids"])

    def test_each_arrow_file_is_split_before_blending(self):
        """80/20 splitting keeps both sources in both train and validation."""
        for name, first in (("a", 10), ("b", 50)):
            rows = [{**deepcopy(self.rows[0]), "input_ids": [first + i, 2, 13, 21, 22]} for i in range(10)]
            self._write(f"split/{name}.arrow", rows)
        for mode in ("intra", "inter"):
            for lazy in (False, True):
                with self.subTest(mode=mode, lazy=lazy):
                    train, valid, test = build_indexed_text_dataset(
                        data_path=str(self.root / "split"),
                        data_config={**self.options, "split": "4,1,0", "simple_blend": mode,
                                     "data_lazy_load": lazy},
                        train_valid_test_num_samples=(16, 4, 0),
                    )
                    self.assertIsNone(test)
                    self.assertEqual(len(train.datasets), 2)
                    self.assertEqual(len(valid.datasets), 2)
                    expected = (list(range(10, 18)) + list(range(50, 58)) if mode == "intra"
                                else [value for pair in zip(range(10, 18), range(50, 58)) for value in pair])
                    self.assertEqual([int(train[i]["input_ids"][0]) for i in range(16)], expected)
                    self.assertEqual(sorted(int(valid[i]["input_ids"][0]) for i in range(4)), [18, 19, 58, 59])

    def test_arrow_discovery_on_distributed_ranks(self):
        """Root and nested shards are discovered identically on all ranks."""
        top = self._write("walk/a.arrow", self.rows)
        left = self._write("walk/left/b.arrow", self.rows)
        right = self._write("walk/right/c.arrow", self.rows)
        (self.root / "walk/ignored.bin").write_bytes(b"not an Arrow source")
        layout = ([str(self.root / "walk/left"), str(self.root / "walk/right")], [top])

        def check_rank(rank):
            """Keep gathered state local to one synchronous rank simulation."""
            seen = []

            def gather(output: list[Any], value: Any) -> None:
                """Emulate serialized object gathering without sharing mutable lists."""
                seen.append(value)
                output[:] = [deepcopy(layout), None] if len(seen) == 1 else [[left], [right]]

            with (self.subTest(rank=rank),
                  patch("hyper_parallel.data.indexed.indexed_data_config.dist.get_rank", return_value=rank),
                  patch("hyper_parallel.data.indexed.indexed_data_config.dist.get_world_size", return_value=2),
                  patch("hyper_parallel.data.indexed.indexed_data_config.dist.all_gather_object", side_effect=gather)):
                paths = resolve_data_paths(str(self.root / "walk"), file_suffix=".arrow")
            self.assertEqual(paths, ["1", top, "1", left, "1", right])
            self.assertEqual(seen[1], [left] if rank == 0 else [right])

        for rank in range(2):
            check_rank(rank)
        weighted = ["2", top, "1", left]
        self.assertEqual(resolve_data_paths(weighted, file_suffix=".arrow", distributed_walk=False), weighted)

    def test_yaml_builds_reader_lazily_not_during_target_resolution(self):
        """The reviewer's class selector reaches the Dataset builder through real YAML."""
        path = self._write("data.arrow", self.rows)
        config = resolve_config(yaml.safe_load("""
model:
  _target_: torch.nn.Linear
  in_features: 4
  out_features: 4
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
training:
  train_iters: 1
  global_batch_size: 2
  micro_batch_size: 2
dataset:
  _target_: hyper_parallel.data.text.build_dataset.build_indexed_text_dataset
  data_path: PLACEHOLDER
  data_config:
    seq_length: 5
    low_level_dataset:
      _target_: hyper_parallel.data.indexed.SFTFromMRDataset
    simple_blend: intra
    data_lazy_load: true
    labels_are_shifted: true
dataloader:
  _target_: hyper_parallel.data.batching.FixedBatchDataLoader
  collate_fn:
    _target_: hyper_parallel.data.batching.build_sft_collate_fn
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: indexed
    attention_mode: dense
""".replace("PLACEHOLDER", path)))
        with patch.object(ArrowDataReader, "__init__", side_effect=AssertionError("eager read")):
            train, _, _ = config.dataset.build(tokenizer=self.tokenizer, training_config=config.training)
        self.assertEqual(train[0]["labels"].tolist(), [-100, 13, -100, 22, -100])
        self.assertIsInstance(config.to_dict()["dataset"]["data_config"]["low_level_dataset"], dict)

    def test_batched_sft_rows_remove_boundary_padding(self):
        """Grouped inputs retain B, while default collation still handles other fields."""
        samples = [
            {"input_ids": [[10, 11, 12, 13]], "labels": [[11, 12, -100, 14]],
             "cu_seq_lens": [[0, 2, 4, -1]], "source_id": 1},
            {"input_ids": [[20, 21, 22, 23]], "labels": [[21, -100, 23, 24]],
             "cu_seq_lens": [[0, 1, 2, 4]], "source_id": 2},
        ]
        batch = build_sft_collate_fn()(samples)
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"].tolist(), [[11, 12, -100, 14], [21, -100, 23, 24]])
        self.assertEqual(batch["cu_seq_lens"], [[0, 2, 4], [0, 1, 2, 4]])
        self.assertEqual(batch["source_id"].tolist(), [1, 2])
        self.assertEqual(IndexedBoundaryResolver(None).resolve(batch).tolist(), [0, 2, 4, 5, 6, 8])

    def test_field_mapping_and_missing_boundaries(self):
        """JT-style column names work; omitted boundaries mean a single document."""
        row = {"ids": [10, 11], "target": [11, 12], "cu_seqlens": [0, 1, 2], "mask": [0, 1]}
        spec = {**self.options["low_level_dataset"], "input_ids_key": "ids", "labels_key": "target",
                "boundaries_key": "cu_seqlens", "loss_mask_key": "mask"}
        dataset = self._build(self._write("jt.arrow", [row]), count=1, seq_length=2, low_level_dataset=spec)
        self.assertEqual(dataset[0]["labels"].tolist(), [-100, 12])
        self.assertEqual(dataset[0]["cu_seq_lens"].tolist(), [0, 1, 2])
        del row["cu_seqlens"]
        dataset = self._build(self._write("single.arrow", [row]), count=1, seq_length=2, low_level_dataset=spec)
        self.assertEqual(dataset[0]["cu_seq_lens"].tolist(), [0, 2])

    def test_invalid_records_fail_before_distribution(self):
        """Do not silently accept damaged boundaries, masks or label alignment."""
        changes = [{"cu_seq_lens": [0, 2, 2, 5]}, {"cu_seq_lens": [0, 4]}, {"labels": [1]},
                   {"loss_mask": [1]}, {"loss_mask": [0.5] * 5}, {"labels": [-100] * 5},
                   {"input_ids": [128] * 5}]
        for i, update in enumerate(changes):
            with self.subTest(update=update):
                row = {**deepcopy(self.rows[0]), **update}
                dataset = self._build(self._write(f"invalid-{i}.arrow", [row]), count=1)
                with self.assertRaises(ValueError):
                    _ = dataset[0]
        self.options["labels_are_shifted"] = False
        with self.assertRaisesRegex(ValueError, "shifted labels"):
            self._runtime()

    def test_invalid_sampling_and_missing_adapter(self):
        """Reject re-packing policy and compressed attention lacking boundary consumption."""
        path = self._write("data.arrow", self.rows)
        with self.assertRaisesRegex(ValueError, "simple_blend"):
            self._build(path, simple_blend="no")
        with self.assertRaisesRegex(ValueError, "exceed"):
            self._build(path, count=3)
        with self.assertRaisesRegex(ValueError, "boundary-aware"):
            self._runtime(attention_mode="compressed")

    def test_bin_mr_explicit_class_preserves_default_collation(self):
        """Bin/idx token shifting and its default row stacking remain unchanged."""
        prefix = str(self.root / "legacy")
        builder = IndexedDatasetBuilder(prefix + ".bin", dtype=np.int32)
        for row in ([10, 11, 12, 13, 14], [21, 22, 23, 24, 25]):
            builder.add_item(torch.tensor(row))
            builder.end_document()
        builder.finalize(prefix + ".idx")
        dataset = self._build(prefix, seq_length=4, low_level_dataset=GPTFromMRDataset)
        batch = build_indexed_collate_fn()([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))
        self.assertEqual(batch["labels"].tolist(), [[11, 12, 13, 14], [22, 23, 24, 25]])
        sft_batch = build_sft_collate_fn()([dataset[0], dataset[1]])
        self.assertTrue(torch.equal(batch["input_ids"], sft_batch["input_ids"]))
        standard = self._build(prefix, seq_length=4, low_level_dataset=GPTDataset, simple_blend="no")
        self.assertEqual(len(standard), 2)
        self.assertEqual(standard[0]["input_ids"].shape, (4,))
        self.assertEqual(standard[0]["labels"].shape, (4,))
        with self.assertRaisesRegex(ValueError, "simple_blend"):
            self._build(prefix, seq_length=4, low_level_dataset=GPTFromMRDataset, simple_blend="no")

    def test_explicit_class_and_blend_are_required(self):
        """No implicit GPT fallback; legacy MR flags do not select a class."""
        for skip in (False, True):
            with self.subTest(skip=skip):
                options = {**self.options, "skip_data_check": skip}
                del options["seq_length"]
                with self.assertRaisesRegex(ValueError, "Missing indexed Dataset build option: seq_length"):
                    build_gpt_dataset_config(["unused.arrow"], options)

        config_args = {"random_seed": 42, "sequence_length": 4, "mock": True}
        for missing in ("low_level_dataset", "simple_blend"):
            with self.subTest(missing=missing):
                fields = {"low_level_dataset": MockGPTDataset, "simple_blend": "no"}
                del fields[missing]
                with self.assertRaisesRegex(ValueError, missing):
                    GPTDatasetConfig(**config_args, **fields)
                with self.assertRaisesRegex(ValueError, missing):
                    GPTDatasetConfig(**config_args, **fields, **{missing: None})
        options = {"seq_length": 4, "random_seed": 42, "tokenizer": self.tokenizer,
                   "simple_blend": "no"}
        with self.assertRaisesRegex(ValueError, "low_level_dataset"):
            build_gpt_dataset_config(["unused"], options)
        with self.assertRaisesRegex(ValueError, "low_level_dataset"):
            build_gpt_dataset_config(["unused"], {**options, "is_dataset_from_mr": True})
        options["low_level_dataset"] = GPTDataset
        config = build_gpt_dataset_config(["unused"], options)
        self.assertIs(config.low_level_dataset, GPTDataset)
        del options["simple_blend"]
        with self.assertRaisesRegex(ValueError, "simple_blend"):
            build_gpt_dataset_config(["unused"], options)
        config = build_gpt_dataset_config([], {**options, "mock_data": True,
                                              "low_level_dataset": MockGPTDataset, "simple_blend": "no"})
        self.assertIs(config.low_level_dataset, MockGPTDataset)
        mock = self._build(None, mock_data=True, low_level_dataset=MockGPTDataset, simple_blend="no")
        self.assertEqual(len(mock), 2)
        self.assertEqual(mock[0]["input_ids"].shape, (5,))

    def test_cp_multibatch_positions_and_targets(self):
        """Row-local CP slices keep both rows and reconstruct their full targets."""
        dataset = self._build(self._write("data.arrow", self.rows))
        batch = build_sft_collate_fn(SimpleNamespace(cp_size=2, tp_size=2))([dataset[0], dataset[1]])
        self.assertEqual(batch["input_ids"].shape, (2, 8))
        parts, positions = [], []
        for rank in range(2):
            runtime = self._runtime(cp_size=2, cp_rank=rank, attention_mode="compressed",
                                    runtime_input_adapter=_BoundaryAdapter(), reset_position_ids=True)
            model, loss = runtime(iter([batch]))
            self.assertEqual(model["cu_seq_lens"].tolist(), [0, 3, 5, 8, 10, 11, 13, 16])
            parts.append(loss["shift_labels"])
            positions.append(model["position_ids"])
        self.assertTrue(torch.equal(torch.cat(parts, dim=1), batch["labels"]))
        self.assertEqual(torch.cat(positions, dim=1).tolist(),
                         [[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 0, 0, 1, 0, 1, 2]])

    def test_source_mask_is_optional_and_not_transported(self):
        """Absent masks preserve labels; explicit masks become -100 once on CPU."""
        row = deepcopy(self.rows[0])
        del row["loss_mask"]
        dataset = self._build(self._write("no-mask.arrow", [row]), count=1)
        self.assertEqual(dataset[0]["labels"].tolist(), row["labels"])
        self.assertNotIn("loss_mask", dataset[0])
        row["cu_seq_lens"].append(-1)
        dataset = self._build(self._write("padded-boundaries.arrow", [row]), count=1)
        self.assertEqual(dataset[0]["cu_seq_lens"].tolist(), [0, 3, 5])

    def test_sft_collator_rejects_invalid_row_metadata(self):
        """A malformed row cannot silently change document boundaries."""
        collate = build_sft_collate_fn()
        for boundaries in ([[0, 2, 2, 4]], [[0, 3]], [[0, 4], [0, 4]], [[[0, 4]]]):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                collate([{"input_ids": [[1, 2, 3, 4]], "labels": [[2, 3, 4, 5]], "cu_seq_lens": boundaries}])
