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
"""Tests for unpacked Indexed source Dataset behavior."""

import unittest
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.build_dataloader import build_dataloader
from hyper_parallel.data.batching.build_collate_fn import build_indexed_collate_fn
from hyper_parallel.data.batching.get_batch import ParallelBatch
from hyper_parallel.data.indexed.indexed_data_config import GPTDatasetConfig
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader
from hyper_parallel.data.indexed.indexed_pretrain_dataset import GPTDataset, IndexedSourceDataset
from hyper_parallel.data.indexed.io import IndexedDatasetBuilder
from hyper_parallel.data.text.build_dataset import build_indexed_text_dataset
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.distributed_data import SampleMetadata, build_distributed_dataloader
from tests.common.mark_utils import arg_mark


class _Split(Enum):
    TRAIN = 0


class _Tokenizer:
    pad = None
    eod = 9
    special_tokens_dict = {"eod": 9}
    unique_identifiers = {"tokenizer": "test"}

    def __len__(self) -> int:
        """Return a small test vocabulary size."""
        return 32


class _LowLevelDataset:
    """Track payload reads independently from index metadata access."""

    def __init__(self, sequences: list[list[int]]) -> None:
        """Store token sequences and their metadata lengths."""
        self.sequences = [np.asarray(sequence, dtype=np.int64) for sequence in sequences]
        self.sequence_lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)
        self.payload_reads = 0

    def __getitem__(self, index: int) -> np.ndarray:
        """Read and count one token payload."""
        self.payload_reads += 1
        return self.sequences[index]


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _write_indexed_source(directory: str, name: str, sequences: list[list[int]]) -> str:
    """Write a real corpus with the same builder used by the offline tool."""
    prefix = str(Path(directory) / name)
    builder = IndexedDatasetBuilder(prefix + ".bin")
    for sequence in sequences:
        builder.add_document(torch.tensor(sequence), [len(sequence)])
    builder.finalize(prefix + ".idx")
    return prefix


def _provider_config(**overrides: object) -> dict[str, object]:
    """Return the provider options for unpacked Indexed text."""
    return {
        "seq_length": 8,
        "split": "1, 0, 0",
        "mock_data": False,
        "is_dataset_from_mr": False,
        "simple_blend": "no",
        "data_lazy_load": False,
        "distributed_walk": False,
        "packing_stage": "distributed_dataloader",
        **overrides,
    }


def _build_source_loader(datasets: tuple, data_config: dict, **worker_options: object) -> object:
    """Use the same DataLoader builder and derived sizes as TextTrainer."""
    mesh_context = SimpleNamespace(device_mesh=_StandaloneMesh(), dp_rank=0, dp_size=1)
    training_config = SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=7)
    loaders, samplers = build_dataloader(
        SimpleNamespace(**worker_options), datasets=datasets, collate_fn=None, mesh_context=mesh_context,
        training_config=training_config, data_config=data_config,
    )
    if samplers != (None, None, None):
        raise ValueError("Distributed packing must own the sample schedule")
    return loaders[0]


def _source_config(sequence_length: int = 8) -> GPTDatasetConfig:
    """Build one unpacked Indexed source configuration."""
    return GPTDatasetConfig(
        random_seed=7,
        sequence_length=sequence_length,
        blend=["unused"],
        split="1, 0, 0",
        tokenizer=_Tokenizer(),
        packing_stage="distributed_dataloader",
    )


class TestIndexedSourceDataset(unittest.TestCase):
    """Verify source reads remain separate from planning metadata."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_metadata_uses_index_lengths_without_payload_read(self) -> None:
        """Feature: Indexed source metadata.
        Description: Query metadata before materializing the source sample.
        Expectation: Token length comes from the index and the binary payload remains unread.
        """
        low_level_dataset = _LowLevelDataset([[1, 2, 9], [3, 4, 5, 9]])
        dataset = IndexedSourceDataset(
            low_level_dataset,
            "unused",
            np.asarray([1, 0], dtype=np.int32),
            1,
            _Split.TRAIN,
            _source_config(),
        )

        metadata = dataset.get_sample_metadata(0)

        self.assertEqual(len(dataset), 2)
        self.assertEqual(metadata, SampleMetadata(pack_tokens=3, sample_id=1))
        self.assertEqual(low_level_dataset.payload_reads, 0)
        self.assertFalse(hasattr(dataset, "sample_index"))

        sample = dataset[0]
        self.assertEqual(sample["input_ids"].tolist(), [3, 4, 5])
        self.assertEqual(sample["labels"].tolist(), [4, 5, 9])
        self.assertEqual(low_level_dataset.payload_reads, 1)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_real_indexed_files_through_trainer_batch_adapter(self) -> None:
        """Feature: Indexed source runtime fields.
        Description: Build and collate two real binary source records.
        Expectation: Metadata precedes reads; labels, positions, attention, and padding match document boundaries.
        """
        with TemporaryDirectory() as directory:
            prefix = _write_indexed_source(directory, "corpus", [[1, 2, 9], [3, 4, 5, 9]])
            config = _provider_config()
            with patch.object(IndexedDataReader, "__getitem__", side_effect=AssertionError("Early payload read")):
                datasets = build_indexed_text_dataset(
                    data_path=prefix, data_config=config, tokenizer=_Tokenizer(),
                    train_valid_test_num_samples=(1, 0, 0),
                )
                loader = _build_source_loader(datasets, config)
                self.assertEqual(len(datasets[0]), 2)
                self.assertEqual(datasets[0].get_sample_metadata(1).pack_tokens, 3)

            batch = next(loader)
            self.assertEqual(batch["input_ids"].tolist(), [[3, 4, 5, 1, 2, 0, 0, 0]])
            self.assertEqual(batch["labels"].tolist(), [[4, 5, 9, 2, 9, -100, -100, -100]])
            self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 3, 5, 8])
            runtime = ParallelBatch(
                mesh_context=None, device="cpu", tokenizer=_Tokenizer(),
                data_config=config, source_type="indexed_source", pp_shared_data=False,
            )
            model_inputs, loss_inputs = runtime(iter([batch]))
            self.assertEqual(model_inputs["position_ids"].tolist(), [[0, 1, 2, 0, 1, 0, 1, 2]])
            self.assertEqual(loss_inputs["loss_mask"].tolist(), [[1, 1, 1, 1, 1, 0, 0, 0]])
            self.assertTrue(torch.equal(model_inputs["shift_labels"], batch["labels"]))
            mask = model_inputs["attention_mask"][0, 0]
            self.assertFalse(bool(mask[3, 2]))
            self.assertTrue(bool(mask[4, 3]))
            with self.assertRaises(StopIteration):
                next(loader)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_default_provider_still_builds_gpt_sample_indices(self) -> None:
        """Feature: Default Indexed provider compatibility.
        Description: Build a corpus without opting into distributed packing.
        Expectation: The original GPT sample-index path and fixed token shapes are retained.
        """
        with TemporaryDirectory() as directory:
            prefix = _write_indexed_source(directory, "corpus", [[1, 2, 3, 4, 9], [5, 6, 7, 8, 9]])
            config = _provider_config(seq_length=4)
            del config["packing_stage"]
            dataset, _, _ = build_indexed_text_dataset(
                data_path=prefix, data_config=config, tokenizer=_Tokenizer(),
                train_valid_test_num_samples=(2, 0, 0),
            )
            self.assertIsInstance(dataset, GPTDataset)
            self.assertTrue(hasattr(dataset, "sample_index"))
            self.assertEqual(dataset[0]["tokens"].shape, (4,))
            self.assertEqual(dataset[0]["labels"].shape, (4,))

    def test_native_batch_sampler_keeps_gpt_fields_and_round_membership(self) -> None:
        """Native GPT slicing, label shifts, masks, and positions survive the new opt-in."""
        with TemporaryDirectory() as directory:
            prefix = _write_indexed_source(
                directory, "native_corpus", [[1, 2, 9], [3, 4, 5, 6, 7, 8, 9], [2, 2, 9]],
            )
            config = _provider_config(
                seq_length=4, packing_stage="dataset", load_balance="native_batch_sampler",
                reset_position_ids=True, reset_attention_mask=True, eod_mask_loss=True,
                create_ltor_fields_in_dataloader=True,
            )
            datasets = build_indexed_text_dataset(
                data_path=prefix, data_config=config, tokenizer=_Tokenizer(),
                train_valid_test_num_samples=(8, 0, 0),
            )
            dataset = datasets[0]
            self.assertIsInstance(dataset, GPTDataset)
            self.assertTrue(np.any(dataset.sample_index[1:, 0] != dataset.sample_index[:-1, 0]))
            collate_fn = build_indexed_collate_fn()
            for sampler_type in ("single", "cyclic"):
                with self.subTest(sampler_type=sampler_type):
                    loaders, samplers = build_dataloader(
                        SimpleNamespace(dataloader_type=sampler_type), datasets=datasets, collate_fn=collate_fn,
                        mesh_context=SimpleNamespace(device_mesh=_StandaloneMesh(), dp_rank=0, dp_size=1),
                        training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                        data_config=config,
                    )
                    self.assertEqual(samplers, (None, None, None))
                    reference_sampler = build_dataset_batch_sampler(
                        total_samples=len(dataset), micro_batch_size=2, global_batch_size=4,
                        dp_rank=0, dp_world_size=1, sampler_type=sampler_type, seed=7,
                    )
                    for expected_indices in reference_sampler:
                        batch = next(loaders[0])
                        planned_indices = [key.dataset_index for key in loaders[0].last_plan.selected_keys]
                        self.assertCountEqual(planned_indices, expected_indices)
                        reference = collate_fn([dataset[index] for index in planned_indices])
                        self.assertEqual(set(batch), set(reference))
                        for field, expected in reference.items():
                            torch.testing.assert_close(batch[field], expected, rtol=0, atol=0)
                    with self.assertRaises(StopIteration):
                        next(loaders[0])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_worker_options_and_resume_with_real_indexed_files(self) -> None:
        """Feature: Plan-aware worker execution and checkpoint replay.
        Description: Read real indexed records with two persistent spawned workers and resume a new loader.
        Expectation: Worker options are forwarded and the resumed records match the original cursor.
        """
        with TemporaryDirectory() as directory:
            prefix = _write_indexed_source(directory, "workers", [[1, 2, 9], [3, 4, 9], [5, 6, 9]])
            config = _provider_config(seq_length=2, data_lazy_load=True)
            datasets = build_indexed_text_dataset(
                data_path=prefix, data_config=config, tokenizer=_Tokenizer(),
                train_valid_test_num_samples=(1, 0, 0),
            )
            options = {
                "num_workers": 2, "prefetch_factor": 2, "persistent_workers": True,
                "multiprocessing_context": "spawn", "timeout": 60,
            }
            with patch(
                "hyper_parallel.data.batching.build_dataloader.build_distributed_dataloader",
                wraps=build_distributed_dataloader,
            ) as builder:
                loader = _build_source_loader(datasets, config, **options)
            self.assertEqual(builder.call_args.kwargs["dataloader_kwargs"]["multiprocessing_context"], "spawn")
            self.assertEqual(builder.call_args.kwargs["dataloader_kwargs"]["timeout"], 60)
            self.assertEqual(next(loader)["input_ids"].tolist(), [[1, 2]])
            checkpoint = loader.state_dict()
            self.assertEqual(next(loader)["input_ids"].tolist(), [[3, 4]])
            loader.wait_for_prefetch()
            loader = _build_source_loader(datasets, config, **options)
            loader.load_state_dict(checkpoint)
            self.assertEqual(next(loader)["input_ids"].tolist(), [[3, 4]])
            self.assertEqual(next(loader)["input_ids"].tolist(), [[5, 6]])
            with self.assertRaises(StopIteration):
                next(loader)
            loader.wait_for_prefetch()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_shared_indexed_source_rejects_presharded_configuration(self) -> None:
        """Feature: Shared Indexed provider configuration.
        Description: Request rank-local sharding on the shared-index provider.
        Expectation: Construction rejects the incompatible sharding override.
        """
        config = _provider_config(distributed_dataloader={"dataset_already_sharded": True})
        with self.assertRaisesRegex(ValueError, "dataset_already_sharded"):
            _build_source_loader((SimpleNamespace(), None, None), config)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_indexed_source_requires_boundary_aware_attention(self) -> None:
        """Feature: Independent document attention.
        Description: Disable masks or request compressed attention without a boundary adapter.
        Expectation: Construction fails instead of mixing document attention.
        """
        for attention_mode, create_mask in (("dense", False), ("compressed", True)):
            with self.subTest(attention_mode=attention_mode), self.assertRaisesRegex(
                ValueError, "document-boundary attention"
            ):
                ParallelBatch(
                    mesh_context=None, device="cpu", tokenizer=_Tokenizer(),
                    data_config={"create_attention_mask_in_dataloader": create_mask},
                    source_type="indexed_source", pp_shared_data=False, attention_mode=attention_mode,
                )

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_trainer_uses_builtin_collator_without_user_callback(self) -> None:
        """Feature: Trainer packing callback selection.
        Description: Build the distributed Indexed collator contract with no custom callback.
        Expectation: Built-in callbacks are selected and gradient accumulation remains Trainer-owned.
        """
        trainer = TextTrainer.__new__(TextTrainer)
        trainer.base = SimpleNamespace(
            config=SimpleNamespace(
                dataloader=SimpleNamespace(collate_fn=None),
                dataset=SimpleNamespace(data_config=_provider_config()),
                training=SimpleNamespace(global_batch_size=32, micro_batch_size=2),
            ),
            mesh=SimpleNamespace(dp_size=8),
        )
        trainer._build_collate_fn()  # pylint: disable=protected-access
        self.assertEqual(trainer.base.num_micro_batches, 2)
        self.assertIsNone(trainer.base.collate_fn)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_lazy_weighted_sources_preserve_metadata_payload_alignment(self) -> None:
        """Feature: Weighted lazy Indexed sources.
        Description: Blend two unequal source corpora and compare metadata with materialized payloads.
        Expectation: Every source is covered and its planning token count matches its payload.
        """
        with TemporaryDirectory() as directory:
            prefixes = [
                _write_indexed_source(directory, "first", [[1, 9], [2, 9]]),
                _write_indexed_source(directory, "second", [[3, 4, 9], [5, 6, 9], [7, 8, 9], [10, 11, 9]]),
            ]
            dataset, _, _ = build_indexed_text_dataset(
                data_path=prefixes, data_config=_provider_config(data_lazy_load=True), tokenizer=_Tokenizer(),
                train_valid_test_num_samples=(1, 0, 0),
            )
            self.assertTrue(dataset.requires_distributed_packing)
            seen = set()
            for index, sample in enumerate(dataset):
                metadata = dataset.get_sample_metadata(index)
                self.assertEqual(metadata.pack_tokens, len(sample["input_ids"]))
                seen.add((sample["dataset_id"], sample["source_id"]))
            self.assertEqual(seen, {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_trainer_resumes_and_continues_across_dynamic_epochs(self) -> None:
        """Feature: Dynamic source epoch and Trainer resume.
        Description: Resume a double-buffer loader and continue across source epoch boundaries.
        Expectation: The restored next batch is preserved and training reaches the requested optimizer step.
        """
        with TemporaryDirectory() as directory:
            prefix = _write_indexed_source(directory, "corpus", [[1, 2, 9], [3, 4, 9], [5, 6, 9]])
            config = _provider_config(seq_length=2, distributed_dataloader={"double_buffer": True})
            datasets = build_indexed_text_dataset(
                data_path=prefix, data_config=config, tokenizer=_Tokenizer(),
                train_valid_test_num_samples=(1, 0, 0),
            )
            original = _build_source_loader(datasets, config)
            self.assertEqual(next(original)["input_ids"].tolist(), [[1, 2]])
            checkpoint = original.state_dict()
            original.wait_for_prefetch()
            loader = _build_source_loader(datasets, config)
            loader.load_state_dict(checkpoint)
            base = SimpleNamespace(
                config=SimpleNamespace(dataloader=SimpleNamespace(drop_last=True)),
                local_rank=0, state=SimpleNamespace(global_step=1, epoch=0),
                train_iters=5, train_epochs=1, train_steps=5, train_dataloader=loader,
                optimizer=Mock(), destroy_distributed=Mock(), on_train_begin=Mock(),
                on_train_end=Mock(), on_epoch_begin=Mock(), on_epoch_end=Mock(),
            )
            trainer = TextTrainer.__new__(TextTrainer)
            trainer.base = base
            delivered = []

            def train_step(iterator: object) -> None:
                """Consume one local batch as a CPU stand-in for an optimizer step."""
                delivered.append(next(iterator)["input_ids"].tolist())
                base.state.global_step += 1

            with patch.object(trainer, "train_step", side_effect=train_step), \
                    patch("hyper_parallel.trainer.text_trainer.synchronize"), \
                    patch("hyper_parallel.trainer.text_trainer.print_device_mem_info"):
                trainer.train()

            self.assertEqual(delivered, [[[3, 4]], [[5, 6]], [[1, 2]], [[3, 4]]])
            self.assertEqual(base.state.global_step, 5)
            self.assertEqual(base.state.epoch, 1)
            self.assertEqual(next(loader)["input_ids"].tolist(), [[5, 6]])
            loader.wait_for_prefetch()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_oversized_source_fails_before_payload_read(self) -> None:
        """Feature: First-version oversized policy.
        Description: Build a source containing more model tokens than one packed row.
        Expectation: Index-only validation rejects the source before binary data is read.
        """
        low_level_dataset = _LowLevelDataset([[1, 2, 3, 4, 9]])

        with self.assertRaisesRegex(ValueError, "cannot exceed sequence_length"):
            IndexedSourceDataset(
                low_level_dataset,
                "unused",
                np.asarray([0], dtype=np.int32),
                1,
                _Split.TRAIN,
                _source_config(sequence_length=3),
            )

        self.assertEqual(low_level_dataset.payload_reads, 0)
