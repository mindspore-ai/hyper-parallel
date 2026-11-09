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
"""Native HP sampling, occurrence conservation, and transactional replay."""

import copy
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.data.batching.build_dataloader import (
    DynamicBatchDataLoader,
    FixedBatchDataLoader,
    build_dataloader,
)
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.data.text.build_dataset import _build_dataloader_context
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.batch_sampler import BatchSamplerReader
from hyper_parallel.distributed_data.metadata import PlannedSampleLoader
from hyper_parallel.distributed_data.balancing_algorithm import (
    FeasibleLPTBalancingAlgorithm,
    LPTBalancingAlgorithm,
    resolve_balancing_algorithm,
)
from hyper_parallel.distributed_data.schema import SampleKey
from hyper_parallel.trainer.config.target import Target


build_dataloader_module = importlib.import_module("hyper_parallel.data.batching.build_dataloader")


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _TrackedDataset:
    """Return simple whole samples and record all physical reads."""

    def __init__(self, size: int = 10) -> None:
        """Initialize the read log and logical Dataset size."""
        self.size = size
        self.reads = []

    def __len__(self) -> int:
        """Return the logical Dataset size."""
        return self.size

    def __getitem__(self, index: int) -> dict:
        """Record and return one native sample."""
        self.reads.append(index)
        return {"id": index, "tokens": 1, "label": index + 1, "position": 7}


def _metadata(sample: dict) -> SampleMetadata:
    return SampleMetadata(pack_tokens=sample["tokens"], sample_id=sample["id"])


def _sampler(**overrides: object) -> object:
    options = {"total_samples": 10, "micro_batch_size": 2, "global_batch_size": 4, "dp_world_size": 1, "dp_rank": 0}
    options.update(overrides)
    return build_dataset_batch_sampler(**options)


def _loader(
        dataset: object, *, sampler: object = None, metadata_mode: bool = False,
        **options: object,
) -> object:
    """Build a standalone native sampler loader for online or precomputed metadata tests."""
    kwargs = {"metadata": [SampleMetadata(pack_tokens=1, sample_id=index) for index in range(len(dataset))]}
    if not metadata_mode:
        kwargs = {"metadata_fn": _metadata}
    loader = build_distributed_dataloader(
        dataset, _StandaloneMesh(),
        DistributedDatasetConfig(seq_len=16, local_batch_size=2, metadata_mode=metadata_mode, **options),
        batch_sampler=sampler if sampler is not None else _sampler(), **kwargs,
        device="cpu", cost_model=lambda metadata: metadata.cost,
    )
    return loader


def _batch_ids(batch: tuple) -> list[int]:
    return sorted(sample["id"] for sample in batch)


def _legacy_loader_state(state: dict, metadata_mode: bool) -> dict:
    """Build the former Reader-tree checkpoint from the shared native fields."""
    sampler = copy.deepcopy(state["_index_sampler_state"])
    payload = dict(state["dataset_state"], epoch=sampler["epoch"])
    reader = {"epoch": sampler["epoch"], "sampler": sampler,
              "sample_loader": None if metadata_mode else payload}
    return {
        "global_rank": 0, "epoch": sampler["epoch"], "step": sampler["consumed_samples"] // 2,
        "stopped": state["_iterator_finished"],
        "dataset_reader": None if metadata_mode else reader,
        "metadata_reader": reader if metadata_mode else None,
        "direct_sample_loader": payload if metadata_mode else None,
    }


class TestNativeBatchSampler(unittest.TestCase):
    """Test without process groups or accelerator hardware."""

    def test_fixed_membership_without_dynamic_selection(self) -> None:
        """Feature: Fixed native round membership.
        Description: Read short samples with online and offline metadata.
        Expectation: Each batch retains native indices and prefetch stays within the current and next windows.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                dataset = _TrackedDataset()
                loader = _loader(dataset, metadata_mode=metadata_mode)
                self.assertEqual(_batch_ids(next(loader)), [0, 1])
                loader.wait_for_prefetch()
                self.assertEqual(sorted(dataset.reads), list(range(4)))
                self.assertEqual(_batch_ids(next(loader)), [2, 3])
                loader.wait_for_prefetch()
                self.assertEqual(sorted(dataset.reads), list(range(6)))

    def test_native_mode_does_not_construct_dynamic_selector(self) -> None:
        """Feature: Native sampler ownership.
        Description: Build online and offline native loaders and consume a batch.
        Expectation: Both paths preserve selected indices without constructing a streaming selector.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                loader = _loader(_TrackedDataset(), metadata_mode=metadata_mode)
                self.assertEqual(_batch_ids(next(loader)), [0, 1])
                self.assertFalse(hasattr(loader, "_step_sample_selector"))

    def test_cyclic_order_and_epoch_remain_native(self) -> None:
        """Feature: Cyclic sampler epoch behavior.
        Description: Compare loader output with both native cyclic sharding policies across epochs.
        Expectation: Every round contains the same occurrences as the reference sampler.
        """
        for data_sharding in (False, True):
            with self.subTest(data_sharding=data_sharding):
                options = {"sampler_type": "cyclic", "data_sharding": data_sharding, "seed": 13}
                reference = _sampler(**options)
                loader = _loader(_TrackedDataset(), sampler=_sampler(**options))
                for epoch in (0, 1):
                    reference.set_epoch(epoch)
                    loader.set_epoch(epoch)
                    expected = [sorted(indices) for indices in reference]
                    self.assertEqual([_batch_ids(batch) for batch in loader], expected)

    def test_duplicate_indices_are_distinct_occurrences(self) -> None:
        """Feature: Independent sampling occurrences.
        Description: Reuse one metadata object for duplicate online or offline sample indices.
        Expectation: Readers bind distinct keys without mutating the supplied metadata.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                dataset = _TrackedDataset()
                shared_metadata = SampleMetadata(1, sample_id=3)
                options = {"metadata": [shared_metadata] * 10} if metadata_mode else {
                    "metadata_fn": lambda _, value=shared_metadata: value,
                }
                with build_distributed_dataloader(
                        dataset, _StandaloneMesh(),
                        DistributedDatasetConfig(seq_len=16, local_batch_size=2, metadata_mode=metadata_mode),
                        batch_sampler=_sampler(index_mapping=[3] * 10), **options,
                        device="cpu", cost_model=lambda metadata: metadata.cost,
                ) as loader:
                    reader = loader._planning_reader()
                    reader.prepare_next_step(1)
                    first, second = reader.metadata()
                    self.assertIsInstance(first, SampleMetadata)
                    self.assertIsNot(first, second)
                    self.assertIsNot(first, shared_metadata)
                    self.assertEqual((first.key, second.key), (SampleKey(0, 3, 0), SampleKey(0, 3, 1)))
                    self.assertEqual(_batch_ids(next(loader)), [3, 3])
                    loader.wait_for_prefetch()
                    self.assertEqual(dataset.reads, [3] * 4)
                self.assertIsNone(shared_metadata.key)

    def test_checkpoint_at_optimizer_boundary_excludes_prefetch(self) -> None:
        """Feature: Native sampler checkpoint replay.
        Description: Restore a complete step with or without legacy version fields in both metadata modes.
        Expectation: Remaining samples replay exactly, excluding speculative prefetch progress.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                sampler = _sampler()
                loader = _loader(_TrackedDataset(), sampler=sampler, metadata_mode=metadata_mode)
                self.assertEqual(_batch_ids(next(loader)), [0, 1])
                self.assertEqual(_batch_ids(next(loader)), [2, 3])
                loader.wait_for_prefetch()
                self.assertEqual(sampler.consumed_samples, 6)
                state = loader.state_dict()
                owner = "metadata_reader" if metadata_mode else "dataset_reader"
                self.assertEqual(state["_index_sampler_state"]["consumed_samples"], 4)
                self.assertNotIn(owner, state)
                self.assertNotIn("version", state["dataset_state"])
                original = [_batch_ids(batch) for batch in loader]
                for reader_key, legacy_version in (
                        ("metadata_reader", None), ("metadata_reader", 0), ("sidecar_reader", 1),
                ):
                    with self.subTest(reader_key=reader_key, legacy_version=legacy_version):
                        restored_state = _legacy_loader_state(state, metadata_mode)
                        if legacy_version is not None:
                            restored_state[owner]["version"] = legacy_version
                            payload_state = (
                                restored_state["direct_sample_loader"] if metadata_mode
                                else restored_state[owner]["sample_loader"]
                            )
                            payload_state["version"] = legacy_version
                        restored_state[reader_key] = restored_state.pop("metadata_reader")
                        resumed = _loader(_TrackedDataset(), metadata_mode=metadata_mode)
                        resumed.load_state_dict(restored_state)
                        self.assertEqual([_batch_ids(batch) for batch in resumed], original)
                self.assertEqual(original[0], [4, 5])

    def test_restore_uses_current_sampler_seed(self) -> None:
        """Feature: Native sampler checkpoint replay.
        Description: Restore the committed cursor with a different sampler seed.
        Expectation: Both metadata modes follow the current sampler's remaining order.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                with _loader(
                        _TrackedDataset(), sampler=_sampler(sampler_type="cyclic", seed=13),
                        metadata_mode=metadata_mode,
                ) as loader:
                    next(loader)
                    state = loader.state_dict()
                reference = _sampler(sampler_type="cyclic", seed=14)
                reference.enable_source_batch_resume()
                reference.load_state_dict(state["_index_sampler_state"])
                with _loader(
                        _TrackedDataset(), sampler=_sampler(sampler_type="cyclic", seed=14),
                        metadata_mode=metadata_mode,
                ) as resumed:
                    resumed.load_state_dict(state)
                    self.assertEqual(
                        [_batch_ids(batch) for batch in resumed], [sorted(indices) for indices in reference],
                        "Restored progress must follow the current sampler's seed.",
                    )

    def test_restore_uses_current_tensor_index_mapping(self) -> None:
        """Feature: Native sampler index mapping.
        Description: Restore the committed cursor with a different tensor mapping.
        Expectation: Both metadata modes fetch the current mapping's remaining samples.
        """
        for metadata_mode in (False, True):
            with self.subTest(metadata_mode=metadata_mode):
                with _loader(
                        _TrackedDataset(), sampler=_sampler(index_mapping=torch.tensor([3] * 10)),
                        metadata_mode=metadata_mode,
                ) as first:
                    self.assertEqual(_batch_ids(next(first)), [3, 3])
                    self.assertEqual(_batch_ids(next(first)), [3, 3])
                    state = first.state_dict()
                with _loader(
                        _TrackedDataset(), sampler=_sampler(index_mapping=torch.tensor([4] * 10)),
                        metadata_mode=metadata_mode,
                ) as resumed:
                    resumed.load_state_dict(state)
                    self.assertEqual([_batch_ids(batch) for batch in resumed], [[4, 4]] * 3)

    def test_existing_native_cursor_is_respected(self) -> None:
        """Feature: Restored native sampler progress.
        Description: Build the loader with a sampler that already has a cursor and epoch.
        Expectation: Reading begins at the restored cursor and preserves the epoch.
        """
        sampler = _sampler(consumed_samples=4)
        sampler.epoch = 2
        loader = _loader(_TrackedDataset(), sampler=sampler)
        self.assertEqual(_batch_ids(next(loader)), [4, 5])
        self.assertEqual(loader.state_dict()["_index_sampler_state"]["epoch"], 2)

    def test_metadata_failure_does_not_look_like_exhaustion(self) -> None:
        """Feature: Metadata callback error reporting.
        Description: Raise StopIteration while extracting sample metadata.
        Expectation: The loader reports a RuntimeError instead of treating the failure as EOF.
        """
        def broken_metadata(sample: object) -> SampleMetadata:
            """Raise an accidental StopIteration outside sampler advancement."""
            del sample
            raise StopIteration("invalid metadata callback")

        loader = build_distributed_dataloader(
            _TrackedDataset(), _StandaloneMesh(), DistributedDatasetConfig(seq_len=16, local_batch_size=2),
            batch_sampler=_sampler(), metadata_fn=broken_metadata,
            device="cpu", cost_model=lambda metadata: metadata.cost,
        )
        with self.assertRaisesRegex(RuntimeError, "invalid metadata callback"):
            next(loader)

    def test_callback_errors_propagate_without_wrapping(self) -> None:
        """Feature: Direct callback error propagation.
        Description: Fail sample metadata extraction with a ValueError.
        Expectation: The original exception is raised without advancing committed sampler progress.
        """
        failure = ValueError("metadata unavailable")
        loader = build_distributed_dataloader(
            _TrackedDataset(), _StandaloneMesh(),
            DistributedDatasetConfig(seq_len=16, local_batch_size=2),
            batch_sampler=_sampler(), metadata_fn=_metadata,
            device="cpu", cost_model=lambda metadata: metadata.cost,
        )
        with loader, patch.object(loader._dataset_reader, "_metadata_fn", side_effect=failure):
            with self.assertRaises(ValueError) as caught:
                next(loader)
        self.assertIs(caught.exception, failure)
        self.assertEqual(loader.state_dict()["_index_sampler_state"]["consumed_samples"], 0)

    def test_invalid_native_sampler_options_fail_at_build(self) -> None:
        """Feature: Native sampler input validation.
        Description: Build with incompatible sampler interfaces, DP parameters, or cursors.
        Expectation: Invalid inputs raise directly before data iteration.
        """
        with self.assertRaisesRegex(ValueError, "owns DP slicing"):
            _loader(_TrackedDataset(), dataset_already_sharded=True)
        with self.assertRaisesRegex(ValueError, "micro_batch_size"):
            _loader(_TrackedDataset(), sampler=_sampler(micro_batch_size=1))
        with self.assertRaisesRegex(ValueError, "drop_last"):
            _loader(_TrackedDataset(), sampler=_sampler(drop_last=False))
        for name, value in (("dp_rank", 1), ("dp_world_size", 2), ("consumed_samples", 1), ("set_epoch", None)):
            with self.subTest(field=name):
                sampler = _sampler()
                setattr(sampler, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    _loader(_TrackedDataset(), sampler=sampler)
        with self.assertRaisesRegex(ValueError, "pack_fn must be omitted"):
            build_distributed_dataloader(
                _TrackedDataset(), _StandaloneMesh(), DistributedDatasetConfig(seq_len=16, local_batch_size=2),
                batch_sampler=_sampler(), metadata_fn=_metadata, pack_fn=lambda samples, _: samples,
            )

    def test_metadata_reader_advances_only_on_complete_commit(self) -> None:
        """Feature: Transactional sampler progress.
        Description: Reject empty or partial commits, repeat preparation, and restore committed progress.
        Expectation: Saved progress advances only after the full batch is consumed.
        """
        sampler = _sampler()
        reader = BatchSamplerReader(
            sampler, reader_rank=0,
            metadata=[SampleMetadata(1)] * 10, metadata_fn=None, sample_loader=None,
        )
        with self.assertRaisesRegex(ValueError, "complete pending"):
            reader.commit(set())
        self.assertIsNone(reader.prepare_next_step(1))
        self.assertIsNone(reader.prepare_next_step(1))
        self.assertEqual(sampler.consumed_samples, 2)
        saved = copy.deepcopy(reader.state_dict())
        self.assertEqual(saved["sampler"]["consumed_samples"], 0)
        with self.assertRaisesRegex(ValueError, "complete pending"):
            reader.commit({reader.metadata()[0].key})
        reader.commit({item.key for item in reader.metadata()})
        self.assertEqual(reader.state_dict()["sampler"]["consumed_samples"], 2)
        with self.assertRaisesRegex(ValueError, "complete pending"):
            reader.commit(set())
        self.assertIsNone(reader.prepare_next_step(1))
        self.assertEqual(sampler.consumed_samples, 4)
        self.assertEqual(reader.state_dict()["sampler"]["consumed_samples"], 2)
        reader.load_state_dict({**saved, "version": 0})
        self.assertEqual(reader.state_dict(), saved)
        self.assertIsNone(reader.prepare_next_step(1))
        self.assertEqual([item.key.dataset_index for item in reader.metadata()], [0, 1])

    def test_reader_uses_explicit_window_size_without_lookahead(self) -> None:
        """Feature: Caller-owned source window sizes.
        Description: Request three then two rounds, followed by a partial epoch tail.
        Expectation: Reads honor each request and committing a window does not start another read.
        """
        sampler = _sampler(total_samples=12, global_batch_size=10)
        reader = BatchSamplerReader(
            sampler, reader_rank=0,
            metadata=[SampleMetadata(1)] * 12, metadata_fn=None, sample_loader=None,
        )
        for requested, start, end in ((3, 0, 6), (2, 6, 10), (3, 10, 12)):
            with self.subTest(requested=requested, start=start):
                reader.prepare_next_step(requested)
                keys = [item.key for item in reader.metadata()]
                self.assertEqual([key.dataset_index for key in keys], list(range(start, end)))
                self.assertEqual(sampler.consumed_samples, end)
                for offset in range(0, len(keys), 2):
                    reader.commit(set(keys[offset:offset + 2]))
                self.assertEqual(reader.buffer_size, 0)
                self.assertEqual(sampler.consumed_samples, end)
        reader.prepare_next_step(3)
        self.assertTrue(reader.exhausted)
        self.assertEqual(reader.metadata(), ())
        reader.close()

    def test_reader_rejects_invalid_window_sizes_before_reading(self) -> None:
        """Feature: Explicit source window validation.
        Description: Request invalid counts before reading the first window.
        Expectation: Each request fails without advancing the sampler or buffering data.
        """
        sampler = _sampler()
        reader = BatchSamplerReader(
            sampler, reader_rank=0,
            metadata=[SampleMetadata(1)] * 10, metadata_fn=None, sample_loader=None,
        )
        for count in (0, -1, True, 1.5, None):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                reader.prepare_next_step(count)
        self.assertEqual(sampler.consumed_samples, 0)
        self.assertEqual(reader.metadata(), ())
        reader.close()

    def test_reader_retains_original_groups_through_split_commit_and_replay(self) -> None:
        """Feature: Original source grouping.
        Description: Read repeated indices with online, offline and split metadata, then commit and restore.
        Expectation: Every source pack retains its document keys and replay reproduces the same layout.
        """
        indices = (3, 3, 1, 0, 2, 3)
        for metadata_mode, split in ((False, False), (True, False), (False, True)):
            with self.subTest(metadata_mode=metadata_mode, split=split):
                dataset = _TrackedDataset(size=6)
                sample_loader = None if metadata_mode else PlannedSampleLoader(
                    dataset, num_workers=0, pin_memory=False, prefetch_factor=None, persistent_workers=False, seed=7,
                )
                reader = BatchSamplerReader(
                    _sampler(total_samples=6, global_batch_size=6, index_mapping=indices), reader_rank=0,
                    metadata=[SampleMetadata(1)] * 6 if metadata_mode else None,
                    metadata_fn=None if metadata_mode else _metadata, sample_loader=sample_loader,
                    split_fn=(lambda sample: [sample] * (sample["id"] + 1)) if split else None,
                )
                try:
                    saved = reader.state_dict()
                    reader.prepare_next_step(3)
                    planning_input = reader.planning_input()
                    expected = tuple(tuple(
                        tuple(SampleKey(0, indices[position], position, sequence)
                              for sequence in range(indices[position] + 1 if split else 1))
                        for position in range(start, start + 2)
                    ) for start in (0, 2, 4))
                    self.assertEqual(planning_input.original_batches, expected)
                    self.assertEqual(tuple(item.key for item in planning_input.metadata), tuple(
                        key for batch in expected for group in batch for key in group
                    ))
                    self.assertEqual(dataset.reads, [] if metadata_mode else list(indices))
                    reader.commit({key for group in expected[0] for key in group})
                    self.assertEqual(reader.planning_input().original_batches, expected)
                    self.assertEqual(reader.state_dict()["sampler"]["consumed_samples"], 0)
                    for batch in expected[1:]:
                        reader.commit({key for group in batch for key in group})
                    self.assertEqual(reader.planning_input().original_batches, ())
                    self.assertEqual(reader.state_dict()["sampler"]["consumed_samples"], 6)
                    reader.load_state_dict(saved)
                    reader.prepare_next_step(3)
                    self.assertEqual(reader.planning_input(), planning_input)
                finally:
                    reader.close()

    def test_trainer_builder_reuses_sampler_and_collator(self) -> None:
        """Feature: Trainer balancing enablement.
        Description: Enable balancing with a configured collator and repeated index mapping.
        Expectation: The collective loader retains both and owns sampler checkpoint progress.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        target = SimpleNamespace(load_balance={"balance_window_size": 3}, data_rearrange_map=[3] * 10)
        collator = object()
        with patch.object(build_dataloader_module, "build_distributed_dataloader") as build:
            loaders, samplers = build_dataloader(
                target, datasets=(_TrackedDataset(), None, None), collate_fn=collator,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config={"seq_length": 16},
            )
        self.assertIs(loaders[0], build.return_value)
        self.assertEqual(build.call_args.args[2].balance_window_size, 3)
        self.assertEqual(samplers, (None, None, None))
        self.assertIs(build.call_args.kwargs["collate_fn"], collator)
        self.assertEqual(next(iter(build.call_args.kwargs["batch_sampler"])), [3, 3])

    def test_trainer_forwards_balance_logging_switch(self) -> None:
        """Feature: Trainer repacking, balancing gain and log configuration.
        Description: Omit or set load_balance options, then try invalid values and size overrides.
        Expectation: The loader receives public config fields unchanged and rejects invalid overrides.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        target = SimpleNamespace(dataloader_type="single")

        def build(options: dict) -> object:
            """Build the Trainer data path with the supplied logging options.

            Args:
                options: Parameters in dataloader.load_balance.
            """
            target.load_balance = options
            return build_dataloader(
                target, datasets=(_TrackedDataset(), None, None), mesh_context=context, collate_fn=tuple,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config={"seq_length": 16},
            )

        with patch.object(build_dataloader_module, "build_distributed_dataloader") as loader_builder:
            for options in (
                {}, {"log_balance": False}, {"log_balance": True},
                {"repack_sequences": True, "pad_token_id": 151645, "log_balance": True},
                {"device_prefetch_depth": 0}, {"device_prefetch_depth": 2},
                {"min_balance_gain": 0.0}, {"min_balance_gain": 0.1}, {"min_balance_gain": 0.99},
            ):
                with self.subTest(options=options):
                    build(options)
                    config = loader_builder.call_args.args[2]
                    self.assertIs(config.log_balance, options.get("log_balance", False))
                    self.assertIs(config.repack_sequences, options.get("repack_sequences", False))
                    self.assertEqual(config.pad_token_id, options.get("pad_token_id"))
                    self.assertEqual(config.device_prefetch_depth, options.get("device_prefetch_depth", 1))
                    self.assertEqual(config.min_balance_gain, options.get("min_balance_gain", 0.0))
                    expected_collator = None if config.repack_sequences else tuple
                    self.assertIs(loader_builder.call_args.kwargs["collate_fn"], expected_collator)
            with self.assertRaisesRegex(ValueError, "log_balance must be boolean"):
                build({"log_balance": "true"})
            with self.assertRaisesRegex(ValueError, "device_prefetch_depth must be a non-negative integer"):
                build({"device_prefetch_depth": -1})
            for gain in (-0.1, 1.0, True):
                with self.subTest(gain=gain), self.assertRaisesRegex(ValueError, "min_balance_gain must be"):
                    build({"min_balance_gain": gain})
            for name, value in (("seq_len", 8), ("local_batch_size", 2), ("num_workers", 1), ("pin_memory", True)):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"cannot override.*{name}"):
                    build({name: value})

    def test_trainer_balance_disabled_preserves_native_loader(self) -> None:
        """Feature: Disabled balancing.
        Description: Construct the Trainer loader without a balancing block.
        Expectation: Native DataLoader construction, sampling and collation are preserved.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1)
        target = Target(FixedBatchDataLoader, target_path="hyper_parallel.data.batching.FixedBatchDataLoader")
        for data_config in (None, {}):
            with self.subTest(data_config=data_config):
                loaders, samplers = build_dataloader(
                    target,
                    datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                    mesh_context=context,
                    training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                    data_config=data_config,
                )
                self.assertIsInstance(loaders[0], FixedBatchDataLoader)
                self.assertIs(loaders[0].batch_sampler, samplers[0])
                self.assertEqual(_batch_ids(next(iter(loaders[0]))), [0, 1])

    def test_trainer_defaults_to_feasibility_repair(self) -> None:
        """Feature: Default feasibility-repair selection.
        Description: Build a balanced loader without an explicit assignment policy.
        Expectation: The shared default policy includes bounded feasibility repair.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        with patch.object(build_dataloader_module, "build_distributed_dataloader") as build:
            build_dataloader(
                SimpleNamespace(load_balance={}), datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=7),
                data_config={"seq_length": 16},
            )
            algorithm = resolve_balancing_algorithm(build.call_args.kwargs["balancing_algorithm"])
            self.assertIs(type(algorithm), FeasibleLPTBalancingAlgorithm)

    def test_trainer_preserves_explicit_balancing_policies(self) -> None:
        """Feature: Explicit assignment policy selection.
        Description: Supply a policy directly or through a configured target.
        Expectation: The explicit policy overrides the repair default without being replaced.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        options = {
            "datasets": (_TrackedDataset(), None, None), "collate_fn": tuple, "mesh_context": context,
            "training_config": SimpleNamespace(micro_batch_size=1, global_batch_size=1, seed=7),
        }
        policy = LPTBalancingAlgorithm(objective="balance")
        for configured in (False, True):
            target = SimpleNamespace(balancing_algorithm=SimpleNamespace(build=Mock(return_value=policy))) if (
                configured
            ) else SimpleNamespace()
            target.load_balance = {}
            supplied = None if configured else policy
            with self.subTest(configured=configured), patch.object(
                    build_dataloader_module, "build_distributed_dataloader",
            ) as build:
                build_dataloader(
                    target, **options, data_config={"seq_length": 16},
                    balancing_algorithm=supplied,
                )
                self.assertIs(resolve_balancing_algorithm(build.call_args.kwargs["balancing_algorithm"]), policy)

    def test_dataset_context_preserves_cache_options(self) -> None:
        """Feature: Standalone Dataset ownership.
        Description: Derive native ownership when no Trainer context is supplied.
        Expectation: Dataset cache options survive without a separate participation flag.
        """
        with patch("hyper_parallel.data.text.build_dataset.create_dataloader_parallel_context") as build:
            context = _build_dataloader_context(object(), {"data_index_cache": True, "no_shared_storage": True})
            self.assertIs(context, build.return_value)
            self.assertTrue(build.call_args.kwargs["data_index_cache"])
            self.assertFalse(build.call_args.kwargs["shared_storage"])

    def test_balance_block_rejects_scalars_and_old_dataset_options(self) -> None:
        """Feature: Balancing configuration migration.
        Description: Supply scalar blocks or obsolete Dataset balancing options.
        Expectation: Invalid inputs report the dataloader.load_balance location before building a loader.
        """
        cases = [(SimpleNamespace(load_balance=value), {}) for value in (True, False, "true", 0, 1, [])]
        cases.extend((SimpleNamespace(load_balance={}), {name: value}) for name, value in (
            ("load_balance", True), ("distributed_dataloader", {}), ("repack_sequences", True),
            ("balance_window_size", 3), ("min_balance_gain", 0.1),
        ))
        for target, data_config in cases:
            with self.subTest(target=target, data_config=data_config):
                with self.assertRaisesRegex(ValueError, "dataloader.load_balance"):
                    build_dataloader(
                        target, datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                        mesh_context=SimpleNamespace(dp_rank=0, dp_size=1),
                        training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                        data_config=data_config,
                    )

    def test_trainer_uses_dataset_metadata_for_direct_reads(self) -> None:
        """Feature: Trainer offline metadata loading.
        Description: Consume a Dataset through the Trainer builder with indexed metadata.
        Expectation: Payloads are read once after planning without payload exchange.
        """
        dataset = _TrackedDataset()
        dataset.metadata = [SampleMetadata(1, sample_id=index) for index in range(len(dataset))]
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        dataloaders, _ = build_dataloader(
            SimpleNamespace(load_balance={"metadata_mode": True, "balance_window_size": 2}),
            datasets=(dataset, None, None), collate_fn=tuple, mesh_context=context,
            training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
            data_config={"seq_length": 16},
            cost_model=lambda metadata: metadata.cost,
        )
        self.assertEqual(dataset.reads, [])
        with dataloaders[0] as loader, patch(
                "hyper_parallel.distributed_data.transport.DataPlaneTransport.exchange_prepared",
                side_effect=AssertionError("Offline metadata must bypass payload exchange"),
        ):
            batches = list(loader)
        self.assertEqual([sample["id"] for batch in batches for sample in batch], list(range(len(dataset))))
        self.assertEqual(dataset.reads, list(range(len(dataset))))

    def test_trainer_online_mode_ignores_dataset_metadata(self) -> None:
        """Feature: Online metadata isolation.
        Description: Supply an unrelated Dataset metadata attribute in online mode.
        Expectation: Online loading still derives metadata from the actual payload.
        """
        dataset = _TrackedDataset()
        dataset.metadata = object()
        dataloaders, _ = build_dataloader(
            SimpleNamespace(load_balance={}), datasets=(dataset, None, None), collate_fn=tuple,
            mesh_context=SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1),
            training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
            data_config={"seq_length": 16},
            metadata_fn=_metadata, cost_model=lambda metadata: metadata.cost,
        )
        with dataloaders[0] as loader:
            self.assertEqual(_batch_ids(next(loader)), [0, 1])

    def test_trainer_offline_mode_requires_explicit_metadata(self) -> None:
        """Feature: Explicit offline metadata source.
        Description: Enable offline mode without supplying indexed metadata.
        Expectation: The Trainer builder propagates the missing-metadata error.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        target = SimpleNamespace(load_balance={"metadata_mode": True})
        data_config = {"seq_length": 16}
        with self.assertRaisesRegex(ValueError, "metadata_mode=True requires metadata"):
            build_dataloader(
                target, datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config=data_config,
            )

    def test_trainer_rejects_dynamic_batch_selection(self) -> None:
        """Feature: Native selection mode validation.
        Description: Enable balancing with a second dynamic batch selector.
        Expectation: Competing sample selection is rejected during construction.
        """
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=_StandaloneMesh(), pp_size=1)
        target = SimpleNamespace(_target_=DynamicBatchDataLoader, load_balance={})
        with self.assertRaisesRegex(ValueError, "dynamic batching"):
            build_dataloader(
                target, datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config={"seq_length": 16},
            )

    def test_trainer_uses_rank_local_payload_device(self) -> None:
        """Feature: Rank-local communication device selection.
        Description: Build a CUDA loader while the current device is rank-local device 3.
        Expectation: The loader receives cuda:3 and uses the default Gloo data transport.
        """
        mesh = _StandaloneMesh()
        mesh.device_type = "cuda"
        context = SimpleNamespace(dp_rank=0, dp_size=1, device_mesh=mesh, pp_size=1)
        with (
            patch.object(torch.cuda, "current_device", return_value=3),
            patch.object(build_dataloader_module, "build_distributed_dataloader") as build,
        ):
            build_dataloader(
                SimpleNamespace(load_balance={}), datasets=(_TrackedDataset(), None, None), collate_fn=tuple,
                mesh_context=context,
                training_config=SimpleNamespace(micro_batch_size=2, global_batch_size=4, seed=7),
                data_config={"seq_length": 16},
            )
        self.assertEqual(str(build.call_args.kwargs["device"]), "cuda:3")
        self.assertEqual(build.call_args.args[2].communication_backend, "gloo")
