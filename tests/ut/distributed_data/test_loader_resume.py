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
"""Checkpoint interoperability for native fixed and balanced data loading."""

import copy
import random
import tempfile
import unittest
from contextlib import ExitStack
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer
from hyper_parallel.data.batching.build_dataloader import FixedBatchDataLoader
from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.topology import DataTopology
from tests.ut.distributed_data.test_batch_sampler import _legacy_loader_state


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _StatefulSamples(list):
    """Keep user Dataset state separate from the loader's compatibility fields."""

    def state_dict(self) -> dict[str, int]:
        """Return user fields that deliberately differ from the Reader settings."""
        return {"dataset_size": 999, "seed": 777}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Reject any corruption of the Dataset's own checkpoint payload."""
        if state != self.state_dict():
            raise ValueError("User Dataset state changed")


def _metadata(sample: dict[str, int]) -> SampleMetadata:
    """Assign the same cost to each source occurrence."""
    return SampleMetadata(pack_tokens=1, sample_id=sample["id"])


def _build_loader(mode: str, *, consumed: int = 0, epoch: int = 0, workers: int = 0,
                  sampler_type: str = "single", data_sharding: bool = False,
                  samples: list[dict[str, int]] | None = None,
                  dp_world_size: int = 1, dp_rank: int = 0,
                  global_batch_size: int = 4, micro_batch_size: int | None = None,
                  ) -> FixedBatchDataLoader | DistributedDataLoader:
    """Build native loaders, optionally repartitioning the fixed loader."""
    samples = [{"id": index} for index in range(24)] if samples is None else samples
    micro_batch_size = min(2, global_batch_size // dp_world_size) if micro_batch_size is None else micro_batch_size
    sampler = build_dataset_batch_sampler(
        total_samples=len(samples), consumed_samples=consumed, micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size, dp_world_size=dp_world_size, dp_rank=dp_rank, sampler_type=sampler_type,
        data_sharding=data_sharding, seed=42,
    )
    sampler.load_state_dict(dict(sampler.state_dict(), epoch=epoch))
    if mode == "fixed":
        return FixedBatchDataLoader(samples, batch_sampler=sampler, collate_fn=list,
                                    num_workers=workers, sampler_type=sampler_type)
    metadata_mode = mode == "metadata"
    options = {"metadata": [_metadata(sample) for sample in samples]} if metadata_mode else {"metadata_fn": _metadata}
    return build_distributed_dataloader(
        samples, _StandaloneMesh(),
        DistributedDatasetConfig(seq_len=16, local_batch_size=micro_batch_size,
                                 balance_window_size=3, metadata_mode=metadata_mode),
        batch_sampler=sampler, device="cpu", cost_model=lambda item: item.cost,
        dataloader_kwargs={"num_workers": workers}, **options,
    )


class TestLoaderResume(unittest.TestCase):
    """Exercise delivered cursors independently of prefetch and saved schemas."""

    def test_gbs_change_preserves_absolute_cursor_and_new_step_boundaries(self) -> None:
        """Feature: Variable-GBS checkpoint restoration.
        Description: Consume thirty samples at GBS ten, then restore with GBS eight and twelve.
        Expectation: All loader modes replay exact suffixes and retain complete-step boundaries on repeated saves.
        """
        # Cover each loader as source and target without a costly worker-start Cartesian product.
        cases = (
            ("fixed", "fixed", "single", 1),
            ("fixed", "online", "single", 0),
            ("online", "metadata", "cyclic", 0),
            ("metadata", "fixed", "cyclic", 1),
        )
        for source_mode, target_mode, sampler_type, workers in cases:
            with self.subTest(source=source_mode, target=target_mode, sampler=sampler_type, workers=workers):
                self._check_gbs_change(source_mode, target_mode, sampler_type, workers)

    def _check_gbs_change(self, source_mode: str, target_mode: str, sampler_type: str, workers: int) -> None:
        """Check partial windows, repeated restoration and a second GBS change."""
        samples = [{"id": index} for index in range(70)]
        order = list(range(70))
        if sampler_type == "cyclic":
            random.Random(44).shuffle(order)
        options = {"samples": samples, "sampler_type": sampler_type, "workers": workers}
        with ExitStack() as stack:
            source = _build_loader(source_mode, global_batch_size=10, epoch=2, **options)
            if source_mode != "fixed":
                stack.enter_context(source)
            iterator = iter(source)
            for _ in range(15):
                next(iterator)
            saved = source.state_dict()
            self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 30)
            list(iterator)
            target = _build_loader(target_mode, global_batch_size=8, **options)
            if target_mode != "fixed":
                stack.enter_context(target)
            original = copy.deepcopy(saved)
            target.load_state_dict(saved)
            self.assertEqual(saved, original)
            iterator = iter(target)
            selected = []
            for index in range(4):
                selected.extend(sample["id"] for sample in next(iterator))
                if target_mode != "fixed":
                    self.assertEqual(target.state_dict()["_index_sampler_state"]["consumed_samples"],
                                     38 if index == 3 else 30)
            self.assertEqual(sorted(selected), sorted(order[30:38]))
            saved = target.state_dict()
            self.assertEqual(saved["_index_sampler_state"], {
                "consumed_samples": 38, "epoch": 2, "global_batch_size": 8, "dp_world_size": 1,
                "optimizer_step_offset": 6,
            })
            list(iterator)
            for gbs in (8, 12):
                restored = _build_loader(target_mode, global_batch_size=gbs, **options)
                if target_mode != "fixed":
                    stack.enter_context(restored)
                restored.load_state_dict(saved)
                iterator = iter(restored)
                selected = [sample["id"] for _ in range(gbs // 2) for sample in next(iterator)]
                self.assertEqual(sorted(selected), sorted(order[38:38 + gbs]))
                self.assertEqual(restored.state_dict()["_index_sampler_state"]["consumed_samples"], 38 + gbs)
                self.assertEqual(restored.state_dict()["_index_sampler_state"]["optimizer_step_offset"], 38 % gbs)
                list(iterator)
                restored.set_epoch(3)
                iterator = iter(restored)
                next(iterator)
                self.assertNotIn("optimizer_step_offset", restored.state_dict()["_index_sampler_state"])
                list(iterator)

    def test_gbs_and_dp_change_repartition_unaligned_cursor(self) -> None:
        """Feature: Joint GBS and DP migration.
        Description: Restore cursor thirty at GBS eight and DP four, with a short dataset tail.
        Expectation: Every rank yields the same batch count and the global suffix contains no skips or duplicates.
        """
        for sampler_type in ("single", "cyclic"):
            with self.subTest(sampler_type=sampler_type):
                samples = [{"id": index} for index in range(65)]
                order = list(range(65))
                if sampler_type == "cyclic":
                    random.Random(44).shuffle(order)
                source = _build_loader("fixed", consumed=30, epoch=2, global_batch_size=10,
                                       samples=samples, sampler_type=sampler_type)
                saved = source.state_dict()
                selected = []
                for rank in range(4):
                    target = _build_loader("fixed", global_batch_size=8, dp_world_size=4, dp_rank=rank,
                                           samples=samples, sampler_type=sampler_type)
                    target.load_state_dict(saved)
                    batches = list(target)
                    self.assertEqual(len(batches), 4)
                    selected.extend(sample["id"] for batch in batches for sample in batch)
                self.assertEqual(sorted(selected), sorted(order[30:62]))

    def test_legacy_step_counter_uses_source_gbs(self) -> None:
        """Feature: Legacy complete-step cursors with changed GBS.
        Description: Restore old micro-step fifteen at cursor thirty from GBS ten into GBS eight.
        Expectation: The old counter is not interpreted as a partial target optimizer step.
        """
        for mode in ("online", "metadata"):
            options = {"samples": [{"id": index} for index in range(70)]}
            with self.subTest(mode=mode), _build_loader(mode, consumed=30, global_batch_size=10, **options) as source:
                saved = _legacy_loader_state(source.state_dict(), mode == "metadata")
            with _build_loader(mode, global_batch_size=8, **options) as target:
                target.load_state_dict(saved)
                self.assertEqual(target._step, 0)
                selected = [sample["id"] for _ in range(4) for sample in next(target)]
                self.assertEqual(sorted(selected), list(range(30, 38)))
                self.assertEqual(target.state_dict()["_index_sampler_state"]["consumed_samples"], 38)

    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier")
    def test_dcp_keeps_optimizer_step_offset(self, *_mocks) -> None:
        """Feature: Persistent boundaries after a GBS change.
        Description: Save offset cursors with native worker snapshots through embedded and per-rank DCP.
        Expectation: Restored GBS-eight steps consume samples 38 through 45 and preserve the training step.
        """
        for mode, workers, per_rank in product(("fixed", "online", "metadata"), (0, 1), (False, True)):
            with self.subTest(mode=mode, workers=workers, per_rank=per_rank), ExitStack() as stack:
                options = {"global_batch_size": 8, "samples": [{"id": index} for index in range(70)], "workers": workers}
                source, target = _build_loader(mode, **options), _build_loader(mode, **options)
                if mode != "fixed":
                    stack.enter_context(source)
                    stack.enter_context(target)
                source.load_state_dict({"_index_sampler_state": {
                    "consumed_samples": 30, "epoch": 2, "global_batch_size": 10, "dp_world_size": 1,
                }})
                if mode == "fixed":
                    source.snapshot_every_n_steps = 3
                iterator = iter(source)
                for _ in range(4):
                    next(iterator)
                state = source.state_dict()
                list(iterator)
                checkpointer = DistributedCheckpointer(extra_state_per_rank=per_rank)
                path = stack.enter_context(tempfile.TemporaryDirectory())
                checkpointer.save(path, {"model": {"w": torch.ones(1)}, "extra_state": {
                    "global_step": 4, "train_dataloader": state,
                }}, global_step=4, save_async=False)
                restored = {"model": {"w": torch.zeros(1)}}
                checkpointer.load(path, restored, extra_state_skeleton={
                    "global_step": 0, "train_dataloader": target.state_dict_for_load(),
                })
                self.assertEqual(restored["extra_state"]["global_step"], 4)
                target.load_state_dict(restored["extra_state"]["train_dataloader"])
                iterator = iter(target)
                selected = [sample["id"] for _ in range(4) for sample in next(iterator)]
                self.assertEqual(sorted(selected), list(range(38, 46)))
                self.assertEqual(target.state_dict()["_index_sampler_state"]["consumed_samples"], 46)
                list(iterator)

    def test_plain_torch_sampler_keeps_native_resume_state(self) -> None:
        """Feature: Ordinary torchdata loader compatibility.
        Description: Restore a fixed loader using the default PyTorch batch sampler.
        Expectation: No native HP cursor is injected and the next batch resumes without repetition.
        """
        source = FixedBatchDataLoader(list(range(12)), batch_size=2, collate_fn=list)
        target = FixedBatchDataLoader(list(range(12)), batch_size=2, collate_fn=list)
        next(iter(source))
        saved = source.state_dict()
        template = source.state_dict_for_load()
        self.assertEqual(template, saved)
        target.load_state_dict(template)
        self.assertEqual(next(iter(target)), [2, 3])

    def test_legacy_balanced_cursor_retains_nonzero_start(self) -> None:
        """Feature: Legacy balanced checkpoint migration.
        Description: Resume nested or duplicated legacy cursors, then switch to fixed loading.
        Expectation: Both readers continue from sample eight without repeating data.
        """
        for mode, duplicated in product(("online", "metadata"), (False, True)):
            with self.subTest(mode=mode, duplicated=duplicated), _build_loader(mode, consumed=4) as source:
                next(source)
                next(source)
                saved = _legacy_loader_state(source.state_dict(), mode == "metadata")
                if duplicated:
                    saved["_index_sampler_state"] = source.state_dict()["_index_sampler_state"]
                original = copy.deepcopy(saved)
            with _build_loader(mode) as restored:
                restored.load_state_dict(saved)
                self.assertEqual(saved, original)
                migrated = restored.state_dict()
                self.assertEqual(migrated["_index_sampler_state"]["consumed_samples"], 8)
                self.assertEqual(set(migrated), {"_index_sampler_state", "_iterator_finished", "dataset_state"})
                for fixed_state in (saved, migrated):
                    fixed = _build_loader("fixed")
                    fixed.load_state_dict(fixed_state)
                    self.assertEqual([sample["id"] for sample in next(iter(fixed))], [8, 9])
                self.assertEqual([sample["id"] for sample in next(restored)], [8, 9])

    def test_mode_switch_repartitions_global_cursor(self) -> None:
        """Feature: Native cursor restoration with a changed DP size.
        Description: Switch DP4 fixed to DP1 balanced and back at a nonzero epoch.
        Expectation: All remaining samples occur once and restored cursors record target DP.
        """
        for mode, sampler_type, workers in product(("online", "metadata"), ("single", "cyclic"), (0, 1)):
            with self.subTest(mode=mode, sampler_type=sampler_type, workers=workers):
                source = _build_loader("fixed", epoch=2, workers=workers,
                                       sampler_type=sampler_type, dp_world_size=4)
                iterator = iter(source)
                next(iterator)
                saved = source.state_dict()
                original = copy.deepcopy(saved)
                list(iterator)
                reference = _build_loader("fixed", consumed=4, epoch=2, sampler_type=sampler_type)
                expected = sorted(sample["id"] for batch in reference for sample in batch)
                with _build_loader(mode, workers=workers, sampler_type=sampler_type) as target:
                    target.load_state_dict(saved)
                    restored = target.state_dict()
                    self.assertEqual(restored["_index_sampler_state"], dict(
                        saved["_index_sampler_state"], dp_world_size=1,
                    ))
                    self.assertEqual(target._step, 2)
                    self.assertEqual(sorted(sample["id"] for batch in target for sample in batch), expected)
                self.assertEqual(saved, original)
                selected = []
                for rank in range(4):
                    fixed = _build_loader("fixed", workers=workers, sampler_type=sampler_type,
                                          dp_world_size=4, dp_rank=rank)
                    fixed.load_state_dict(restored)
                    self.assertEqual(fixed.batch_sampler.state_dict(), dict(
                        restored["_index_sampler_state"], dp_world_size=4,
                    ))
                    selected.extend(sample["id"] for batch in fixed for sample in batch)
                self.assertEqual(sorted(selected), expected)
                self.assertEqual(restored["_index_sampler_state"]["dp_world_size"], 1)

    def test_mode_switch_rejects_changed_dp_with_sharded_shuffle(self) -> None:
        """Feature: Native sampler topology validation.
        Description: Change DP in either direction with per-rank shuffle enabled.
        Expectation: Validate the original source DP before rewriting or consuming samples.
        """
        for mode in ("fixed", "online", "metadata"):
            source_dp, target_dp = (1, 4) if mode == "fixed" else (4, 1)
            source = build_dataset_batch_sampler(
                total_samples=24, consumed_samples=4, micro_batch_size=1,
                global_batch_size=4, dp_world_size=source_dp, dp_rank=0,
                sampler_type="cyclic", data_sharding=True, seed=42,
            )
            saved = {"_index_sampler_state": source.state_dict()}
            original = copy.deepcopy(saved)
            with self.subTest(mode=mode), ExitStack() as stack:
                target = _build_loader(mode, sampler_type="cyclic", data_sharding=True, dp_world_size=target_dp)
                if mode != "fixed":
                    stack.enter_context(target)
                with self.assertRaisesRegex(ValueError, "data_sharding=False"):
                    target.load_state_dict(saved)
                self.assertEqual(saved, original)
                self.assertEqual(target.state_dict()["_index_sampler_state"]["consumed_samples"], 0)

    def test_source_sampler_dp_change_requires_complete_step(self) -> None:
        """Feature: Source-batch sampler DP migration.
        Description: Restore partial source cursors with unchanged or changed DP.
        Expectation: Only complete optimizer steps can be repartitioned across DP ranks.
        """
        for sampler_type, source_dp in product(("single", "cyclic"), (1, 2)):
            with self.subTest(sampler_type=sampler_type, source_dp=source_dp):
                sampler = _build_loader("fixed", sampler_type=sampler_type).batch_sampler
                sampler.enable_source_batch_resume()
                state = dict(sampler.state_dict(), consumed_samples=2, dp_world_size=source_dp)
                if source_dp == 1:
                    sampler.load_state_dict(state)
                    self.assertEqual(sampler.consumed_samples, 2)
                else:
                    with self.assertRaisesRegex(ValueError, "complete global optimizer batch"):
                        sampler.load_state_dict(state)
                    self.assertEqual(sampler.consumed_samples, 0)

    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1)
    @patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier")
    def test_dcp_restores_shared_sampler_cursor(self, *_mocks) -> None:
        """Feature: Native loader checkpoint interoperability.
        Description: Restore embedded and per-rank state across fixed and balanced loaders.
        Expectation: Step and remaining samples match with workers and shuffled epochs.
        """
        pairs = (("fixed", "fixed"), ("fixed", "online"), ("online", "fixed"), ("online", "online"),
                 ("fixed", "metadata"), ("metadata", "fixed"), ("metadata", "metadata"))
        for (source_mode, target_mode), per_rank, workers in product(pairs, (False, True), (0, 1)):
            with self.subTest(source=source_mode, target=target_mode, per_rank=per_rank, workers=workers):
                self._check_dcp_resume(source_mode, target_mode, per_rank, workers)
        for mode, per_rank in product(("online", "metadata"), (False, True)):
            with self.subTest(source=mode, per_rank=per_rank, legacy=True):
                self._check_dcp_resume(mode, mode, per_rank, 0, legacy=True)
            with self.subTest(source=mode, per_rank=per_rank, target_workers=1):
                self._check_dcp_resume(mode, "fixed", per_rank, 0, target_workers=1)
            for legacy in (False, True):
                with self.subTest(source=mode, per_rank=per_rank, finished=True, legacy=legacy):
                    self._check_dcp_resume(mode, mode, per_rank, 0, legacy=legacy, finished=True)
        for target in ("fixed", "online", "metadata"):
            with self.subTest(target=target, user_dataset_state=True):
                self._check_dcp_resume("fixed", target, False, 0, stateful_data=True)

    def _check_dcp_resume(self, source_mode: str, target_mode: str, per_rank: bool, workers: int,
                          *, legacy: bool = False, target_workers: int | None = None,
                          finished: bool = False, stateful_data: bool = False) -> None:
        """Check one checkpoint layout and loader-mode combination."""
        with ExitStack() as stack:
            samples = _StatefulSamples({"id": index} for index in range(24)) if stateful_data else None
            source = _build_loader(source_mode, epoch=2, workers=workers, sampler_type="cyclic", samples=samples)
            if target_workers is None:
                target_workers = workers if source_mode == target_mode == "fixed" else 0
            target = _build_loader(target_mode, epoch=0, workers=target_workers, sampler_type="cyclic", samples=samples)
            for mode, loader in ((source_mode, source), (target_mode, target)):
                if mode != "fixed":
                    stack.enter_context(loader)
            iterator = iter(source)
            next(iterator)
            next(iterator)
            if finished:
                list(iterator)
            saved = source.state_dict()
            self.assertNotIn("native_sampler", saved)
            self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 24 if finished else 4)
            if source_mode != "fixed":
                self.assertEqual(set(saved), {"_index_sampler_state", "_iterator_finished", "dataset_state"})
                if legacy:
                    saved = _legacy_loader_state(saved, source_mode == "metadata")
            expected = sorted(sample["id"] for batch in iterator for sample in batch)
            checkpointer = DistributedCheckpointer(extra_state_per_rank=per_rank)
            path = stack.enter_context(tempfile.TemporaryDirectory())
            checkpointer.save(path, {"model": {"w": torch.ones(1)}, "extra_state": {
                "global_step": 1, "train_dataloader": saved,
            }}, global_step=1, save_async=False)
            restored = {"model": {"w": torch.zeros(1)}}
            checkpointer.load(path, restored, extra_state_skeleton={
                "global_step": 0, "train_dataloader": target.state_dict_for_load(),
            })
            self.assertEqual(restored["extra_state"]["global_step"], 1)
            target.load_state_dict(restored["extra_state"]["train_dataloader"])
            if finished:
                self.assertTrue(target.state_dict()["_iterator_finished"])
            self.assertEqual(sorted(sample["id"] for batch in target for sample in batch), expected)
            if finished:
                target.set_epoch(3)
                self.assertEqual(len(list(target)), 12)

    def test_legacy_fixed_cursor_and_unsafe_worker_state(self) -> None:
        """Feature: Legacy fixed checkpoint migration.
        Description: Restore the original zero-worker and nested worker snapshots.
        Expectation: Restore the old single-process cursor and reject ambiguous worker progress.
        """
        for workers, mode in product((0, 1), ("online", "metadata")):
            source = _build_loader("fixed", workers=workers)
            iterator = iter(source)
            next(iterator)
            next(iterator)
            saved = source.state_dict()
            if workers:
                saved.pop("_index_sampler_state")
            list(iterator)
            with self.subTest(workers=workers, mode=mode), _build_loader(mode) as target:
                if workers:
                    with self.assertRaisesRegex(ValueError, "no native sampler cursor"):
                        target.load_state_dict(saved)
                else:
                    target.load_state_dict(saved)
                    self.assertEqual([sample["id"] for sample in next(target)], [4, 5])

    def test_portable_cursor_excludes_prefetched_windows(self) -> None:
        """Feature: Delivered checkpoint progress.
        Description: Complete one step, wait for lookahead, and resume a nonzero start.
        Expectation: Only delivered complete steps contribute to the exported cursor.
        """
        for mode in ("online", "metadata"):
            with self.subTest(mode=mode), _build_loader(mode, consumed=4) as source:
                next(source)
                self.assertEqual(source.state_dict()["_index_sampler_state"]["consumed_samples"], 4)
                next(source)
                source.wait_for_prefetch()
                saved = source.state_dict()
                self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 8)
                self.assertEqual(set(saved), {"_index_sampler_state", "_iterator_finished", "dataset_state"})
                with _build_loader(mode) as restored:
                    restored.load_state_dict(saved)
                    self.assertEqual(restored.state_dict()["_index_sampler_state"], saved["_index_sampler_state"])
                    next(restored)
                    next(restored)
                    self.assertEqual(restored.state_dict()["_index_sampler_state"]["consumed_samples"], 12)

    def test_shared_cursor_restores_all_reader_epochs(self) -> None:
        """Feature: Loader-owned progress restoration.
        Description: Restore only the native sampler cursor, or a template retaining fresh Reader epochs.
        Expectation: Loader, Reader and payload loader use the saved epoch and resume at sample eight.
        """
        for mode, template in product(("online", "metadata"), (False, True)):
            with self.subTest(mode=mode, template=template), _build_loader(mode) as target:
                saved = target.state_dict_for_load() if template else {}
                saved["_index_sampler_state"] = {
                    "consumed_samples": 8, "epoch": 2, "global_batch_size": 4, "dp_world_size": 1,
                }
                original = copy.deepcopy(saved)
                target.load_state_dict(saved)
                restored = target.state_dict()
                reader_state = target._planning_reader().state_dict()
                sample_state = (target._direct_sample_loader.state_dict() if mode == "metadata"
                                else reader_state["sample_loader"])
                self.assertEqual((restored["_index_sampler_state"]["epoch"],
                                  reader_state["epoch"], sample_state["epoch"]), (2, 2, 2))
                self.assertEqual(target._step, 4)
                self.assertEqual([sample["id"] for sample in next(target)], [8, 9])
                self.assertEqual(saved, original)

    def test_reader_checkpoint_requires_sampler_cursor(self) -> None:
        """Feature: Deduplicated Reader checkpoint validation.
        Description: Restore a Reader checkpoint without a valid sampler state in either location.
        Expectation: Reject missing progress without changing the target loader or input state.
        """
        for mode, missing in product(("online", "metadata"), (True, False, None)):
            with self.subTest(mode=mode, missing=missing), _build_loader(mode, consumed=4) as source:
                saved = source.state_dict()
                if missing is None:
                    saved = source.state_dict_for_load()
                elif missing:
                    saved.pop("_index_sampler_state")
                else:
                    saved["_index_sampler_state"] = None
                original = copy.deepcopy(saved)
            with _build_loader(mode) as target:
                initial = target.state_dict()
                with self.assertRaisesRegex(ValueError, "no native sampler cursor"):
                    target.load_state_dict(saved)
                self.assertEqual(saved, original)
                self.assertEqual(target.state_dict(), initial)

    def test_shared_cursor_preserves_payload_reader_validation(self) -> None:
        """Feature: Shared-cursor restoration compatibility checks.
        Description: Restore a valid sampler with incompatible payload-reader metadata.
        Expectation: Dataset size and seed mismatches remain rejected after normalizing epochs.
        """
        for mode, field in product(("online", "metadata"), ("dataset_size", "seed")):
            with self.subTest(mode=mode, field=field), _build_loader(mode) as target:
                saved = target.state_dict()
                saved["dataset_state"][field] += 1
                original = copy.deepcopy(saved)
                with self.assertRaisesRegex(ValueError, field):
                    target.load_state_dict(saved)
                self.assertEqual(saved, original)

    def test_switch_rejects_invalid_native_cursor(self) -> None:
        """Feature: Complete-step cursor validation.
        Description: Switch with incomplete source steps, negative cursors or invalid offsets in either direction.
        Expectation: Reject invalid progress before advancing the target sampler.
        """
        changes = ({"consumed_samples": 2}, {"consumed_samples": -4}, {"global_batch_size": 8},
                   {"epoch": -1}, {"dp_world_size": 2, "consumed_samples": 2},
                   {"optimizer_step_offset": -1}, {"optimizer_step_offset": 4},
                   {"optimizer_step_offset": True}, {"optimizer_step_offset": 2})
        for mode, changed in product(("fixed", "online", "metadata"), changes):
            with self.subTest(mode=mode, changed=changed), ExitStack() as stack:
                source = stack.enter_context(_build_loader("online", consumed=4))
                saved = source.state_dict()
                if mode != "fixed":
                    saved = {"_index_sampler_state": saved["_index_sampler_state"]}
                saved["_index_sampler_state"].update(changed)
                target = _build_loader(mode)
                if mode != "fixed":
                    stack.enter_context(target)
                with self.assertRaises(ValueError):
                    target.load_state_dict(saved)

    def test_new_gbs_does_not_hide_partial_source_step(self) -> None:
        """Feature: Source-boundary validation for variable GBS.
        Description: Load cursor thirty-two at offset six with source GBS eight and target GBS sixteen.
        Expectation: Divisibility by the target GBS cannot make the partial source step valid.
        """
        for mode in ("fixed", "online", "metadata"):
            with self.subTest(mode=mode), ExitStack() as stack:
                target = _build_loader(mode, global_batch_size=16, samples=[{"id": i} for i in range(64)])
                if mode != "fixed":
                    stack.enter_context(target)
                with self.assertRaisesRegex(ValueError, "complete optimizer"):
                    target.load_state_dict({"_index_sampler_state": {
                        "consumed_samples": 32, "epoch": 2, "global_batch_size": 8,
                        "dp_world_size": 1, "optimizer_step_offset": 6,
                    }})

    def test_legacy_model_peer_does_not_invent_absolute_cursor(self) -> None:
        """Feature: Legacy non-Reader checkpoint compatibility.
        Description: Resume a model peer whose old state has no sampler or initial cursor.
        Expectation: Same-mode progress restores without fabricating a portable source cursor.
        """
        topology = DataTopology.from_layout((1, 2), ("dp", "tp"), (0, 1), 1)
        with DistributedDataLoader(
                topology=topology, dataset_reader_ranks=(0,), dataset_reader=None, metadata_reader=None,
                direct_sample_loader=None, metadata_mode=False, planner=SimpleNamespace(local_batch_size=2),
                data_constructor=Mock(), data_plane=Mock(), model_transport=Mock(), accumulation_steps=2,
        ) as target:
            saved = {"global_rank": 1, "epoch": 0, "step": 2, "stopped": False}
            target.load_state_dict(saved)
            restored = target.state_dict()
            self.assertEqual(restored["step"], 2)
            self.assertNotIn("_index_sampler_state", restored)
            with self.assertRaisesRegex(ValueError, "no native sampler cursor"):
                _build_loader("fixed").load_state_dict(restored)

    def test_worker_cursor_includes_deliveries_since_snapshot(self) -> None:
        """Feature: Portable cursor for infrequent worker snapshots.
        Description: Save after deliveries between torchdata snapshots while workers prefetch.
        Expectation: Replay begins at the delivered cursor, excluding worker lookahead.
        """
        source = _build_loader("fixed", workers=1)
        source.snapshot_every_n_steps = 3
        iterator = iter(source)
        next(iterator)
        next(iterator)
        saved = source.state_dict()
        self.assertGreater(saved["_steps_since_snapshot"], 0)
        self.assertEqual(saved["_index_sampler_state"]["consumed_samples"], 4)
        snapshot_cursor = saved["_snapshot"]["_main_snapshot"]["_index_sampler_state"]["consumed_samples"]
        self.assertEqual(snapshot_cursor, 4 - saved["_steps_since_snapshot"] * 2)
        self.assertEqual(source.state_dict(), saved)
        list(iterator)
        for mode in ("online", "metadata"):
            with self.subTest(mode=mode), _build_loader(mode) as target:
                target.load_state_dict(saved)
                self.assertEqual([sample["id"] for sample in next(target)], [4, 5])
        target = _build_loader("fixed", workers=1)
        target.load_state_dict(saved)
        self.assertEqual([sample["id"] for batch in target for sample in batch], list(range(4, 24)))
