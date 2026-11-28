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
"""DCP round-trip checkpoint interoperability for native loaders.

Split from ``test_loader_resume.py``: these two scenarios dominate that
file's runtime (full save/load round trips through ``DistributedCheckpointer``
across the loader-mode Cartesian products), so they live in their own module
to keep per-file wall time balanced under ``pytest-xdist --dist loadfile``.
The in-memory cursor and legacy-migration tests stay in the original file.
"""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from itertools import product
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer
from tests.common.mark_utils import arg_mark
from tests.ut.distributed_data.test_batch_sampler import _legacy_loader_state
from tests.ut.distributed_data.test_loader_resume import _StatefulSamples, _build_loader

_SINGLE_RANK_DCP = (
    patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_rank", return_value=0),
    patch("hyper_parallel.core.distributed_checkpoint.api.dist.get_world_size", return_value=1),
    patch("hyper_parallel.core.distributed_checkpoint.api.dist.barrier"),
)


def _single_rank_dcp(method):
    """Run one test with the DCP collective calls pinned to a single rank."""
    for patcher in _SINGLE_RANK_DCP:
        method = patcher(method)
    return method


class TestLoaderResumeDcp(unittest.TestCase):
    """Exercise full DCP save/load round trips across loader modes."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    @_single_rank_dcp
    def test_dcp_restores_shared_sampler_cursor(self, *_mocks: Any) -> None:
        """Feature: Native loader checkpoint interoperability.
        Description: Restore embedded and per-rank state across fixed and balanced loaders.
        Expectation: Step and remaining samples match with workers and shuffled epochs.
        """
        pairs = (("fixed", "fixed"), ("fixed", "online"), ("online", "fixed"), ("online", "online"),
                 ("fixed", "metadata"), ("metadata", "fixed"), ("metadata", "metadata"))
        for (source_mode, target_mode), per_rank, workers in product(pairs, (False, True), (0, 1)):
            with self.subTest(source=source_mode, target=target_mode, per_rank=per_rank, workers=workers):
                self._check_dcp_resume(source_mode, target_mode, per_rank, workers)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    @_single_rank_dcp
    def test_dcp_restores_legacy_and_finished_cursors(self, *_mocks: Any) -> None:
        """Feature: Legacy snapshots and exhausted iterators through DCP.
        Description: Restore legacy balanced snapshots, worker-mode switches and finished epochs.
        Expectation: Restored loaders resume the saved suffix and re-iterate after set_epoch.
        """
        for mode, per_rank in product(("online", "metadata"), (False, True)):
            with self.subTest(source=mode, per_rank=per_rank, legacy=True):
                self._check_dcp_resume(mode, mode, per_rank, 0, legacy=True)
            with self.subTest(source=mode, per_rank=per_rank, target_workers=1):
                self._check_dcp_resume(mode, "fixed", per_rank, 0, target_workers=1)
            for legacy in (False, True):
                with self.subTest(source=mode, per_rank=per_rank, finished=True, legacy=legacy):
                    self._check_dcp_resume(mode, mode, per_rank, 0, legacy=legacy, finished=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    @_single_rank_dcp
    def test_dcp_preserves_user_dataset_state(self, *_mocks: Any) -> None:
        """Feature: User Dataset state alongside the loader cursor.
        Description: Round-trip a Dataset with its own state_dict through every target mode.
        Expectation: The user payload survives unchanged and the cursor still resumes.
        """
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
            saved = self._check_saved_format(saved, source_mode, legacy)
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

    def _check_saved_format(self, saved: dict, source_mode: str, legacy: bool) -> dict:
        """Check the balanced cursor schema and optionally build its legacy representation."""
        if source_mode != "fixed":
            self.assertEqual(set(saved), {"_index_sampler_state", "_iterator_finished", "dataset_state"})
            if legacy:
                return _legacy_loader_state(saved, source_mode == "metadata")
        return saved

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    @_single_rank_dcp
    def test_dcp_keeps_optimizer_step_offset(self, *_mocks: Any) -> None:
        """Feature: Persistent boundaries after a GBS change.
        Description: Save offset cursors with native worker snapshots through embedded and per-rank DCP.
        Expectation: Restored GBS-eight steps consume samples 38 through 45 and preserve the training step.
        """
        for mode, workers, per_rank in product(("fixed", "online", "metadata"), (0, 1), (False, True)):
            with self.subTest(mode=mode, workers=workers, per_rank=per_rank), ExitStack() as stack:
                samples = [{"id": index} for index in range(70)]
                options = {"global_batch_size": 8, "samples": samples, "workers": workers}
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


if __name__ == "__main__":
    unittest.main()
