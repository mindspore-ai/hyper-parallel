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
"""Tests for process-backed external delivery and staging lifecycle."""

import os
import pickle
import time
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import psutil
import torch

from hyper_parallel.distributed_data import (
    CausalAttentionCostModel, DistributedDatasetConfig, SampleMetadata, build_distributed_dataloader,
)
from hyper_parallel.distributed_data.process_data import _serialize_source, create_data_worker
from tests.common.mark_utils import arg_mark


class _Mesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


def _loader(collate_fn, source=(((0,),), ((1,),))):
    return build_distributed_dataloader(
        None, _Mesh(), DistributedDatasetConfig(seq_len=10, local_batch_size=1, communication_backend="gloo"),
        external_step_source=source, metadata_fn=lambda _: SampleMetadata(1), pack_fn=lambda samples, _: samples[0],
        collate_fn=collate_fn, cost_model=CausalAttentionCostModel(), device="cpu",
    )


def _worker_pid_bin(samples):
    return [[(os.getpid(), sample) for sample in samples]]


def _exit_with_open_loader(pid_file: Path) -> None:
    loader = _loader(list)
    loader.prefetch()
    pid_file.write_text(str(loader._worker._process.pid), encoding="utf-8")
    raise SystemExit(17)


@patch("hyper_parallel.distributed_data.packed_balancing.create_data_worker", create_data_worker)
class TestPackedBalancingPrefetch(unittest.TestCase):
    """Exercise the public loader across a real CPU process boundary."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_source_callbacks_without_dill(self) -> None:
        """Feature: Data process startup with standard-library pickle.
        Description: Serialize an importable callback and a local callback without dill.
        Expectation: The importable callback round-trips and the local callback reports the requirement.
        """
        with patch("hyper_parallel.distributed_data.process_data._SOURCE_PICKLE", pickle):
            self.assertEqual(pickle.loads(_serialize_source((SampleMetadata,))), (SampleMetadata,))
            with self.assertRaisesRegex(TypeError, "importable module"):
                _serialize_source((lambda sample: sample,))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_prefetch_preserves_order_and_shared_storage_after_close(self) -> None:
        """Feature: External process prefetch.
        Description: Repeat prefetch hooks, consume both batches and replay an epoch.
        Expectation: Only delivery advances the step and shared views survive child exit.
        """
        def collate(ids: list[int]) -> tuple:
            """Return two views of the same child-owned storage."""
            data = torch.arange(4) + ids[0]
            return os.getpid(), data, data[::2]

        with closing(_loader(collate)) as loader:
            loader.prefetch_plan()
            loader.prefetch()
            loader.wait_for_prefetch()
            self.assertEqual(loader.step, 0)
            self.assertIsNone(loader.last_balance_stats)
            pid, data, view = next(loader)
            self.assertNotEqual(pid, os.getpid())
            self.assertTrue(data.is_shared())
            self.assertEqual(data.untyped_storage().data_ptr(), view.untyped_storage().data_ptr())
            self.assertEqual(loader.step, 1)
            self.assertEqual(next(loader)[1].tolist(), [1, 2, 3, 4])
            self.assertIsNone(next(loader, None))
            loader.set_epoch(1)
            self.assertEqual(next(loader)[1].tolist(), [0, 1, 2, 3])
        self.assertEqual(view.tolist(), [0, 2])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_rolling_copy_preserves_a_ready_external_step(self) -> None:
        """Feature: External H2D lookahead.
        Description: Warm the first two inputs, then report that the third CPU reply is late.
        Expectation: The second step is delivered immediately and the third is copied once on demand.
        """
        stage = Mock(side_effect=lambda batch: Mock(take_batch=lambda: batch))
        with closing(_loader(list, source=(((0,),), ((1,),), ((2,),)))) as loader:
            loader._device_prefetch = stage
            loader.prefetches_to_device = True
            self.assertEqual(next(loader), [0])
            self.assertEqual(stage.call_count, 2)
            with patch.object(loader._worker, "is_ready", return_value=False):
                self.assertEqual(next(loader), [1])
                self.assertEqual(stage.call_count, 2)
            self.assertEqual(next(loader), [2])
            self.assertIsNone(next(loader, None))
            self.assertEqual(stage.call_count, 3)
            self.assertEqual(loader.step, 3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_child_error_follows_preceding_batch(self) -> None:
        """Feature: Speculative error delivery.
        Description: Fail collation in the child on the second batch.
        Expectation: The first batch is delivered before the traceback and close reaps the worker.
        """
        def collate(ids: list[int]) -> list[int]:
            """Fail only the successor."""
            if ids == [1]:
                raise ValueError("failed on second batch")
            return ids

        with closing(_loader(collate)) as loader:
            self.assertEqual(next(loader), [0])
            pid = loader._worker._process.pid
            with self.assertRaisesRegex(RuntimeError, "ValueError: failed on second batch"):
                next(loader)
            self.assertEqual(loader.step, 1)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_spawn_failure_before_ready(self) -> None:
        """Feature: Spawn startup failure.
        Description: Remove the staged source before the child opens it.
        Expectation: Prefetch returns the child traceback and releases the process and its temporary files.
        """
        with closing(_loader(list)) as loader, closing(create_data_worker(loader._source, loader._transport)) as worker:
            active = set(torch.multiprocessing.active_children())
            worker._source_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "FileNotFoundError"):
                worker.prefetch()
            self.assertFalse(worker._source_path.parent.exists())
            self.assertFalse(set(torch.multiprocessing.active_children()) - active)
            worker.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_close_after_outer_process_is_killed(self) -> None:
        """Feature: Nested DataLoader worker lifetime.
        Description: Kill the outer data process while its persistent worker is still alive.
        Expectation: Closing the loader reaps the outer process and stops its nested worker.
        """
        source = torch.utils.data.DataLoader(
            range(2), collate_fn=_worker_pid_bin, num_workers=1,
            persistent_workers=True, multiprocessing_context="spawn",
        )
        with closing(_loader(list, source=source)) as loader:
            (inner_pid, sample), = next(loader)
            outer = loader._worker._process
            outer_pid = outer.pid
            inner = psutil.Process(inner_pid)
            self.assertEqual(sample, 0)
            self.assertEqual(inner.ppid(), outer_pid)
            self.assertEqual(os.getpgid(inner_pid), outer_pid)
            outer.kill()
            outer.join(timeout=10)
            self.assertIsNotNone(outer.exitcode)
        with self.assertRaises(ProcessLookupError):
            os.kill(outer_pid, 0)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if inner.status() == psutil.STATUS_ZOMBIE:
                    return
            except psutil.NoSuchProcess:
                return
            time.sleep(0.05)
        self.fail(f"Nested DataLoader worker {inner_pid} survived close")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_training_process_exits_with_open_loader(self) -> None:
        """Feature: Non-daemonic data process shutdown.
        Description: Exit the training process with a live loader and a pending request.
        Expectation: Training exits with its original status and the data process is reaped.
        """
        with TemporaryDirectory() as directory:
            pid_file = Path(directory) / "worker.pid"
            context = torch.multiprocessing.get_context("spawn")
            trainer = context.Process(target=_exit_with_open_loader, args=(pid_file,))
            trainer.start()
            try:
                trainer.join(timeout=90)
                self.assertEqual(trainer.exitcode, 17)
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid_file.read_text(encoding="utf-8")), 0)
            finally:
                if trainer.is_alive():
                    trainer.kill()
                trainer.join()
                trainer.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_live_generator_rejected_before_delivery(self) -> None:
        """Feature: Process source validation.
        Description: Supply a live generator to the external loader.
        Expectation: Prefetch reports the serialization requirement before delivering a batch.
        """
        with closing(_loader(list, source=([[index]] for index in range(2)))) as loader:
            with self.assertRaisesRegex(ValueError, "serializable, unstarted CPU sources"):
                loader.prefetch()
