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
"""Check bounded manual GC and restoration without changing the caller process policy."""

import gc
import multiprocessing
import unittest
from multiprocessing.connection import Connection
from types import SimpleNamespace
from unittest.mock import patch

from hyper_parallel.trainer.callbacks.garbage_collection_callback import GarbageCollectionCallback
from hyper_parallel.trainer.state import TrainerState


class TestGarbageCollectionCallback(unittest.TestCase):
    """Verify cleanup scheduling and process-global policy ownership."""

    @staticmethod
    def make_callback(manual_gc: bool = False, gc_steps: int = 0) -> GarbageCollectionCallback:
        """Build a callback with no model, device or process group."""
        trainer = SimpleNamespace(mesh=None, config=SimpleNamespace(training=SimpleNamespace(
            manual_gc=manual_gc, gc_steps=gc_steps, empty_cache_steps=0)))
        return GarbageCollectionCallback(trainer)

    def test_default_keeps_automatic_gc_and_existing_periodic_cleanup(self) -> None:
        """Feature: default policy. Description: automatic GC. Expectation: preserve existing cadence."""
        callback = self.make_callback(gc_steps=3)
        with patch("gc.disable") as disable, patch("gc.enable") as enable, patch("gc.collect") as collect:
            callback.on_train_begin(TrainerState())
            callback.on_step_begin(TrainerState(global_step=3))
            collect.assert_not_called()
            callback.on_step_end(TrainerState(global_step=3))
            collect.assert_called_once_with()
            callback.on_train_end(TrainerState())
            disable.assert_not_called()
            enable.assert_not_called()

    def test_manual_gc_is_periodic_and_charged_to_the_next_step(self) -> None:
        """Feature: manual policy. Description: interval cleanup. Expectation: no hidden end-step cleanup."""
        callback = self.make_callback(True, 50)
        with patch("gc.isenabled", return_value=True), patch("gc.disable") as disable, \
                patch("gc.enable") as enable, patch("gc.collect") as collect:
            callback.on_train_begin(TrainerState())
            disable.assert_called_once_with()
            collect.assert_called_once_with()
            collect.reset_mock()
            callback.on_step_begin(TrainerState(global_step=0))
            collect.assert_not_called()
            for completed in range(1, 51):
                callback.on_step_end(TrainerState(global_step=completed))
                callback.on_step_begin(TrainerState(global_step=completed))
            collect.assert_called_once_with()
            callback.close()
            callback.close()
            enable.assert_called_once_with()

    def test_restores_an_initially_disabled_policy(self) -> None:
        """Feature: policy ownership. Description: caller disables GC. Expectation: leave it disabled."""
        callback = self.make_callback(True, 10)
        with patch("gc.isenabled", return_value=False), patch("gc.disable") as disable, \
                patch("gc.enable") as enable, patch("gc.collect"):
            callback.on_train_begin(TrainerState())
            callback.on_train_end(TrainerState())
            self.assertEqual(disable.call_count, 2)
            enable.assert_not_called()

    def test_manual_policy_requires_bounded_cleanup(self) -> None:
        """Feature: cleanup bound. Description: no interval. Expectation: reject unbounded GC suppression."""
        for interval in (0, -1):
            with self.subTest(gc_steps=interval):
                with self.assertRaisesRegex(ValueError, "positive gc_steps"):
                    self.make_callback(True, interval)

    def test_manual_gc_keeps_device_cache_cleanup_independent(self) -> None:
        """Feature: device cleanup. Description: manual Python GC. Expectation: preserve cache cadence."""
        callback = self.make_callback(True, 50)
        callback.empty_cache_steps = 3
        module = "hyper_parallel.trainer.callbacks.garbage_collection_callback"
        with (
            patch(f"{module}.get_device_type", return_value="npu"),
            patch(f"{module}.empty_cache") as empty_cache,
            patch("gc.collect") as collect,
        ):
            for step in range(1, 4):
                callback.on_step_end(TrainerState(global_step=step))
            empty_cache.assert_called_once_with()
            collect.assert_not_called()

    def test_forked_data_worker_restores_its_automatic_gc_policy(self) -> None:
        """Feature: worker isolation. Description: fork during training. Expectation: worker GC stays automatic."""
        if "fork" not in multiprocessing.get_all_start_methods():
            self.skipTest("This check requires the fork worker start method")
        previous = gc.isenabled()
        gc.enable()
        callback = self.make_callback(True, 50)
        context = multiprocessing.get_context("fork")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=_send_gc_policy, args=(writer,))
        try:
            callback.on_train_begin(TrainerState())
            self.assertFalse(gc.isenabled())
            process.start()
            writer.close()
            self.assertTrue(reader.poll(5))
            self.assertTrue(reader.recv())
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            self.assertFalse(gc.isenabled())
        finally:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
            reader.close()
            writer.close()
            callback.close()
            if not previous:
                gc.disable()


def _send_gc_policy(connection: Connection) -> None:
    """Expose only the forked worker's policy, without importing an accelerator runtime."""
    connection.send(gc.isenabled())
    connection.close()
