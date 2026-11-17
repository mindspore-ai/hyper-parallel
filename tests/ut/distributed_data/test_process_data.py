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
"""Exercise the real CPU process handoff without distributed setup or devices."""

import os
from typing import Any
import unittest

import torch

from hyper_parallel.distributed_data.process_data import ProcessDataWorker
from hyper_parallel.distributed_data.transport import DataGroups, DataPlaneTransport


from tests.common.mark_utils import arg_mark


class _Source:
    """Generate aliasing CPU views and expose ownership through the result."""

    global_rank = 0

    def __init__(self, fail=False):
        self.step = 0
        self.fail = fail

    def bind_transport(self, transport: Any) -> Any:
        """Bind the test transport."""
        self.transport = transport

    def __next__(self):
        if self.fail:
            raise ValueError("source preparation failed")
        if self.step == 2:
            raise StopIteration
        self.step += 1
        storage = torch.arange(12).reshape(3, 4)
        return {"pid": os.getpid(), "data": storage, "view": storage[:, ::2], "step": self.step}

    def set_epoch(self, epoch: Any) -> Any:
        """Reset the deterministic test cursor."""
        self.step = 0

    def reset_iteration(self) -> Any:
        """Re-enter the deterministic test source."""
        self.step = 0

    def close(self) -> Any:
        """Release the test source."""


def _worker(source):
    return ProcessDataWorker(source, DataPlaneTransport(DataGroups((0,), None, None, None, 0, False), 0))


class TestProcessDataWorker(unittest.TestCase):
    """Validate shared-storage ownership, error delivery and lifecycle."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_child_cpu_sharing_epoch_and_surviving_storage(self) -> None:
        """Feature: CPU tensor IPC.
        Description: Exchange aliasing tensor views through a real child process.
        Expectation: Shared storage survives child exit and epoch replay resets its cursor.
        """
        source = _Source()
        worker = _worker(source)
        try:
            worker.prefetch()
            worker.prefetch()
            first = next(worker)
            self.assertNotEqual(first["pid"], os.getpid())
            self.assertTrue(first["data"].is_shared())
            self.assertEqual(first["data"].untyped_storage().data_ptr(),
                             first["view"].untyped_storage().data_ptr())
            self.assertEqual(next(worker)["step"], 2)
            with self.assertRaises(StopIteration):
                next(worker)
            worker.set_epoch(3)
            self.assertEqual(next(worker)["step"], 1)
            worker.prefetch()
            worker.close()
            worker.close()
            self.assertEqual(source.step, 0)
            self.assertIsNotNone(worker._process.poll())
            torch.testing.assert_close(first["view"], torch.tensor([[0, 2], [4, 6], [8, 10]]))
        finally:
            worker.close()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_worker_failure_includes_traceback_and_is_reaped(self) -> None:
        """Feature: Child failure cleanup.
        Description: Raise an exception during source preparation.
        Expectation: The parent receives its traceback and reaps the child.
        """
        worker = _worker(_Source(fail=True))
        try:
            with self.assertRaisesRegex(RuntimeError, "ValueError: source preparation failed"):
                next(worker)
        finally:
            worker.close()
        self.assertIsNotNone(worker._process.poll())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_unstarted_close_does_not_launch_child(self) -> None:
        """Feature: Lazy process startup.
        Description: Close a prepared worker before its first request.
        Expectation: No child starts and its temporary capsule is deleted.
        """
        worker = _worker(_Source())
        worker.close()
        self.assertIsNone(worker._process)
        self.assertFalse(worker._capsule.exists())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_active_generator_reports_serialization_boundary(self) -> None:
        """Feature: Process source validation.
        Description: Capture a live generator inside an otherwise valid source.
        Expectation: Construction reports the unsupported source before launching a child.
        """
        source = _Source()
        source.generator = (index for index in range(2))
        with self.assertRaisesRegex(ValueError, "serializable, unstarted CPU sources"):
            _worker(source)
