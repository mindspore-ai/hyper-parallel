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
"""Real buffer contracts around the reused producer and partial admission control."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from rl.async_runtime.protocol import AsyncTrainingConfig, decode_group
from rl.async_runtime.colocated import ColocateAsyncRollouter
from tests.ut.rl.fully_async.support import group


class TestColocateAsyncRollouter(unittest.TestCase):
    """Exercise credit conservation and staleness at actual transport boundaries."""

    def setUp(self) -> None:
        """Install CPU fixtures and register cleanup for each isolated case."""
        self.settings = AsyncTrainingConfig(max_inflight_groups=4, max_buffer_groups=4)
        self.config = {"rollout": {"num_return_sequences": 2}, "colocate_async": {"warmup_batches": 1}}
        self.runtime = Mock()
        with patch("rl.async_runtime.colocated.AsyncRuntime", return_value=self.runtime):
            self.rollouter = ColocateAsyncRollouter(self.config, self.settings)

    def _accept(self, version=1, update=0):
        task = self.rollouter.buffer.reserve(version, update)
        self.rollouter.buffer.accept(group(task).encode(2, self.settings.max_group_bytes))

    def test_complete_groups_are_leased_until_update_commit(self):
        """Verify complete groups are leased until update commit."""
        original = deepcopy(self.config)
        for _ in range(3):
            self._accept()
        payloads = self.rollouter.sample_batch(1, 0, 2)
        self.assertEqual(len(payloads), 2)
        self.assertEqual(len(self.rollouter.buffer.ready), 3)
        self.assertEqual(len(self.rollouter.buffer.selected), 2)
        self.rollouter.commit_batch()
        self.assertEqual(self.rollouter.metrics()["colocate/ready_groups"], 1)
        self.assertEqual(self.rollouter.metrics()["colocate/consumed_groups"], 2)
        self.assertEqual(self.config, original)

    def test_fill_pump_cannot_sneak_stale_groups_into_a_batch(self):
        self.runtime.fill.side_effect = lambda *_: (self._accept(1, 0), self._accept(7, 6))
        payloads = self.rollouter.sample_batch(7, 6, 1)
        self.assertEqual(decode_group(payloads[0], 2, self.settings.max_group_bytes).task.policy_version, 7)
        self.assertEqual(self.rollouter.buffer.dropped_groups, 1)

    def test_snapshot_requires_committed_update_and_drains_before_source(self):
        """Verify snapshot requires committed update and drains before source."""
        self._accept()
        self.rollouter.sample_batch(1, 0, 1)
        with self.assertRaisesRegex(RuntimeError, "uncommitted"):
            self.rollouter.drain_and_snapshot()
        self.runtime.drain.assert_not_called()
        self.rollouter.commit_batch()
        self.runtime.snapshot.return_value = {"epoch": 1}
        snapshot = self.rollouter.drain_and_snapshot()
        self.assertEqual(snapshot["source"], {"epoch": 1})
        self.assertEqual(snapshot["buffer"]["consumed_groups"], 1)
        calls = [call[0] for call in self.runtime.mock_calls]
        self.assertLess(calls.index("drain"), calls.index("snapshot"))

    def test_pause_resume_and_cleanup_use_existing_transport(self):
        self.rollouter.pause()
        self.rollouter.resume(3)
        self.rollouter.close(graceful=True)
        self.runtime.pause.assert_called_once_with()
        self.runtime.resume.assert_called_once_with(3)
        self.runtime.close.assert_called_once_with(graceful=True)

    def test_restore_preserves_ready_experience_and_producer_cursor(self):
        self._accept()
        restored = {"buffer": self.rollouter.buffer.state_dict(), "source": {"cursor": 12}}
        with patch("rl.async_runtime.colocated.AsyncRuntime") as runtime:
            result = ColocateAsyncRollouter(self.config, self.settings, restored)
        self.assertEqual(result.buffer.next_identity, 1)
        self.assertEqual(result.buffer.ready, self.rollouter.buffer.ready)
        self.assertEqual(runtime.call_args.args[2], {"cursor": 12})
        self.assertNotIn("colocate_async", runtime.call_args.args[0])
        self.assertEqual(runtime.call_args.args[0]["async_training"]["max_buffer_groups"], 4)

    def test_wait_has_a_deadline_and_transport_failures_propagate(self):
        with patch("rl.async_runtime.colocated.time.monotonic", side_effect=[0, 1000]):
            with self.assertRaises(TimeoutError):
                self.rollouter.sample_batch(1, 0, 1)
        self.runtime.pump.side_effect = RuntimeError("producer died")
        with self.assertRaisesRegex(RuntimeError, "producer died"):
            self.rollouter.sample_batch(1, 0, 1)

    def test_invalid_version_does_not_dispatch(self):
        with self.assertRaises(ValueError):
            self.rollouter.sample_batch(2, 0, 1)
        self.runtime.fill.assert_not_called()

    def test_rejection_releases_only_rejected_groups_and_preserves_other_leases(self):
        """Verify rejection releases only rejected groups and preserves other leases."""
        for _ in range(3):
            self._accept()
        self.rollouter.sample_batch(1, 0, 2)
        self.rollouter.reject_batch_groups(("0",))
        self.assertEqual(set(self.rollouter.buffer.ready), {1, 2})
        self.assertEqual(self.rollouter.buffer.selected, ())
        self.assertEqual(self.rollouter.buffer.consumed_groups, 0)
        self.assertEqual(self.rollouter.buffer.dropped_groups, 1)
        self.assertEqual(self.rollouter.buffer.next_identity, 3)
        self.rollouter.sample_batch(1, 0, 2)
        self.assertEqual(self.rollouter.buffer.selected, (1, 2))

    def test_rejection_cannot_remove_unselected_groups(self):
        for _ in range(3):
            self._accept()
        self.rollouter.sample_batch(1, 0, 2)
        with self.assertRaises(ValueError):
            self.rollouter.reject_batch_groups(("2",))
        self.assertEqual(self.rollouter.buffer.selected, (0, 1))
        self.assertEqual(len(self.rollouter.buffer.ready), 3)
