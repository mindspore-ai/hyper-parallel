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
"""Collective heap growth decisions and destructive-failure boundaries without hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import heap_manager as manager_module
from hyper_parallel.core.multicore.modules.mega_moe.heap_manager import MegaMoeHeapManager
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace


class TestMegaMoeHeapManager(unittest.TestCase):
    """Exercise capacity planning, ownership checks and live workspace rebinding."""

    def setUp(self) -> None:
        """Bind a CPU-only root with mocked native and NPU lifecycle operations."""
        self.specification = {"local_num_tokens": 4096, "top_k": 8, "hidden_size": 5120,
                              "intermediate_size": 1792, "num_experts": 48, "ep_size": 8,
                              "expert_capacity_factor": 1.0, "dispatch_mode": "push", "capacity_policy": "grow"}
        self.spec = SimpleNamespace(**self.specification, ep_group=None, rank_id=0, routed_slots=32768,
                                    receive_capacity=32768)
        self.addCleanup(patch.stopall)
        patch.object(manager_module._lifecycle, "_shutdown_failed", False).start()
        patch.dict(manager_module.os.environ, {}, clear=True).start()
        patch.object(manager_module.dist, "is_initialized", return_value=False).start()
        self.manager = MegaMoeHeapManager(self.spec, torch.empty(0, dtype=torch.bfloat16), (self.specification,))
        self.resource = SimpleNamespace(spec=self.spec, workspace=MegaMoeWorkspace(shared=True))
        self.state = {"reference_count": 1, "config": {"heap_size_bytes": self.manager.heap_bytes},
                      "active_allocations": []}
        patch.object(manager_module.shmem, "debug_state", side_effect=lambda: self.state).start()
        self.manager.bind(self.resource, self.specification)
        self.free = patch.object(manager_module.shmem, "free").start()
        patch.object(manager_module.shmem, "host_barrier").start()
        patch.object(torch.npu, "is_current_stream_capturing", return_value=False).start()
        self.sync = patch.object(torch.npu, "synchronize").start()
        self.reinitialize = patch.object(manager_module._lifecycle, "_reinitialize", return_value={}).start()

    def test_monotonic_growth_and_fast_path(self) -> None:
        """Do no control work within capacity and grow enough to cover jumps beyond the geometric margin."""
        with patch.object(self.manager, "_rebuild") as rebuild:
            self.manager.ensure_capacity(self.resource, 32768)
            rebuild.assert_not_called()
            for received, capacity, heap_mib in ((32769, 49152, 802), (71920, 71936, 1024), (262144, 262144, 2882)):
                with self.subTest(received=received):
                    self.manager.ensure_capacity(self.resource, received)
                    rebuild.assert_called_with([capacity], heap_mib * 1024**2)

    def test_explicit_budget_uses_minimum_capacity_before_rejecting(self) -> None:
        """Fit a route without growth headroom and preserve the heap when even the minimum will not fit."""
        self.manager.heap_limit = 660 * 1024**2
        with patch.object(self.manager, "_rebuild") as rebuild:
            self.manager.ensure_capacity(self.resource, 32769)
            rebuild.assert_called_once_with([32896], self.manager.heap_limit)
            rebuild.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "explicit heap budget"):
                self.manager.ensure_capacity(self.resource, 40000)
            rebuild.assert_not_called()
            self.assertEqual(self.manager.epoch, 0)

    def test_unmanaged_users_allocations_and_active_leases_keep_old_heap(self) -> None:
        """Reject unsafe ownership before any destructive operation."""
        for reason in ("owner", "allocation", "lease", "capture"):
            with self.subTest(reason=reason):
                self.state["reference_count"] = 2 if reason == "owner" else 1
                self.state["active_allocations"] = [{"allocation_base": 1234}] if reason == "allocation" else []
                self.resource.workspace.in_use = reason == "lease"
                with patch.object(torch.npu, "is_current_stream_capturing", return_value=reason == "capture"):
                    with self.assertRaisesRegex(RuntimeError, "reconfiguration failed"):
                        self.manager.ensure_capacity(self.resource, 40000)
                self.free.assert_not_called()
                self.reinitialize.assert_not_called()
                self.assertEqual(self.manager.state, "ready")

    def test_peer_manifest_mismatch_precedes_teardown(self) -> None:
        """A rank disagreement must not free any still-valid symmetric tensor."""
        self.manager.members = (0, 1)

        def disagree(peers: list, payload: tuple, **_kwargs: object) -> None:
            """Simulate a peer entering the collective with an incompatible manifest."""
            peers[:] = [payload, (("different epoch",), None)]

        with patch.object(manager_module.dist, "all_gather_object", side_effect=disagree):
            with self.assertRaisesRegex(RuntimeError, "differs across EP ranks"):
                self.manager.ensure_capacity(self.resource, 40000)
        self.reinitialize.assert_not_called()
        self.assertEqual(self.manager.state, "ready")

    def test_sync_failure_preserves_heap_but_init_failure_is_terminal(self) -> None:
        """Differentiate failures before destruction from failures after the old heap is gone."""
        self.sync.side_effect = RuntimeError("sync failed")
        with self.assertRaisesRegex(RuntimeError, "sync failed"):
            self.manager.ensure_capacity(self.resource, 40000)
        self.assertEqual(self.manager.state, "ready")
        self.reinitialize.assert_not_called()
        self.sync.side_effect = None
        self.reinitialize.side_effect = RuntimeError("init failed")
        with self.assertRaisesRegex(RuntimeError, "init failed"):
            self.manager.ensure_capacity(self.resource, 40000)
        self.assertEqual(self.manager.state, "failed")
        self.assertEqual(self.manager.epoch, 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.resource.workspace.claim()

    def test_growth_preserves_lazy_reservations_and_logical_workspace(self) -> None:
        """Keep the object held by old autograd contexts while allocating its next generation lazily."""
        other = dict(self.specification, hidden_size=1024, dispatch_mode="pull", capacity_policy="static")
        self.manager._reserve((other,))
        workspace = self.resource.workspace
        expected = self.manager._required_bytes([49152, 32768])
        self.manager.ensure_capacity(self.resource, 40000)
        self.assertIs(self.resource.workspace, workspace)
        self.assertEqual(workspace.capacity_floor, 49152)
        self.assertIsNone(workspace.expert_buffer)
        self.assertEqual(self.manager.entries[1].capacity, 32768)
        self.assertEqual(self.manager.heap_bytes, expected)
        self.assertEqual(self.manager.epoch, 1)
        self.manager.ensure_capacity(self.resource, 1)
        self.reinitialize.assert_called_once_with(expected)

    def test_partial_allocation_failure_never_publishes_new_epoch(self) -> None:
        """Prevent further launches after replacement allocation fails on a participating rank."""
        workspace = self.resource.workspace
        workspace.completion_event = Mock()
        workspace.dtype, workspace.device = torch.bfloat16, torch.device("cpu")
        with patch.object(workspace, "close"), patch.object(workspace, "ensure", side_effect=RuntimeError("OOM")):
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                self.manager.ensure_capacity(self.resource, 40000)
        self.assertEqual(self.manager.state, "failed")
        self.assertEqual(self.manager.epoch, 0)
        self.assertFalse(self.manager.growth_records)
