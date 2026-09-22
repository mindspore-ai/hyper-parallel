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
"""CPU regressions for retryable explicit and automatic resource ownership."""

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from torch import Tensor, ones
from torch.autograd import Function

from hyper_parallel.core.multicore import _automatic
from hyper_parallel.core.multicore.modules import module as module_api
from hyper_parallel.core.multicore.modules.mega_moe.function import _release_completed_graph
from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace

# pylint: disable=protected-access


class _Graph(Function):  # pylint: disable=abstract-method
    """Exercise the production lease with a real CPU autograd context."""

    @staticmethod
    # pylint: disable-next=arguments-differ
    def forward(ctx: Any, tensor: Tensor, workspace: MegaMoeWorkspace) -> Tensor:
        """Attach a lease to the autograd context."""
        ctx.workspace = workspace
        workspace.track_graph(ctx)
        return tensor.clone()

    @staticmethod
    # pylint: disable-next=arguments-differ
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        """Release the lease only when backward will not retain the graph."""
        _release_completed_graph(ctx, ctx.workspace)
        return gradient, None


class TestLifecycle(unittest.TestCase):
    """Exercise public cleanup using real membership and mocked native resources."""

    def setUp(self) -> None:
        """Isolate resource ownership and mock distributed initialization."""
        self.manager = module_api._MulticoreResourceManager()
        for patcher in (
            patch.object(module_api, "_RESOURCE_MANAGER", self.manager),
            patch.object(module_api.dist, "is_initialized", return_value=False),
            patch.object(_automatic, "register"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _bind(self):
        """Lazily bind a module to mocked native resources."""
        member = module_api.MulticoreModule(
            resource_specification="spec", resource_compatibility_key="key", resource_scope_key="scope")
        resources = Mock()
        resources.can_close.return_value = True
        resources.lifecycle_signature.return_value = ("test",)
        with patch.object(member, "_create_execution_resources", return_value=resources):
            member._get_execution_resources(SimpleNamespace(device="cpu", dtype="float32"))
        return member, resources

    def test_failed_close_retains_handles_for_retry(self):
        """Failed explicit close keeps its finalizer, handles and retry path."""
        member, resources = self._bind()
        resources.close.side_effect = RuntimeError("busy")
        with self.assertRaisesRegex(RuntimeError, "busy"):
            member.close()
        self.assertIs(member._resource_group.resources, resources)
        self.assertTrue(member._resource_finalizer.alive)
        with self.assertRaisesRegex(RuntimeError, "closing"):
            member._get_execution_resources(None)
        resources.close.side_effect = None
        member.close()
        member.close()
        self.assertEqual(resources.close.call_count, 2)

    def test_remote_failure_keeps_successful_rank_registered(self):
        """Peer resource/runtime failures retain retry ownership without double release."""
        for stage in (1, 2):
            member, resources = self._bind()
            release = self.manager.runtime_release = Mock()
            failures = Mock(side_effect=[None] * (stage - 1) + ["remote failed"])
            with patch.object(_automatic, "exchange", side_effect=lambda value, failures=failures: [
                    value, failures() if value is None else value]):
                with self.assertRaisesRegex(RuntimeError, "remote failed"):
                    member.close()
            self.assertFalse(member._resource_closed)
            self.assertIs(self.manager.runtime_release, release)
            member.close()
            release.assert_called_once_with()
            self.assertEqual(resources.close.call_count, 3 - stage)
            self.assertFalse(self.manager._groups)

    def test_retained_graph_and_partial_free(self):
        """A retained graph blocks close; retry never repeats a successful free."""
        workspace = MegaMoeWorkspace(shared=False)
        output = _Graph.apply(ones(1, requires_grad=True), workspace)
        output.sum().backward(retain_graph=True)
        with self.assertRaisesRegex(RuntimeError, "backward graphs"):
            workspace.close()
        output.sum().backward()
        self.assertTrue(workspace.can_close())
        workspace.expert_buffer, workspace.routed_buffer = Mock(), Mock()
        with patch("hyper_parallel.core.multicore.modules.mega_moe.workspace.shmem.free",
                   side_effect=[None, RuntimeError("free failed"), None]) as free:
            with self.assertRaisesRegex(RuntimeError, "free failed"):
                workspace._free_symmetric_tensors()
            self.assertIsNone(workspace.expert_buffer)
            workspace._free_symmetric_tensors()
        self.assertEqual(free.call_count, 3)

    def test_orphan_pool_reuses_and_evicts_without_explicit_close(self):
        """Repeated model replacement reuses one slot and evicts incompatible orphans."""
        member, resources = self._bind()
        for _ in range(8):
            del member
            member, _ = self._bind()
            self.assertIs(member._resource_group.resources, resources)
            self.assertEqual(len(self.manager._groups), 1)
        self.manager.runtime_release = Mock()
        member._resource_group.compatibility_key = "different"
        del member
        replacement, _ = self._bind()
        resources.close.assert_called_once_with()
        self.manager.runtime_release.assert_not_called()
        release = self.manager.runtime_release
        replacement.close()
        release.assert_called_once_with()

    def test_workspace_retry_keeps_stages_aligned_and_rejects_claim(self):
        """A local free failure retains the event and re-enters both barriers on retry."""
        workspace = MegaMoeWorkspace(shared=False, completion_event=Mock(), gmm_workspace=Mock())
        workspace.gmm_workspace.untyped_storage.return_value.resize_.side_effect = [RuntimeError("busy"), None]
        with patch("hyper_parallel.core.multicore.modules.mega_moe.workspace.torch.npu", create=True), patch(
                "hyper_parallel.core.multicore.modules.mega_moe.workspace.shmem.host_barrier") as barrier:
            with self.assertRaisesRegex(RuntimeError, "busy"):
                workspace.close()
            self.assertIsNotNone(workspace.completion_event)
            with self.assertRaisesRegex(RuntimeError, "closing"):
                workspace.claim()
            workspace.close()
            with patch.object(_automatic, "exchange", side_effect=lambda value: [
                    value, False if isinstance(value, bool) else value]):
                workspace.close()
        self.assertEqual(barrier.call_count, 6)
        self.assertIsNone(workspace.completion_event)

    def test_orphan_reuse_requires_all_ranks_idle(self):
        """A remote live owner or a local graph blocks reuse without blocking allocation."""
        member, resources = self._bind()
        del member
        resources.can_close.return_value = False
        replacement, _ = self._bind()
        self.assertIsNot(replacement._resource_group.resources, resources)
        resources.can_close.return_value = True
        del replacement
        with patch.object(self.manager, "_exchange", side_effect=lambda value: [
                value, (value[0], value[1], [(False, compatible) for _, compatible in value[2]])]):
            replacement, _ = self._bind()
        self.assertIsNot(replacement._resource_group.resources, resources)
        resources.close.assert_not_called()

    def test_backend_hook_orders_cleanup_and_preserves_retry(self):
        """Backend interception covers aliases, unrelated groups, exit and release errors."""
        events = []
        backend = type("Backend", (), {"shutdown": Mock(side_effect=lambda *_: events.append("backend"))})
        dependency, unrelated = backend(), backend()
        cleanup = _automatic._Cleanup()
        callback = Mock(side_effect=lambda: events.append("resources"))
        with patch.object(_automatic.c10d, "ProcessGroup", backend), patch.object(_automatic.atexit, "register"):
            cleanup.install()
            cleanup.install()
            cleanup.callback = callback
            cleanup.dependencies.add(dependency)
            unrelated.shutdown()
            callback.assert_not_called()
            callback.side_effect = RuntimeError("free failed")
            with self.assertRaisesRegex(RuntimeError, "free failed"):
                dependency.shutdown()
            self.assertEqual(events, ["backend"])
            callback.side_effect = lambda: events.append("resources")
            dependency.shutdown()
            cleanup.at_exit()
            self.assertEqual(events, ["backend", "resources", "backend"])
            self.assertIsNone(cleanup.callback)
