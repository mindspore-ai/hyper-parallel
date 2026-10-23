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
"""Unit tests for MegaMoe workspace capacity and SHMEM Heap policy."""

from __future__ import annotations

import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore import _collective, shmem
from hyper_parallel.core.multicore.modules.mega_moe import heap_manager as heap_manager_module
from hyper_parallel.core.multicore.modules.mega_moe.heap_manager import (
    _HeapManager,
    _WorkspaceLayout,
    _get_heap_manager,
    _workspace_layout,
)
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_parallel.core.multicore.modules.mega_moe.workspace import (
    MegaMoeWorkspace,
    _workspace_allocation_specs,
)
from hyper_parallel.core.multicore.shmem import _lifecycle
from tests.common.mark_utils import arg_mark


class TestMegaMoeHeapManager(unittest.TestCase):
    """Characterize MegaMoe capacity policy independently of NPU allocation."""

    # These tests deliberately inspect the private manager and lifecycle transaction.
    # pylint: disable=protected-access

    def _automatic_workspace(self, mode: str) -> tuple:
        """Materialize a small symmetric layout using CPU storage for restart validation."""
        specification = {"local_num_tokens": 128, "hidden_size": 16, "intermediate_size": 8,
                         "num_experts": 4, "top_k": 2, "ep_size": 2, "dispatch_mode": mode,
                         "initial_capacity_factor": 1.25 if mode == "push" else None, "auto_tokens": True}
        layout = _workspace_layout(specification, torch.bfloat16)
        manager = self._manager_with_reservation(specification, layout)
        spec = MegaMoeSpec(128, 16, 8, 4, 2, specification["initial_capacity_factor"],
                           layout.receive_capacity, 2, None, 0, 24, dispatch_mode=mode,
                           capacity_growth_factor=1.25 if mode == "push" else None, auto_tokens=True)
        workspace = MegaMoeWorkspace(shared=True)
        workspace.capacity_floor = manager.bind(workspace, spec, specification)
        for name, allocation in self._allocations(spec, layout.receive_capacity):
            setattr(workspace, name, torch.empty(allocation.shape, dtype=allocation.dtype))
        workspace.completion_event = object()
        return manager, workspace, spec

    @staticmethod
    def _cpu_restart(**kwargs: Any) -> _lifecycle._RestartResult:
        """Run the real preflight then provide ordinary replacement tensors."""
        kwargs["preflight_hook"]()
        tensors = tuple(torch.empty(item.target.shape, dtype=item.target.dtype) for item in kwargs["replacements"])
        return _lifecycle._RestartResult(tensors, {})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_source_and_receive_growth_commit_together_and_reuse(self) -> None:
        """Feature: Automatic MegaMoe token storage.

        Description: Both transports grow source rows without modifying retained plans or shrinking on reuse.
        Expectation: Source and receive capacity commit together and smaller inputs reuse the committed layout.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                manager, workspace, retained_spec = self._automatic_workspace(mode)
                old_routed = workspace.routed_buffer
                with patch.object(shmem, "_restart", side_effect=self._cpu_restart) as restart:
                    committed = manager.ensure_token_capacity(workspace, 512, 1700)
                    self.assertEqual(restart.call_count, 1)
                    self.assertEqual(committed.local_num_tokens, 512)
                    self.assertEqual(workspace.routed_slots, 1024)
                    self.assertEqual(tuple(workspace.routed_buffer.shape), (1024, 16))
                    self.assertIsNot(workspace.routed_buffer, old_routed)
                    if mode == "pull":
                        self.assertEqual(tuple(workspace.source_buffer.shape), (1024, 16))
                    else:
                        self.assertGreaterEqual(workspace.expert_buffer.shape[0], 1700)
                    self.assertEqual(retained_spec.local_num_tokens, 128)
                    self.assertEqual(tuple(old_routed.shape), (256, 16))
                    self.assertIs(manager.ensure_token_capacity(workspace, 128, 1), committed)
                    self.assertEqual(restart.call_count, 1)
                self.assertEqual(manager.epoch, 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_budget_failure_preserves_committed_source_and_receive_capacity(self) -> None:
        """Feature: Automatic MegaMoe token storage.

        Description: An over-budget source growth fails before any restart or specification mutation.
        Expectation: A rejected layout preserves both committed capacities and the current heap state.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                manager, workspace, spec = self._automatic_workspace(mode)
                manager.heap_limit = manager.heap_bytes
                original = workspace.routed_buffer
                with (patch.object(shmem, "_restart") as restart,
                      self.assertRaisesRegex(RuntimeError, "explicit Heap budget")):
                    manager.ensure_token_capacity(workspace, 131072, 262144)
                restart.assert_not_called()
                self.assertIs(manager.entries[0].bound_spec, spec)
                self.assertIs(workspace.routed_buffer, original)
                self.assertEqual(manager.epoch, 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_growth_falls_back_to_required_rows_within_heap_budget(self) -> None:
        """Feature: Automatic MegaMoe token storage.

        Description: A geometric source reservation may exceed the budget while the actual request still fits.
        Expectation: The required aligned layout fits without applying the larger growth factor.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                manager, workspace, _ = self._automatic_workspace(mode)
                with patch.object(shmem, "_restart", side_effect=self._cpu_restart):
                    manager.ensure_token_capacity(workspace, 16384, 0)
                    manager.heap_limit = manager.heap_bytes
                    entry = manager.entries[0]
                    entry.bound_spec = replace(entry.bound_spec, capacity_growth_factor=2.0)
                    committed = manager.ensure_token_capacity(workspace, 17024, 0)
                self.assertEqual(committed.local_num_tokens, 17024)
                self.assertEqual(workspace.routed_slots, 34048)
                self.assertEqual(manager.heap_bytes, manager.heap_limit)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_growth_rejects_an_active_pull_lease_before_replacement(self) -> None:
        """Feature: Automatic MegaMoe token storage.

        Description: A pull source cannot be invalidated while permutation or a kernel owns it.
        Expectation: An active source lease rejects growth before replacing any allocation.
        """
        manager, workspace, spec = self._automatic_workspace("pull")
        workspace.in_use = True
        with (patch.object(shmem, "_restart", side_effect=self._cpu_restart),
              patch.object(_lifecycle, "_runtime_is_healthy", return_value=True),
              self.assertRaisesRegex(RuntimeError, "active workspace lease")):
            manager.ensure_token_capacity(workspace, 512, 1)
        self.assertIs(manager.entries[0].bound_spec, spec)
        self.assertEqual(manager.state, "ready")

    def setUp(self) -> None:
        """Install one-rank lifecycle and configuration boundaries."""
        self.reference = SimpleNamespace(
            device="npu:0",
            dtype=torch.bfloat16,
            element_size=lambda: 2,
        )
        self.root_group = object()
        self.root_patch = patch.object(_lifecycle, "root_members", return_value=(0,))
        self.dist = SimpleNamespace(
            is_initialized=Mock(return_value=True),
            get_world_size=Mock(return_value=1),
            all_gather_object=Mock(side_effect=lambda output, value, **_kw: output.__setitem__(slice(None), [value])),
        )
        self.collective_patch = patch.object(_collective, "dist", self.dist)
        self.environment_patch = patch.dict(heap_manager_module.os.environ, {}, clear=True)
        self.root_patch.start()
        self.collective_patch.start()
        self.environment_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.collective_patch.stop)
        self.addCleanup(self.environment_patch.stop)
        self.addCleanup(self._reset_manager)
        self._reset_manager()

    @staticmethod
    def _reset_manager() -> None:
        """Reset the process singleton between isolated tests."""
        heap_manager_module._HEAP_MANAGER = None  # pylint: disable=protected-access

    @staticmethod
    def _layout(required_bytes: int, capacity: int, name: str = "layout") -> _WorkspaceLayout:
        """Build a deterministic layout without duplicating the capacity formula."""
        return _WorkspaceLayout(required_bytes, (("name", name),), capacity)

    def _manager_with_reservation(
        self,
        specification: dict,
        layout: _WorkspaceLayout,
    ) -> _HeapManager:
        """Create a manager and submit one reservation."""
        manager = _HeapManager(self.root_group, self.reference)
        manager.reserve(((specification, layout),), allow_resize=True)
        return manager

    @staticmethod
    def _allocations(
        specification: Any,
        capacity: int,
    ) -> tuple[tuple[str, _lifecycle._AllocationSpec], ...]:  # pylint: disable=protected-access
        """Return one workspace's ordered symmetric Allocation specifications."""
        return _workspace_allocation_specs(
            dispatch_mode=specification.dispatch_mode,
            routed_slots=specification.routed_slots,
            receive_capacity=capacity,
            hidden_size=specification.hidden_size,
            ep_size=specification.ep_size,
            num_experts=specification.num_experts,
            dtype=torch.bfloat16,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_reservations_define_layout_and_roll_back_failed_additions(self) -> None:
        """Align aggregate reservations and roll back only a failed identity-based addition."""
        reservations = (
            ({"name": "first"}, self._layout(1_500_000, 64, "first")),
            ({"name": "second"}, self._layout(1_000_000, 32, "second")),
        )
        manager = _HeapManager(self.root_group, self.reference)
        added = manager.reserve(reservations, allow_resize=True)

        self.assertEqual(manager.heap_bytes, 4 * 1024**2)
        self.assertEqual(added, tuple(item[0] for item in reservations))
        self.assertIs(self.dist.all_gather_object.call_args.kwargs["group"], self.root_group)
        configuration, failure = self.dist.all_gather_object.call_args.args[1]
        self.assertEqual(configuration, (None, str(self.reference.dtype), tuple(item[1] for item in reservations)))
        self.assertIsNone(failure)

        first = {"shape": 1}
        equal = {"shape": 1}
        manager = self._manager_with_reservation(first, self._layout(512, 1))

        with (
            patch.object(self.dist, "get_world_size", side_effect=RuntimeError("agreement failed")),
            self.assertRaisesRegex(RuntimeError, "agreement failed"),
        ):
            manager.reserve(((equal, self._layout(512, 1)),), allow_resize=True)
        self.assertEqual(len(manager.entries), 1)
        self.assertIs(manager.entries[0].specification, first)

        added = manager.reserve(((equal, self._layout(512, 1)),), allow_resize=True)
        self.assertEqual(added, (equal,))
        self.assertEqual(
            self.dist.all_gather_object.call_args.args[1],
            (tuple(entry.layout for entry in manager.entries), None),
        )
        manager.rollback_reservations(added, None)
        self.assertEqual([entry.specification for entry in manager.entries], [first])

        workspace = Mock()
        manager.bind(workspace, SimpleNamespace(name="fixed"), first)
        with self.assertRaisesRegex(RuntimeError, "fixed SHMEM layout"):
            manager.reserve(((equal, self._layout(3 * 1024**2, 2)),), allow_resize=False)
        self.assertEqual([entry.specification for entry in manager.entries], [first])

        self._assert_configuration_failures(reservations)

    def _assert_configuration_failures(self, reservations: tuple) -> None:
        """Check local, peer and communication failures through the real Heap agreement."""
        for scenario in ("local", "local peers", "remote", "mismatch", "query", "gather"):
            with self.subTest(scenario=scenario):
                manager = _HeapManager(self.root_group, self.reference)
                local_error = ValueError("invalid heap") if scenario.startswith("local") else None
                communication_error = RuntimeError("communication unavailable")

                def gather(
                    output: list[Any], payload: Any, *, group: Any,
                    case: str = scenario, gather_error: RuntimeError = communication_error,
                ) -> None:
                    """Return business agreement or a peer failure on the manager's Root."""
                    self.assertIs(group, self.root_group)
                    if case == "gather":
                        raise gather_error
                    output[:] = [payload]
                    if case == "local peers":
                        output.append(payload)
                    elif case == "remote":
                        output.append((None, ("ValueError", "peer invalid heap")))
                    elif case == "mismatch":
                        output.append(("different configuration", None))

                with (
                    patch.object(manager, "_configure", wraps=manager._configure, side_effect=local_error),
                    patch.object(self.dist, "get_world_size",
                                 return_value=2 if scenario in ("local peers", "remote", "mismatch") else 1,
                                 side_effect=communication_error if scenario == "query" else None),
                    patch.object(self.dist, "all_gather_object", side_effect=gather),
                    self.assertRaises(ValueError if scenario == "local" else RuntimeError) as raised,
                ):
                    manager.reserve(reservations, allow_resize=True)
                if scenario == "local":
                    self.assertIs(raised.exception, local_error)
                else:
                    self.assertIn("SHMEM Heap failed during Configuration", str(raised.exception))
                    self.assertIs(raised.exception.__cause__,
                                  communication_error if scenario in ("query", "gather") else local_error)
                if scenario in ("query", "gather"):
                    self.assertIsInstance(raised.exception, _collective.CollectiveConvergenceError)
                if scenario == "remote":
                    self.assertIn("root_rank=1", str(raised.exception))
                    self.assertTrue(raised.exception.__suppress_context__)
                if scenario == "local peers":
                    self.assertIn("root_ranks=[0, 1]", str(raised.exception))
                self.assertEqual(manager.entries, [])
                self.assertFalse(manager._configured)
                self.assertEqual(manager.heap_bytes, 0)

    def test_manager_owns_and_reuses_one_retryable_runtime_reference(self) -> None:
        """Reuse an attached manager and commit detach only after its sole release succeeds."""
        manager = self._manager_with_reservation({"name": "runtime"}, self._layout(512, 1))
        with (
            patch.object(heap_manager_module.shmem, "acquire") as acquire,
            patch.object(heap_manager_module.shmem, "release") as release,
        ):
            self.assertTrue(manager.attach_runtime())
            self.assertFalse(manager.attach_runtime())
            acquire.assert_called_once_with(self.root_group, heap_size_bytes=2 * 1024**2)
            release.side_effect = RuntimeError("busy")
            with self.assertRaisesRegex(RuntimeError, "busy"):
                manager.detach_runtime()
            self.assertTrue(manager.runtime_attached)
            release.side_effect = None
            manager.detach_runtime()
        self.assertFalse(manager.runtime_attached)
        self.assertEqual(release.call_count, 2)

        first = _get_heap_manager(None, self.reference)
        first._runtime_attached = True
        self.assertIs(_get_heap_manager(None, self.reference), first)
        first._runtime_attached = False
        self.assertIsNot(_get_heap_manager(None, self.reference), first)

    def test_restart_binds_materialized_workspaces_and_preserves_lazy_entries(self) -> None:
        """Bind ordered result slices for shared and private workspaces around one lazy entry."""
        specifications = ({"name": "push"}, {"name": "lazy"}, {"name": "pull"})
        current_layouts = [
            self._layout(512, 1, "push"),
            self._layout(256, 1, "lazy"),
            self._layout(1024, 2, "pull"),
        ]
        target_layouts = [
            self._layout(1024, 3, "push"),
            self._layout(256, 1, "lazy"),
            self._layout(1536, 4, "pull"),
        ]
        manager = _HeapManager(self.root_group, self.reference)
        manager.reserve(
            tuple(zip(specifications, current_layouts)),
            allow_resize=True,
        )
        manager._runtime_attached = True
        bound_specs = (
            SimpleNamespace(dispatch_mode="push", routed_slots=2, hidden_size=4, ep_size=1, num_experts=2),
            SimpleNamespace(dispatch_mode="push", routed_slots=2, hidden_size=4, ep_size=1, num_experts=2),
            SimpleNamespace(dispatch_mode="pull", routed_slots=3, hidden_size=4, ep_size=1, num_experts=2),
        )
        workspaces = [MegaMoeWorkspace(shared=True), MegaMoeWorkspace(shared=False), MegaMoeWorkspace(shared=False)]
        materialized_indices = (0, 2)
        old_tensors = []
        for index in materialized_indices:
            workspace = workspaces[index]
            spec = bound_specs[index]
            allocations = self._allocations(spec, current_layouts[index].receive_capacity)
            current_tensors = tuple(torch.empty(target.shape, dtype=target.dtype) for _, target in allocations)
            for tensor, (field_name, _) in zip(current_tensors, allocations):
                setattr(workspace, field_name, tensor)
            workspace.completion_event = object()
            workspace.capacity_floor = current_layouts[index].receive_capacity
            old_tensors.extend(current_tensors)
        for workspace, bound_spec, specification in zip(workspaces, bound_specs, specifications):
            manager.bind(workspace, bound_spec, specification)
        replacements, bindings, agreement = manager._describe_restart(target_layouts)
        with self.assertRaisesRegex(RuntimeError, "entries changed"):
            manager._restart_preflight(target_layouts[:-1], replacements, bindings, agreement)
        stages = {
            "preflight_ms": 1.0, "quiesce_ms": 2.0, "release_ms": 3.0, "finalize_ms": 4.0,
            "bootstrap_ms": 5.0, "initialize_ms": 6.0, "restore_ms": 7.0,
        }

        def restart(**kwargs: Any) -> _lifecycle._RestartResult:
            """Validate the declarative request and return replacement Tensors."""
            heap_size_bytes = kwargs["heap_size_bytes"]
            replacements = kwargs["replacements"]
            preflight_hook = kwargs["preflight_hook"]
            self.assertEqual(heap_size_bytes, 2 * 1024**2)
            self.assertEqual(len(replacements), 8)
            for field, value, message in (
                ("in_use", True, "active workspace lease"),
                ("closing", True, "closing workspace"),
                ("completion_event", None, "lazy workspace"),
                ("expert_buffer", torch.empty(1), "mapping changed"),
            ):
                with self.subTest(field=field), patch.object(workspaces[0], field, value):
                    with self.assertRaisesRegex(RuntimeError, message):
                        preflight_hook()
            agreement = preflight_hook()
            self.assertEqual([entry.is_materialized for entry in agreement.entries], [True, False, True])
            result_tensors = tuple(
                torch.empty(replacement.target.shape, dtype=replacement.target.dtype)
                for replacement in replacements
            )
            self.assertNotIn("commit_hook", kwargs)
            return _lifecycle._RestartResult(result_tensors, stages)  # pylint: disable=protected-access

        events = []
        validate = manager._validate_restart_result
        bind = manager._bind_restart_result

        def validate_result(*args: Any, **kwargs: Any) -> None:
            """Record local validation before invoking the manager implementation."""
            events.append(("validate", manager.epoch))
            validate(*args, **kwargs)

        def bind_result(*args: Any, **kwargs: Any) -> None:
            """Record result binding before delegating to the manager implementation."""
            events.append(("bind", manager.epoch))
            bind(*args, **kwargs)

        with (
            patch.object(shmem, "_restart", side_effect=restart),
            patch.object(manager, "_validate_restart_result", side_effect=validate_result),
            patch.object(manager, "_bind_restart_result", side_effect=bind_result),
            patch.object(heap_manager_module.time, "perf_counter", side_effect=(10.0, 10.125)),
        ):
            manager._restart(target_layouts, 2 * 1024**2)

        rebound_tensors = []
        for index in materialized_indices:
            workspace = workspaces[index]
            spec = bound_specs[index]
            targets = self._allocations(spec, target_layouts[index].receive_capacity)
            rebound_tensors.extend(getattr(workspace, field_name) for field_name, _ in targets)
            self.assertEqual(
                [(tuple(getattr(workspace, field_name).shape), getattr(workspace, field_name).dtype)
                 for field_name, _ in targets],
                [(target.shape, target.dtype) for _, target in targets],
            )
        self.assertTrue(all(actual is not old for actual, old in zip(rebound_tensors, old_tensors)))
        self.assertIsNone(workspaces[1].completion_event)
        self.assertEqual(events, [("validate", 0), ("bind", 0)])
        self.assertEqual(manager.epoch, 1)
        self.assertEqual(manager.growth_records[-1]["stages_ms"], stages)
        self.assertEqual(manager.growth_records[-1]["total_ms"], 125.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_capacity_fast_path_and_explicit_limit_fallback(self) -> None:
        """Feature: receive capacity budgeting.
        Description: Keep the sufficient-capacity path and fall back to minimum rows within the heap limit.
        Expectation: Reuse needs no lock; growth preserves source reservation and fits the explicit budget.
        """
        manager = _HeapManager(self.root_group, self.reference)
        workspace = SimpleNamespace(capacity_floor=2)
        with patch.object(manager, "access") as access:
            manager.ensure_capacity(workspace, 2)
        access.assert_not_called()

        specification = {"shape": "dynamic"}
        manager = self._manager_with_reservation(specification, self._layout(512, 1))
        manager.heap_limit = 2 * 1024**2
        manager.heap_bytes = manager.heap_limit
        workspace = Mock(capacity_floor=1)
        workspace.describe_restart_allocations.return_value = ()
        workspace.validate_restart_allocations.return_value = False
        manager.bind(workspace, SimpleNamespace(name="bound"), specification)

        def layout(specification: dict, dtype: torch.dtype, capacity: int,
                   token_capacity: int | None) -> _WorkspaceLayout:
            """Return a capacity-dependent layout for fallback selection."""
            del specification, dtype, token_capacity
            required = 3 * 1024**2 if capacity == 3 else 1024
            return self._layout(required, capacity)

        def restart(**kwargs: Any) -> _lifecycle._RestartResult:
            """Run the preflight hook and return an empty successful restart result."""
            kwargs["preflight_hook"]()
            self.assertNotIn("commit_hook", kwargs)
            return _lifecycle._RestartResult((), {})

        with (
            patch.object(heap_manager_module, "_grow_receive_capacity", return_value=(3, 2)),
            patch.object(heap_manager_module, "_workspace_layout", side_effect=layout),
            patch.object(shmem, "_restart", side_effect=restart),
        ):
            manager.ensure_capacity(workspace, 2)
        self.assertEqual(manager.entries[0].layout.receive_capacity, 2)
        self.assertEqual(workspace.capacity_floor, 2)

    def test_resize_lock_rejects_concurrent_access(self) -> None:
        """Reject another submission while one capacity change owns the manager lock."""
        specification = {"shape": "dynamic"}
        manager = self._manager_with_reservation(specification, self._layout(512, 1))
        workspace = Mock(capacity_floor=1)
        workspace.describe_restart_allocations.return_value = ()
        workspace.validate_restart_allocations.return_value = False
        manager.bind(workspace, SimpleNamespace(name="bound"), specification)
        entered = threading.Event()
        resume = threading.Event()

        def restart(**kwargs: Any) -> _lifecycle._RestartResult:
            """Hold the restart call while another thread probes manager access."""
            kwargs["preflight_hook"]()
            entered.set()
            if not resume.wait(timeout=2):
                raise RuntimeError("test did not release restart")
            self.assertNotIn("commit_hook", kwargs)
            return _lifecycle._RestartResult((), {})

        with (
            patch.object(heap_manager_module, "_grow_receive_capacity", return_value=(2, 2)),
            patch.object(heap_manager_module, "_workspace_layout", return_value=self._layout(1024, 2)),
            patch.object(shmem, "_restart", side_effect=restart),
        ):
            thread = threading.Thread(target=manager.ensure_capacity, args=(workspace, 2))
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=2))
                with self.assertRaisesRegex(RuntimeError, "concurrent Heap operations"), manager.access():
                    self.fail("Concurrent access entered a restart transaction")
            finally:
                resume.set()
                thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_restart_failure_boundary_controls_manager_state(self) -> None:
        """Preflight failures are retryable, and destructive restart failures are terminal."""
        initial = self._layout(512, 1)
        target = self._layout(1024, 2)
        manager = self._manager_with_reservation({"name": "failure"}, initial)
        with (
            patch.object(shmem, "_restart", side_effect=RuntimeError("preflight failed")),
            patch.object(_lifecycle, "_runtime_is_healthy", return_value=True),
            self.assertRaisesRegex(RuntimeError, "preflight failed"),
        ):
            manager._restart([target], 2 * 1024**2)
        self.assertEqual((manager.state, manager.entries[0].layout, manager.epoch), ("ready", initial, 0))

        with (
            patch.object(shmem, "_restart", side_effect=RuntimeError("destructive failed")),
            patch.object(_lifecycle, "_runtime_is_healthy", return_value=False),
            self.assertRaisesRegex(RuntimeError, "destructive failed"),
        ):
            manager._restart([target], 2 * 1024**2)
        self.assertEqual(manager.state, "failed")
        with self.assertRaisesRegex(RuntimeError, "restart the process"), manager.access():
            self.fail("A failed SHMEM Heap allowed another operation")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_local_restart_failures_do_not_publish_business_state(self) -> None:
        """Feature: Local restart result commit.
        Description: Inject validation and binding defects after the underlying SHMEM restart.
        Expectation: Cause and local-failure diagnostics remain distinct; neither publishes layout or epoch.
        """
        for validation in (True, False):
            with self.subTest(validation=validation):
                initial, target = self._layout(512, 1), self._layout(1024, 2)
                manager = self._manager_with_reservation({"name": "local failure"}, initial)
                message = "invalid local restart result" if validation else "binding failed"
                error = RuntimeError(message)
                with (patch.object(shmem, "_restart", return_value=_lifecycle._RestartResult((), {})),
                      patch.object(manager, "_validate_restart_result", side_effect=error if validation else None),
                      patch.object(manager, "_bind_restart_result", side_effect=None if validation else error) as bind,
                      self.assertRaises(RuntimeError) as raised):
                    manager._restart([target], 2 * 1024**2)
                text = str(raised.exception)
                self.assertIn(message, text)
                self.assertIn("result validation failed" if validation else "workspace binding failed", text)
                self.assertNotIn("workspace binding failed" if validation else "result validation failed", text)
                for fragment in (("local implementation error", "cannot be rolled back", "terminate") if validation
                                 else ("implementation error", "terminate")):
                    self.assertIn(fragment, text)
                self.assertNotIn("all ranks", text)
                self.assertIs(raised.exception.__cause__, error)
                self.assertEqual(manager.state, "failed")
                self.assertEqual((manager.entries[0].layout, manager.epoch, manager.growth_records), (initial, 0, []))
                self.assertEqual(bind.call_count, int(not validation))


if __name__ == "__main__":
    unittest.main()
