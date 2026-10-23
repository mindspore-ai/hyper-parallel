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
"""Unit tests for MegaMoe workspace sizing and stream ordering."""

import math
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe import workspace as workspace_module
from hyper_parallel.core.multicore.modules.mega_moe.heap_manager import _HeapManager, _workspace_layout
from hyper_parallel.core.multicore.modules.mega_moe.spec import (
    _resolve_receive_capacity,
)
from hyper_parallel.core.multicore.modules.mega_moe.workspace import (
    _WORKSPACE_ALIGNMENT,
    MegaMoeWorkspace,
    _spec_workspace_bytes,
    _workspace_allocation_specs,
)
from hyper_parallel.core.multicore.scheduler.config import event_workspace_bytes
from tests.common.mark_utils import arg_mark


class TestMegaMoeWorkspaceSizing(unittest.TestCase):
    """Validate SHMEM planning without allocating accelerator memory."""

    def test_pending_backward_prevents_close_without_changing_buffers(self) -> None:
        """An outstanding graph must release its claim before symmetric memory teardown."""
        workspace = workspace_module.MegaMoeWorkspace(shared=False)
        context = Mock()
        workspace.track_graph(context)
        self.assertFalse(workspace.can_close())
        workspace.release_graph(context)
        self.assertTrue(workspace.can_close())
        with (
            patch.object(workspace_module._collective, "exchange") as exchange,
            self.assertRaisesRegex(RuntimeError, "owning HeapManager"),
        ):
            workspace.close()
        exchange.assert_not_called()
        self.assertFalse(workspace.closing)
        workspace.heap_manager = SimpleNamespace(root_group=None, access=nullcontext)
        with (
            patch.object(workspace_module._collective.dist, "is_initialized", return_value=False),
            patch.object(workspace_module._collective.dist, "all_gather_object") as gather,
        ):
            workspace.close()
        gather.assert_not_called()
        self.assertTrue(workspace.closing)

    @staticmethod
    def _specification(initial_capacity_factor):
        """Build a fixed capacity-planning specification."""
        return {
            "local_num_tokens": 128,
            "hidden_size": 16,
            "num_experts": 8,
            "top_k": 2,
            "initial_capacity_factor": 1.25 if initial_capacity_factor is None else initial_capacity_factor,
            "ep_size": 8,
        }

    def test_capacity_planning_defaults_aligns_and_caps_initial_factor(self) -> None:
        """Use one ordered Allocation specification for aligned capacity planning."""
        specification = self._specification(None)
        routed_slots = 256
        expected_capacity = 384
        allocations = _workspace_allocation_specs(
            dispatch_mode="push",
            routed_slots=routed_slots,
            receive_capacity=expected_capacity,
            hidden_size=specification["hidden_size"],
            ep_size=specification["ep_size"],
            num_experts=specification["num_experts"],
            dtype=torch.bfloat16,
        )
        expected_bytes = sum(
            spec.dtype.itemsize * math.prod(spec.shape) + spec.alignment_bytes - 1
            for _, spec in allocations
        )

        actual_capacity = _resolve_receive_capacity(
            specification["initial_capacity_factor"],
            routed_slots,
            specification["ep_size"],
        )
        actual_bytes = _spec_workspace_bytes(specification, torch.bfloat16)

        self.assertEqual(actual_capacity, expected_capacity)
        self.assertEqual(actual_bytes, expected_bytes)
        self.assertEqual(
            tuple(name for name, _ in allocations),
            ("expert_buffer", "routed_buffer", "forward_event_counters", "backward_event_counters"),
        )
        self.assertEqual(_resolve_receive_capacity(1e308, routed_slots, 8), 2048)
        self.assertEqual(
            _resolve_receive_capacity(
                1.5,
                routed_slots,
                specification["ep_size"],
            ),
            384,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_heap_requires_sufficient_aligned_capacity(self) -> None:
        """Feature: explicit heap requires sufficient aligned capacity.
        Description: Configure aligned, undersized, partial-page and zero budgets.
        Expectation: Accept page-aligned capacity while rejecting undersized and partial physical pages.
        """
        spec = {"local_num_tokens": 4096, "hidden_size": 5120, "top_k": 8, "num_experts": 48,
                "ep_size": 4, "initial_capacity_factor": 4.0}
        reference = torch.empty(0, dtype=torch.bfloat16)
        for mib, error in ((1602, None), (1664, None), (1600, RuntimeError), (1603, ValueError), (0, ValueError)):
            with self.subTest(mib=mib), patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_HEAP_SIZE": str(mib * 1024**2)}):
                if error is None:
                    manager = _HeapManager(None, reference)
                    manager.reserve(((spec, _workspace_layout(spec, reference.dtype)),), allow_resize=True)
                    self.assertEqual(manager.heap_bytes, mib * 1024**2)
                else:
                    with self.assertRaises(error):
                        manager = _HeapManager(None, reference)
                        manager.reserve(((spec, _workspace_layout(spec, reference.dtype)),), allow_resize=True)

    def test_allocation_covers_dynamic_events_and_ready_tail(self) -> None:
        """Retain expanded event storage when allocating through the SHMEM API."""
        spec = SimpleNamespace(receive_capacity=128, routed_slots=256, hidden_size=16,
                               ep_size=64, num_experts=1024, dispatch_mode="push")
        workspace = MegaMoeWorkspace(shared=True)
        with (
            patch.object(workspace_module.shmem, "empty", side_effect=lambda shape, **kw: torch.empty(
                shape, dtype=kw["dtype"],
            )) as allocate,
            patch.object(workspace_module.torch.npu, "Event"),
        ):
            workspace.ensure(spec, torch.float32, torch.device("cpu"))
            workspace.ensure(spec, torch.float32, torch.device("cpu"))
        self.assertEqual(allocate.call_count, 4)
        self.assertEqual(workspace.event_counter_bytes, 1104 * 4)
        for tensor in (workspace.forward_event_counters, workspace.backward_event_counters):
            self.assertEqual(tensor.numel(), event_workspace_bytes(spec.ep_size, spec.num_experts))
        for allocation in allocate.call_args_list:
            self.assertEqual(allocation.kwargs["alignment"], _WORKSPACE_ALIGNMENT)
        self.assertEqual(workspace.gmm_workspace.numel(), 512)
        self.assertEqual(workspace.swiglu_grad_workspace.numel(), 512)

    def test_reuse_waits_with_event_without_host_synchronize(self) -> None:
        """Order serial reuse across streams without blocking the CPU."""
        workspace = MegaMoeWorkspace(shared=True)
        completion_event = Mock()
        current_stream = Mock()
        workspace.completion_event = completion_event
        workspace.used = True

        with (
            patch.object(
                workspace_module.torch.npu,
                "current_stream",
                return_value=current_stream,
            ),
            patch.object(workspace_module.torch.npu, "synchronize") as mock_synchronize,
        ):
            workspace.claim()
            workspace.release()

        current_stream.wait_event.assert_called_once_with(completion_event)
        completion_event.record.assert_called_once_with(current_stream)
        mock_synchronize.assert_not_called()


class TestReadyEventWorkspace(unittest.TestCase):
    """Keep each direction's peer generations alive across per-call clears."""

    @patch.object(workspace_module.shmem, "free")
    @patch.object(workspace_module.shmem, "host_barrier")
    def test_initializes_once_and_preserves_ready_tail_per_direction(self, mock_barrier, mock_free) -> None:
        """Clear stale task counters without clearing a later peer signal."""
        workspace = MegaMoeWorkspace(shared=True, event_counter_bytes=1104 * 4)
        events = [torch.full((event_workspace_bytes(64, 1024),), 123, dtype=torch.uint8) for _ in range(2)]
        workspace.forward_event_counters, workspace.backward_event_counters = events
        for forward, tensor in zip((True, False), events):
            with self.subTest(forward=forward):
                self.assertIs(workspace.prepare_event_counters(forward=forward), tensor)
                self.assertEqual(torch.count_nonzero(tensor).item(), 0)
                tensor[:workspace.event_counter_bytes].fill_(51)
                tensor[workspace.event_counter_bytes:].fill_(7)
                workspace.prepare_event_counters(forward=forward)
                self.assertEqual(torch.count_nonzero(tensor[:workspace.event_counter_bytes]).item(), 0)
                self.assertTrue(torch.all(tensor[workspace.event_counter_bytes:] == 7))
        self.assertEqual(mock_barrier.call_count, 2)
        workspace._free_symmetric_tensors()  # pylint: disable=protected-access
        self.assertFalse(workspace.forward_ready_initialized)
        self.assertFalse(workspace.backward_ready_initialized)
        self.assertEqual(mock_free.call_count, 2)

    @patch.object(workspace_module.shmem, "host_barrier")
    def test_single_rank_needs_no_peer_initialization(self, mock_barrier) -> None:
        """Keep EP1 event storage and execution free of a peer-ready barrier."""
        workspace = MegaMoeWorkspace(shared=False)
        workspace.forward_event_counters = torch.ones(workspace.event_counter_bytes, dtype=torch.uint8)
        workspace.prepare_event_counters(forward=True)
        self.assertEqual(torch.count_nonzero(workspace.forward_event_counters).item(), 0)
        mock_barrier.assert_not_called()

    def test_reuse_wait_precedes_claim_without_consuming_lease(self) -> None:
        """A pre-collective wait keeps the release marker valid for claim."""
        workspace = MegaMoeWorkspace(shared=True)
        workspace.completion_event = Mock()
        stream = Mock()
        with patch.object(workspace_module.torch.npu, "current_stream", return_value=stream):
            workspace.wait_for_reuse()
            stream.wait_event.assert_not_called()
            workspace.used = True
            workspace.wait_for_reuse()
            self.assertFalse(workspace.in_use)
            workspace.claim()
            workspace.release()
        self.assertEqual(stream.wait_event.call_count, 2)
        workspace.completion_event.record.assert_called_once_with(stream)


if __name__ == "__main__":
    unittest.main()
