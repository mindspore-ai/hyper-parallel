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
"""CPU-only state, error admission, and lifetime checks for parked W storage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from hyper_parallel.distributed.expert_parallel.moonep.fused import weights
from tests.common.mark_utils import arg_mark
from tests.ut.distributed.expert_parallel.moonep.test_pool import _Streams


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestWeightRetirement(unittest.TestCase):
    """Use real CPU storage and mock only stream/communication boundaries."""

    def setUp(self) -> None:
        """Create a two-rank domain without creating a process group or device."""
        self.streams = _Streams()
        self.distributed = Mock()
        self.distributed.get_world_size.return_value = 2
        self.enterContext(patch.object(torch, "npu", self.streams, create=True))
        self.records = self.enterContext(patch.object(torch.Tensor, "record_stream"))
        self.enterContext(patch.object(weights, "dist", self.distributed))
        self.shapes = ((2, 4), (2, 2))
        self.home = (torch.ones((1, 2, 4), dtype=torch.bfloat16), torch.ones((1, 2, 2), dtype=torch.bfloat16))

    def _domain(self, *, capacity: int = 1) -> weights.TransferDomain:
        domain = weights.TransferDomain(object(), 0, torch.device("cpu"), quota_bytes=64,
                                        control_group=object())
        domain.admit(1, capacity, self.shapes, (True, False))
        return domain

    def _lease(self, domain: weights.TransferDomain) -> weights.WeightLease:
        slot, generation, remote = domain._claim_slot((0,))
        return weights.WeightLease(remote, Mock(), domain._lease_owner, slot, generation)

    def test_drop_refs_keeps_old_view_bytes_fences_and_generation(self) -> None:
        """Retiring never invalidates external queued views or drains a stream."""
        domain = self._domain()
        lease = self._lease(domain)
        view = domain._pool.slots[0].matrices["gate"]
        view.fill_(5)
        saved = view.clone()
        domain.release_lease(lease)
        slot = domain._pool.slots[0]
        generation, event = slot.generation, slot.last_read
        before = domain.reserved_bytes
        stream_calls = list(domain.stream.mock_calls)
        compute_calls = list(self.streams.compute.mock_calls)
        domain.drain = Mock(side_effect=AssertionError("retire must not drain"))

        self.assertEqual(domain.retire_storage(), before)
        self.assertEqual(domain.reserved_bytes, 0)
        self.assertTrue(domain.parked)
        self.assertFalse(domain.has_free_slot)
        self.assertEqual(domain.retire_storage(), 0)
        self.assertEqual(slot.generation, generation)
        self.assertIs(slot.last_read, event)
        self.assertEqual(domain.stream.mock_calls, stream_calls)
        self.assertEqual(self.streams.compute.mock_calls, compute_calls)
        self.distributed.all_reduce.assert_not_called()
        torch.testing.assert_close(view, saved)
        domain.drain.assert_not_called()

    def test_restore_preserves_slot_identity_and_waits_old_last_read(self) -> None:
        """New storage follows the original slot fence on the first new borrow."""
        domain = self._domain()
        lease = self._lease(domain)
        domain.release_lease(lease)
        slot = domain._pool.slots[0]
        event, generation = slot.last_read, slot.generation
        old = slot.matrices["gate"]
        domain.retire_storage()

        domain.restore_storage()
        self.assertIs(domain._pool.slots[0], slot)
        self.assertIs(slot.last_read, event)
        self.assertEqual(slot.generation, generation)
        self.assertFalse(domain.parked)
        self.assertTrue(domain.has_free_slot)
        self.assertNotEqual(slot.matrices["gate"].data_ptr(), old.data_ptr())
        self.assertTrue(slot.matrices["gate"].transpose(-2, -1).is_contiguous())
        self.distributed.all_reduce.assert_called_once()
        self.assertIs(self.distributed.all_reduce.call_args.kwargs["group"], domain.control_group)
        again = self._lease(domain)
        domain.stream.wait_event.assert_called_with(event)
        self.assertEqual(again.generation, generation + 1)

    def test_all_six_partial_allocation_failures_converge_and_clear(self) -> None:
        """Every failure boundary clears new tensors and poisons before payload."""
        original = torch.empty
        for failure in range(1, 7):
            with self.subTest(failure=failure):
                domain = self._domain()
                domain.retire_storage()
                references, calls = [], 0

                def _allocate(*args: Any, **kwargs: Any) -> torch.Tensor:
                    nonlocal calls
                    calls += 1
                    if calls == failure:
                        raise MemoryError("partial W restore")
                    value = original(*args, **kwargs)
                    references.append(weakref.ref(value))
                    return value

                self.distributed.reset_mock()
                with patch.object(torch, "empty", side_effect=_allocate):
                    with self.assertRaisesRegex(RuntimeError, "replica preparation failed"):
                        domain.restore_storage()
                self.assertTrue(domain.poisoned)
                self.assertTrue(domain.parked)
                self.assertEqual(domain.reserved_bytes, 0)
                self.assertEqual(domain.buffers, {})
                self.assertTrue(all(not slot.matrices for slot in domain._pool.slots))
                self.distributed.all_reduce.assert_called_once()
                self.distributed.all_to_all_single.assert_not_called()
                self.assertTrue(all(reference() is None for reference in references))

    def test_remote_failure_clears_successful_local_allocations(self) -> None:
        """A peer error poisons this successful rank without any W payload."""
        domain = self._domain()
        domain.retire_storage()
        self.distributed.all_reduce.side_effect = lambda status, **_kwargs: status.fill_(1)
        with self.assertRaisesRegex(RuntimeError, "replica preparation failed"):
            domain.restore_storage()
        self.assertEqual(domain.reserved_bytes, 0)
        self.assertTrue(domain.poisoned)
        self.assertTrue(domain.parked)
        self.distributed.all_to_all_single.assert_not_called()


    def _runtime(self) -> tuple[execution.SparseMoonEPRuntime, tuple]:
        with patch.object(execution.dist, "get_rank", return_value=0):
            experts = [execution.SparseMoonEPExecutor(torch.nn.Module(), Mock(return_value=self.home),
                       ep_group=object(), ep_size=2, num_experts=2, top_k=1) for _ in range(2)]
        runtime = execution.SparseMoonEPRuntime(experts)
        runtime.arena = PhaseArena("cpu")
        runtime.workspace = SimpleNamespace(device=torch.device("cpu"), control_group=object(), metadata={})
        runtime.weight_domain = self._domain()
        route = SimpleNamespace(active_experts=(0,))
        for expert in experts:
            expert.shapes, expert.transposed = self.shapes, (True, False)
            key = runtime.new_key(expert)
            runtime.executions[key] = expert, SimpleNamespace(route=route, weights=())
        return runtime, tuple(runtime.trace)

    def test_final_original_retirement_skips_early_replay_and_no_grad(self) -> None:
        """Only the final original training F parks W; saved tensors are untouched."""
        runtime, keys = self._runtime()
        runtime.retire_forward_scratch(keys[0], requires_backward=True)
        self.assertFalse(runtime.weight_domain.parked)
        with patch.object(execution, "is_replaying", return_value=True):
            runtime.retire_forward_scratch(keys[-1], requires_backward=True)
        self.assertFalse(runtime.weight_domain.parked)
        with torch.no_grad():
            runtime.retire_forward_scratch(keys[-1], requires_backward=False)
        self.assertFalse(runtime.weight_domain.parked)
        runtime.retire_forward_scratch(keys[-1], requires_backward=True)
        self.assertTrue(runtime.weight_domain.parked)
        self.distributed.all_reduce.assert_not_called()


    def test_completed_step_reuses_restored_storage_in_next_original_forward(self) -> None:
        """Restoration belongs to one F/B pair, and optimizer version advance is unchanged."""
        runtime, keys = self._runtime()
        runtime.retire_forward_scratch(keys[-1], requires_backward=True)
        runtime.backward_active = True
        runtime.backward_token = tuple(reversed(keys))
        runtime.weight_domain.materialize = Mock(return_value=SimpleNamespace(remote={}, ready=Mock()))
        runtime.prepare(keys[-1])
        runtime.tickets.clear()
        runtime.trace.clear()
        runtime.executions.clear()
        runtime.backward_active = False
        runtime.backward_token = None
        runtime.finish_step()
        self.assertEqual(runtime.version, 1)
        self.assertFalse(runtime.weight_domain.parked)
        self.distributed.reset_mock()
        expert = runtime.executors[0]
        key = runtime.new_key(expert)
        runtime.executions[key] = expert, SimpleNamespace(route=SimpleNamespace(active_experts=(0,)), weights=())
        runtime.prepare(key)
        self.distributed.all_reduce.assert_not_called()
