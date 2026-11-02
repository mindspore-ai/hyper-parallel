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
"""CPU-only ownership convergence and current-ticket retirement coverage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestOwnershipInitialization(unittest.TestCase):
    """Reject every rank before creating token or weight/gradient payload state."""

    def setUp(self) -> None:
        """Install metadata-only home tensors and fake communication resource factories."""
        owners = [torch.nn.Linear(4, 4, bias=False) for _ in range(2)]
        self.experts = [SimpleNamespace(owner=owner, ranks=2, experts=4, top_k=2,
                                        replica_capacity=2, home_count=2, rank=0, group=object(),
                                        token_dedup=False, forward_ratio=0.75, backward_ratio=1.0,
                                        swiglu_limit=None) for owner in owners]
        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        self.home = tuple(SimpleNamespace(dtype=torch.bfloat16, device=torch.device("npu"),
                                           shape=shape, stride=Mock(return_value=stride))
                          for shape, stride in (((2, 16, 32), (512, 32, 1)), ((2, 16, 16), (256, 16, 1))))
        for name, value in (("is_initialized", True), ("get_world_size", 2), ("get_rank", 0),
                            ("get_process_group_ranks", [0, 1])):
            self.enterContext(patch.object(execution.dist, name, return_value=value))
        self.gather = self.enterContext(patch.object(execution.dist, "all_gather_object", side_effect=self._gather))
        self.groups = self.enterContext(patch.object(execution.dist, "new_group", return_value=object()))
        self.workspace = self.enterContext(patch.object(execution, "RingWorkspace"))
        self.arena = self.enterContext(patch.object(execution, "PhaseArena"))
        self.domains = self.enterContext(patch.object(execution, "TransferDomain"))
        for name in ("Stream", "Event", "current_stream", "synchronize", "init"):
            self.enterContext(patch.object(torch.npu, name, side_effect=AssertionError("NPU operation invoked")))

    @staticmethod
    def _gather(output: list, value: Any, **_kwargs: Any) -> None:
        output[:] = [value, value]

    def _assert_no_resources(self) -> None:
        self.groups.assert_not_called()
        self.workspace.assert_not_called()
        self.arena.assert_not_called()
        self.domains.assert_not_called()
        self.assertTrue(self.runtime.poisoned)

    def test_local_overlap_is_gathered_before_rejection(self) -> None:
        """A local invalid owner participates in the shared manifest admission first."""
        self.experts[1].owner.weight = self.experts[0].owner.weight
        with self.assertRaisesRegex(ValueError, "ownership admission failed"):
            self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self.assertIn("tied parameter", self.gather.call_args.args[1][1])
        self._assert_no_resources()

    def test_remote_overlap_rejects_locally_valid_owners(self) -> None:
        """Remote ownership failure prevents every local payload allocation."""
        def _remote_error(output: list, value: Any, **_kwargs: Any) -> None:
            output[:] = [value, (value[0], "ValueError: overlapping parameter storage")]

        self.gather.side_effect = _remote_error
        with self.assertRaisesRegex(ValueError, "ownership admission failed"):
            self.runtime.initialize(self.experts[0], self.home)
        self.assertIsNone(self.gather.call_args.args[1][1])
        self._assert_no_resources()

    def test_manifest_mismatch_still_rejects_before_groups(self) -> None:
        """Adding ownership status does not weaken the existing model identity check."""
        def _mismatch(output: list, value: Any, **_kwargs: Any) -> None:
            output[:] = [value, ((99, *value[0][1:]), None)]

        self.gather.side_effect = _mismatch
        with self.assertRaisesRegex(ValueError, "manifests differ"):
            self.runtime.initialize(self.experts[0], self.home)
        self.groups.assert_not_called()
        self.domains.assert_not_called()

    def test_valid_ownership_adds_no_collective(self) -> None:
        """The same single manifest gather admits model and parameter ownership."""
        self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self.assertIsNone(self.gather.call_args.args[1][1])
        self.assertEqual(self.groups.call_count, 5)
        self.assertEqual(self.domains.call_count, 2)
        self.assertEqual(self.domains.return_value.admit.call_args.kwargs,
                         {"caller_accumulator": True, "caller_staging": True})
        self.workspace.return_value.check_preflight.assert_called_once_with(None)
        self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()

    def test_first_owner_unsharded_others_sharded(self) -> None:
        """Startup checks stable shards while matching each unit's current wrappers."""
        for index, expert in enumerate(self.experts):
            current = expert.owner.weight
            shard = current if index else torch.nn.Parameter(torch.ones(8))
            manager = SimpleNamespace(sharded_param=shard, unsharded_param=current, shard_world_size=1)
            expert.owner.hsdp_scheduler = SimpleNamespace(
                modules=[expert.owner], cell=expert.owner,
                hsdp_state=SimpleNamespace(hsdp_params=[manager], is_shard=bool(index)))
        self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self.workspace.return_value.check_preflight.assert_called_once_with(None)

    def test_missing_manager_coverage_is_converged(self) -> None:
        """An expert removed by FSDP deduplication reports through the shared admission."""
        expert = self.experts[0]
        expert.owner.hsdp_scheduler = SimpleNamespace(
            modules=[expert.owner], cell=expert.owner,
            hsdp_state=SimpleNamespace(hsdp_params=[], is_shard=True))
        with self.assertRaisesRegex(ValueError, "ownership admission failed"):
            self.runtime.initialize(expert, self.home)
        self.assertIn("cover exactly", self.gather.call_args.args[1][1])
        self._assert_no_resources()

    def test_release_retires_only_current_ticket(self) -> None:
        """Current release preserves the future prefetch without draining the W tail."""
        current, future = object(), object()
        first, second = object(), object()
        self.runtime.tickets = {current: first, future: second}
        domain = self.runtime.weight_domain = SimpleNamespace(drain=Mock(), release_lease=Mock())
        self.runtime.release(current)
        domain.release_lease.assert_called_once_with(first)
        domain.drain.assert_not_called()
        self.assertEqual(self.runtime.tickets, {future: second})
        self.runtime.release(current)
        domain.release_lease.assert_called_once_with(first)
        domain.drain.assert_not_called()

    def test_close_preserves_full_domain_drains(self) -> None:
        """Shutdown still joins both entire communication domains before freeing workspace."""
        weight = self.runtime.weight_domain = SimpleNamespace(drain=Mock())
        gradient = self.runtime.gradient_domain = SimpleNamespace(drain=Mock())
        workspace = self.runtime.workspace = SimpleNamespace(close=Mock())
        arena = self.runtime.arena = SimpleNamespace(close=Mock(), release_regions=Mock(), reserved_bytes=0)
        self.runtime.close()
        weight.drain.assert_called_once_with()
        gradient.drain.assert_called_once_with()
        arena.release_regions.assert_called_once()
        workspace.close.assert_called_once_with()
        arena.close.assert_called_once_with()

    def test_invalid_home_metadata_participates_in_manifest_admission(self) -> None:
        """Invalid tuple, dtype, device and attributes converge before resource creation."""
        valid = self.home
        cases = [(), (valid[0],), (None, valid[1]),
                 (SimpleNamespace(dtype=torch.float32), valid[1]),
                 (torch.empty((2, 16, 32), dtype=torch.bfloat16),
                  torch.empty((2, 16, 16), dtype=torch.bfloat16)),
                 (valid[0], SimpleNamespace(dtype=torch.bfloat16, device=torch.device("cpu")))]
        for home in cases:
            with self.subTest(home=repr(home)[:80]):
                self.runtime = execution.SparseMoonEPRuntime(self.experts)
                self.gather.reset_mock()
                with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
                    self.runtime.initialize(self.experts[0], home)
                self.gather.assert_called_once()
                self.assertIsNone(self.gather.call_args.args[1][0])
                self.assertIsInstance(self.gather.call_args.args[1][1], str)
                self._assert_no_resources()

    def test_rank0_bad_home_and_rank1_valid_home_reject_the_same_gather(self) -> None:
        """One bad rank and one valid rank both reject one unchanged object exchange."""
        expert = self.experts[0]
        valid_manifest = (expert.ranks, expert.experts, expert.top_k, expert.replica_capacity,
                          tuple(self.home[0].shape), tuple(self.home[1].shape),
                          self.home[0].stride(), self.home[1].stride(), len(self.experts),
                          False, 0.75, 1.0, None)
        batch = []

        def bad_rank(output: list, value: Any, **_kwargs: Any) -> None:
            """Capture the invalid rank alongside a valid peer manifest."""
            batch[:] = [value, (valid_manifest, None)]
            output[:] = batch

        self.gather.side_effect = bad_rank
        with self.assertRaisesRegex(ValueError, "home/ownership admission failed") as rank0:
            self.runtime.initialize(expert, ())
        self.gather.assert_called_once()
        self._assert_no_resources()

        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        self.gather.reset_mock()

        def valid_rank(output: list, value: Any, **_kwargs: Any) -> None:
            """Replay the same peer error from the valid rank's perspective."""
            self.assertEqual(value, (valid_manifest, None))
            output[:] = batch

        self.gather.side_effect = valid_rank
        with self.assertRaisesRegex(ValueError, "home/ownership admission failed") as rank1:
            self.runtime.initialize(expert, self.home)
        self.gather.assert_called_once()
        self.assertEqual(str(rank0.exception), str(rank1.exception))
        self._assert_no_resources()

    def test_manifest_property_failure_is_converged(self) -> None:
        """An exception while assembling shape/stride metadata still reaches the manifest gather."""
        self.home[0].stride.side_effect = MemoryError("manifest stride failure")
        with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
            self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self.assertIn("manifest stride failure", self.gather.call_args.args[1][1])
        self._assert_no_resources()

    def test_manifest_transport_failure_poisons_runtime(self) -> None:
        """A failed object collective is fatal without attempting payload initialization."""
        self.gather.side_effect = RuntimeError("manifest collective failed")
        with self.assertRaisesRegex(RuntimeError, "manifest collective failed"):
            self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self._assert_no_resources()

    def test_bad_home_geometry_converges_on_invalid_and_valid_rank(self) -> None:
        """Both ranks reject incompatible logical shapes before any new process group."""
        shapes = [((16, 32), (2, 16, 16)), ((2, 16, 32), (16, 16)),
                  ((3, 16, 32), (2, 16, 16)), ((2, 16, 32), (3, 16, 16)),
                  ((2, 0, 32), (2, 16, 16)), ((2, 16, 32), (2, 0, 16)),
                  ((2, 32, 32), (2, 16, 16)), ((2, 16, 31), (2, 16, 16))]
        expert = self.experts[0]
        valid_manifest = (expert.ranks, expert.experts, expert.top_k, expert.replica_capacity,
                          tuple(self.home[0].shape), tuple(self.home[1].shape),
                          self.home[0].stride(), self.home[1].stride(), len(self.experts),
                          False, 0.75, 1.0, None)
        for gate_shape, down_shape in shapes:
            with self.subTest(gate=gate_shape, down=down_shape):
                home = tuple(SimpleNamespace(dtype=torch.bfloat16, device=torch.device("npu"),
                                             shape=shape, stride=Mock(return_value=(1, 1, 1)))
                             for shape in (gate_shape, down_shape))
                self.runtime = execution.SparseMoonEPRuntime(self.experts)
                self.gather.reset_mock()
                batch = []

                def bad_rank(output: list, value: Any, **_kwargs: Any) -> None:
                    """Publish malformed local geometry with a valid remote manifest."""
                    batch[:] = [value, (valid_manifest, None)]
                    output[:] = batch

                self.gather.side_effect = bad_rank
                with self.assertRaisesRegex(ValueError, "home/ownership admission failed") as rank0:
                    self.runtime.initialize(expert, home)
                self.gather.assert_called_once()
                self.assertIn("[home,H,2I]", self.gather.call_args.args[1][1])
                self._assert_no_resources()

                self.runtime = execution.SparseMoonEPRuntime(self.experts)
                self.gather.reset_mock()

                def valid_rank(output: list, value: Any, **_kwargs: Any) -> None:
                    """Check that the valid peer observes the identical gathered error."""
                    self.assertEqual(value, (valid_manifest, None))
                    output[:] = batch

                self.gather.side_effect = valid_rank
                with self.assertRaisesRegex(ValueError, "home/ownership admission failed") as rank1:
                    self.runtime.initialize(expert, self.home)
                self.gather.assert_called_once()
                self.assertEqual(str(rank0.exception), str(rank1.exception))
                self._assert_no_resources()

    def test_valid_transposed_storage_keeps_logical_geometry_and_manifest(self) -> None:
        """Dense transposed physical weights retain the admitted [home,H,2I]/[home,I,H] contract."""
        def home(shape: tuple, strides: tuple) -> Any:
            """Describe an unsharded tensor without initializing an accelerator."""
            return SimpleNamespace(dtype=torch.bfloat16, device=torch.device("npu"), shape=shape,
                                   stride=lambda dimension=None: strides if dimension is None else strides[dimension])

        gate = home((2, 16, 32), (512, 1, 16))
        down = home((2, 16, 16), (256, 1, 16))
        self.runtime.initialize(self.experts[0], (gate, down))
        self.gather.assert_called_once()
        manifest, error = self.gather.call_args.args[1]
        self.assertIsNone(error)
        self.assertEqual(manifest[4:8], ((2, 16, 32), (2, 16, 16), (512, 1, 16), (256, 1, 16)))
        self.assertEqual(self.experts[0].transposed, (True, True))
        self.workspace.return_value.check_preflight.assert_called_once_with(None)

    def test_every_layer_partition_is_admitted_before_resources(self) -> None:
        """Later mismatches cannot reach a different count or replica collective."""
        peer = self.experts[1]
        changes = (("ranks", 1), ("rank", 1), ("experts", 8), ("home_count", 4),
                   ("top_k", 1), ("replica_capacity", 0), ("ranks", True))
        for name, value in changes:
            with self.subTest(field=name):
                original = getattr(peer, name)
                setattr(peer, name, value)
                self.runtime = execution.SparseMoonEPRuntime(self.experts)
                self.gather.reset_mock()
                with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
                    self.runtime.initialize(self.experts[0], self.home)
                self.gather.assert_called_once()
                self.assertIn("share WORLD", self.gather.call_args.args[1][1])
                self._assert_no_resources()
                setattr(peer, name, original)

    def test_later_group_membership_or_local_rank_is_rejected(self) -> None:
        """Equal ep_size declarations do not conceal different actual process groups."""
        peer = self.experts[1]
        for name, wrong in (("get_process_group_ranks", [0]), ("get_world_size", 1), ("get_rank", 1)):
            with self.subTest(field=name):
                self.runtime = execution.SparseMoonEPRuntime(self.experts)
                self.gather.reset_mock()
                original = getattr(execution.dist, name)

                def _metadata(group: Any = None, **kwargs: Any) -> Any:
                    return wrong if group is peer.group else original(group, **kwargs)

                with patch.object(execution.dist, name, side_effect=_metadata):
                    with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
                        self.runtime.initialize(self.experts[0], self.home)
                self.assertIs(self.gather.call_args.kwargs["group"], self.experts[0].group)
                self.gather.assert_called_once()
                self._assert_no_resources()

    def test_efs_metadata_rejects_sharded_experts_without_restoring_weights(self) -> None:
        """EFS reads the stable FSDP degree even while a later owner remains sharded."""
        expert = self.experts[1]
        manager = SimpleNamespace(shard_world_size=2, sharded_param=expert.owner.weight)
        expert.owner.hsdp_scheduler = SimpleNamespace(
            modules=[expert.owner], cell=expert.owner,
            hsdp_state=SimpleNamespace(hsdp_params=[manager], is_shard=True))
        expert.read_weights = Mock(side_effect=AssertionError("must not restore a later owner"))
        with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
            self.runtime.initialize(self.experts[0], self.home)
        self.assertIn("EFS shard_world_size=1", self.gather.call_args.args[1][1])
        self.gather.assert_called_once()
        expert.read_weights.assert_not_called()
        self._assert_no_resources()

    def test_subgroup_rejection_uses_actual_canonical_gather_size(self) -> None:
        """A real EP subgroup fails through its existing admission before any new group."""
        def size(group: Any = None) -> int:
            """Expose a two-rank subgroup inside the four-rank world."""
            return 4 if group is None else 2
        with patch.object(execution.dist, "get_world_size", side_effect=size):
            with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
                self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        self.assertEqual(len(self.gather.call_args.args[0]), 2)
        self.assertIn("spanning WORLD", self.gather.call_args.args[1][1])
        self._assert_no_resources()

    def test_wrong_first_layer_still_uses_canonical_admission_group(self) -> None:
        """The invoked layer cannot redirect startup into its own process group."""
        with self.assertRaisesRegex(ValueError, "home/ownership admission failed"):
            self.runtime.initialize(self.experts[1], self.home)
        self.assertIs(self.gather.call_args.kwargs["group"], self.experts[0].group)
        self.assertIn("first registered", self.gather.call_args.args[1][1])
        self._assert_no_resources()
