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
from hyper_parallel.distributed.expert_parallel.moonep.fused import communicators
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestOwnershipInitialization(unittest.TestCase):
    """Reject every rank before creating token or weight/gradient payload state."""

    def setUp(self) -> None:
        """Install metadata-only home tensors and fake communication resource factories."""
        self.enterContext(patch.object(communicators, "_bundles", {}))
        owners = [torch.nn.Linear(4, 4, bias=False) for _ in range(2)]
        self.experts = [SimpleNamespace(owner=owner, ranks=2, experts=4, top_k=2,
                                        replica_capacity=2, home_count=2, rank=0, group=object(),
                                        token_dedup=False, forward_ratio=0.75, backward_ratio=1.0,
                                        swiglu_limit=None) for owner in owners]
        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        self.enterContext(patch.object(execution, "vector_core_count", return_value=48))
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
                self.assertIn("share EP", self.gather.call_args.args[1][1])
                self._assert_no_resources()
                setattr(peer, name, original)


    def test_efs_metadata_admits_sharded_experts_without_restoring_weights(self) -> None:
        """Stable ownership admission does not require materializing later EFS weights."""
        expert = self.experts[1]
        manager = SimpleNamespace(shard_world_size=2, sharded_param=expert.owner.weight)
        expert.owner.hsdp_scheduler = SimpleNamespace(
            modules=[expert.owner], cell=expert.owner,
            hsdp_state=SimpleNamespace(hsdp_params=[manager], is_shard=True))
        expert.read_weights = Mock(side_effect=AssertionError("must not restore a later owner"))
        self.runtime.initialize(self.experts[0], self.home)
        self.gather.assert_called_once()
        expert.read_weights.assert_not_called()
        self.workspace.return_value.check_preflight.assert_called_once_with(None)
        self.assertEqual(self.groups.call_count, 5)
