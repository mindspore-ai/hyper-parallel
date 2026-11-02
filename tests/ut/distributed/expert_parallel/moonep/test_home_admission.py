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
"""CPU-only restored weight geometry admission before count and payload work."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fused import executor as execution
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestRestoredHomeAdmission(unittest.TestCase):
    """Use CPU weights and mock only initialization plus count exchange."""

    def setUp(self) -> None:
        """Bind the already admitted geometry without constructing native state."""
        group = object()
        self.home = (torch.ones((2, 16, 32), dtype=torch.bfloat16),
                     torch.ones((2, 16, 16), dtype=torch.bfloat16))
        with patch.object(execution.dist, "get_rank", return_value=0):
            self.experts = [execution.SparseMoonEPExecutor(torch.nn.Module(), Mock(return_value=self.home),
                            ep_group=group, ep_size=2, num_experts=4, top_k=2) for _ in range(2)]
        self.runtime = execution.SparseMoonEPRuntime(self.experts)
        self.runtime.initialize = Mock()
        self.runtime.workspace = SimpleNamespace(metadata={})
        self.runtime.home_geometry = (*(tuple(value.shape) for value in self.home),
                                      *(value.stride() for value in self.home))
        self.runtime.home_device = torch.device("cpu")
        for expert in self.experts:
            expert.shapes = ((16, 32), (16, 16))
        self.ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
        self.scores = torch.ones((2, 2), dtype=torch.bfloat16)
        self.hidden = torch.ones((2, 16), dtype=torch.bfloat16)
        self.gather = self.enterContext(patch.object(execution.dist, "all_gather_into_tensor",
                                                    side_effect=self._gather))
        self.enterContext(patch.object(execution, "upload_metadata",
                                       side_effect=lambda value, device: value.to(device)))
        self.apply = self.enterContext(patch.object(execution.SparseExpertFunction, "apply"))

    @staticmethod
    def _gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> None:
        output.view(2, -1).copy_(value.expand(2, -1))

    def test_later_full_geometry_fails_through_existing_count_header(self) -> None:
        """Shape, physical stride, dtype and malformed pairs fail before any W/T payload."""
        released = self.home[0].clone()
        released.untyped_storage().resize_(0)
        variants = [(self.home[0][:1], self.home[1]),
                    (self.home[0], self.home[1].transpose(-1, -2)),
                    (self.home[0].float(), self.home[1]),
                    (self.home[0].reshape(-1), self.home[1]), (released, self.home[1]), ()]
        for home in variants:
            with self.subTest(shapes=tuple(value.shape for value in home)):
                self.runtime.trace.clear()
                self.runtime.sequence = 0
                self.runtime.poisoned = False
                self.runtime.new_key(self.experts[0])
                self.gather.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
                    self.experts[1](self.hidden, self.ids, self.scores, expert_weights=home)
                self.gather.assert_called_once()
                self.assertEqual(self.gather.call_args.args[1][-1].item(), 1)
                self.assertTrue(self.runtime.poisoned)
                self.apply.assert_not_called()

    def test_valid_peer_rejects_remote_geometry_error_before_apply(self) -> None:
        """One bad restored view propagates through the unchanged counts header."""
        self.runtime.new_key(self.experts[0])

        def _remote_invalid(output: torch.Tensor, value: torch.Tensor, **kwargs: object) -> None:
            self._gather(output, value, **kwargs)
            output.view(2, -1)[1, -1] = 1

        self.gather.side_effect = _remote_invalid
        with self.assertRaisesRegex(RuntimeError, "rank-consistent order"):
            self.experts[1](self.hidden, self.ids, self.scores, expert_weights=self.home)
        self.assertEqual(self.gather.call_args.args[1][-1].item(), 0)
        self.gather.assert_called_once()
        self.apply.assert_not_called()
        self.assertTrue(self.runtime.poisoned)

    def test_fresh_storage_with_identical_metadata_is_accepted(self) -> None:
        """Valid FSDP storage rebinding does not require a stale pointer match."""
        fresh = tuple(value.clone() for value in self.home)
        self.assertNotEqual(fresh[0].data_ptr(), self.home[0].data_ptr())
        self.runtime.validate_home(fresh)

    def test_replay_geometry_failure_is_fatal_without_new_count_collective(self) -> None:
        """Replay already owns its plan and must poison rather than re-admit or fallback."""
        key = self.runtime.new_key(self.experts[0])
        self.runtime.executions[key] = self.experts[0], object()
        with patch.object(execution, "bind_execution", return_value=key):
            with self.assertRaisesRegex(RuntimeError, "whole-job termination") as raised:
                invalid_home = tuple(value.float() for value in self.home)
                self.experts[0](self.hidden, self.ids, self.scores, expert_weights=invalid_home)
        self.gather.assert_not_called()
        self.apply.assert_not_called()
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertTrue(self.runtime.poisoned)
