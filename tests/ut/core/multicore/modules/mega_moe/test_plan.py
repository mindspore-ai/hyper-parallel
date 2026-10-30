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
"""CPU tests for external-route MegaMoe scheduler bounds."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_moe.module import _MegaMoeExecutionResources
from hyper_parallel.core.multicore.modules.mega_moe.plan import _build_task_values
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec


class TestExternalRouteSchedule(unittest.TestCase):
    """Keep native task limits equal after top-k rows are externally expanded."""

    def test_expanded_route_matches_native_topk_schedule(self) -> None:
        """Eight sorted rows per token retain the ordinary top-k task bound."""
        native = MegaMoeSpec(
            local_num_tokens=8192, hidden_size=2048, intermediate_size=768,
            num_experts=128, top_k=8, initial_capacity_factor=1.25,
            receive_capacity=65536, ep_size=16, ep_group=None, rank_id=0, num_cube_cores=24,
        )
        external = replace(native, local_num_tokens=65536, top_k=1,
                           receive_capacity=65536, external_route_top_k=8)

        native_values = _build_task_values(native)
        external_values = _build_task_values(external)
        self.assertEqual(native_values.seq_size, external_values.seq_size)
        self.assertEqual(native_values.top_k, external_values.top_k)
        self.assertEqual(native_values.per_rank_seq, external_values.per_rank_seq)
        self.assertEqual(native_values.per_expert_seq_to_other, external_values.per_expert_seq_to_other)

    def test_incomplete_external_topk_group_is_rejected(self) -> None:
        """A schedule with a partial original token cannot silently drop rows."""
        spec = MegaMoeSpec(
            local_num_tokens=65535, hidden_size=2048, intermediate_size=768,
            num_experts=128, top_k=1, initial_capacity_factor=1.25,
            receive_capacity=65536, ep_size=16, ep_group=None, rank_id=0,
            num_cube_cores=24, external_route_top_k=8,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            _build_task_values(spec)

    def test_non_power_of_two_topk_uses_a_complete_schedule_bucket(self) -> None:
        """A geometric workspace bucket never truncates an external Top-K row."""
        spec = MegaMoeSpec(
            local_num_tokens=384, hidden_size=256, intermediate_size=128,
            num_experts=12, top_k=1, initial_capacity_factor=1.25,
            receive_capacity=384, ep_size=2, ep_group=None, rank_id=0,
            num_cube_cores=24, external_route_top_k=3,
        )
        resources = object.__new__(_MegaMoeExecutionResources)
        resources.spec = spec
        resources._token_capacity = 511
        resources._plans = {}
        with patch("hyper_parallel.core.multicore.modules.mega_moe.module.build_mega_moe_plan",
                   side_effect=lambda planned, _device: SimpleNamespace(spec=planned)):
            plan = resources.plan_for_tokens(384, torch.device("cpu"))
        self.assertEqual(plan.spec.plan_tokens, 510)
        self.assertEqual(_build_task_values(plan.spec).per_rank_seq, 510)


if __name__ == "__main__":
    unittest.main()
