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
"""Regression tests for codegen parallel-strategy combination boundaries."""

import unittest

from torch import nn

from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models.kimi_k3.adapter.policies.sharding import (
    build_parameter_sharding_rules as kimi_rules,
)


class TestModelShardingRules(unittest.TestCase):
    """Model-owned policies cover multimodal and Kimi-specific parameters."""

    def test_kimi_special_parameters_are_replicated(self):
        """Cover Kimi special parameters in its model adapter.

        Feature: model-owned sharding policy
        Description: Collect the replicated patterns declared by the Kimi adapter.
        Expectation: Delta-attention special parameters are present in that policy.
        """
        rules = kimi_rules()
        replicated = {
            pattern
            for patterns, role in rules
            if role == ParamRole.REPLICATED
            for pattern in ([patterns] if isinstance(patterns, str) else patterns)
        }
        self.assertTrue({
            "A_log", "dt_bias", "q_a_proj", "output_attn_res_proj",
        }.issubset(replicated))

    def test_kimi_sparse_gate_classifies_as_router(self):
        """Assign Kimi sparse gates their adapter-declared router role.

        Feature: codegen parameter classification
        Description: Classify a minimal block_sparse_moe gate with the shared native planner.
        Expectation: Its weight is classified as ``MOE_GATE``.
        """
        model = nn.Module()
        model.block_sparse_moe = nn.Module()
        model.block_sparse_moe.gate = nn.Linear(4, 4, bias=False)
        roles = ShardingPlanner()._classify_all_params(model, "kimi_k3")
        self.assertEqual(
            roles["block_sparse_moe.gate.weight"], ParamRole.MOE_GATE,
        )


if __name__ == "__main__":
    unittest.main()
