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
"""Comb-test regression guards: kimi naming coverage in parameter classification.

Two failure modes from the deepseek/kimi combination matrix (2026-10-24):

* kimi_k3 / kimi_linear keep the MoE router under
  ``*.block_sparse_moe.gate.*`` (Mixtral-style).  The generic MOE_GATE naming
  rules only knew the qwen/deepseek-style ``.mlp.gate.`` spelling, so the
  router weight and ``e_score_correction_bias`` fell through to SKIP and the
  plan-time coverage check failed with 395 uncovered trainable parameters.
* the kimi family sharding rules left ``q_a_proj`` (MLA down-projection),
  ``self_attention_res_proj`` / ``mlp_res_proj`` (per-layer residual
  projections) and the KDA decay params ``A_log`` / ``dt_bias`` unclassified.
  SKIP-on-trainable is a coverage error, and COLWISE-on-A_log was worse: the
  probe checkpoints keep A_log in a 4-D conv-style layout ``(1, 1, heads, 1)``
  whose heads sit on dim 2, so Shard(0) on dim0=1 failed the plan-time shard
  divisibility check under any TP > 1.
"""

import unittest

from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParameterClassifier,
    ParamRole,
)


class TestBlockSparseMoeGateClassification(unittest.TestCase):
    """Generic MOE_GATE rules must cover block_sparse_moe.gate naming."""

    def setUp(self):
        self.classifier = ParameterClassifier()

    def _classify(self, fqn: str) -> ParamRole:
        return self.classifier.classify_param(fqn)

    def test_gate_weight(self):
        self.assertEqual(
            self._classify("model.layers.1.block_sparse_moe.gate.weight"),
            ParamRole.MOE_GATE,
        )

    def test_gate_score_bias(self):
        self.assertEqual(
            self._classify(
                "language_model.model.layers.3.block_sparse_moe.gate.e_score_correction_bias"
            ),
            ParamRole.MOE_GATE,
        )

    def test_mlp_gate_still_matches(self):
        self.assertEqual(
            self._classify("model.layers.0.mlp.gate.weight"),
            ParamRole.MOE_GATE,
        )

    def test_gate_proj_not_moe_gate(self):
        # qwen-style dense gate/up projection must stay COLWISE
        self.assertEqual(
            self._classify("model.layers.0.mlp.gate_proj.weight"),
            ParamRole.COLWISE,
        )


class TestKimiFamilyShardingRules(unittest.TestCase):
    """kimi_k3 / kimi_linear family rules must classify every trainable param."""

    def setUp(self):
        from hyper_parallel.models.kimi_k3.adapter.policies.sharding import (  # pylint: disable=C0415
            build_parameter_sharding_rules,
        )

        self.rules = build_parameter_sharding_rules()
        self.classifier = ParameterClassifier(
            arch_overrides={"KimiLinearForCausalLM": self.rules}
        )

    def _classify(self, fqn: str) -> ParamRole:
        return self.classifier.classify_param(fqn, self.rules)

    def test_mla_q_a_proj_replicated(self):
        self.assertEqual(
            self._classify("language_model.model.layers.3.self_attn.q_a_proj.weight"),
            ParamRole.REPLICATED,
        )

    def test_res_proj_replicated(self):
        for leaf in ("self_attention_res_proj", "mlp_res_proj"):
            with self.subTest(leaf=leaf):
                self.assertEqual(
                    self._classify(f"model.layers.0.{leaf}.weight"),
                    ParamRole.REPLICATED,
                )

    def test_kda_decay_params_replicated(self):
        # 4-D conv-style layout: heads on dim 2, COLWISE Shard(0) would fail
        # the plan-time divisibility check under TP
        for leaf in ("A_log", "dt_bias"):
            with self.subTest(leaf=leaf):
                self.assertEqual(
                    self._classify(f"model.layers.0.self_attn.{leaf}"),
                    ParamRole.REPLICATED,
                )

    def test_kda_colwise_params_unchanged(self):
        for leaf in ("q_conv1d", "f_b_proj", "g_proj"):
            with self.subTest(leaf=leaf):
                self.assertEqual(
                    self._classify(f"model.layers.0.self_attn.{leaf}.weight"),
                    ParamRole.COLWISE,
                )


if __name__ == "__main__":
    unittest.main()
