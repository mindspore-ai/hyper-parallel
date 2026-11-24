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
"""Regression tests for the ported workload cost models."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from hyper_parallel.distributed_data import (
    BackboneFlopsConfig, CausalAttentionCostModel, DefaultCostModel, SampleKey, SampleMetadata, WorkloadCost,
)
from hyper_parallel.distributed_data.cost_model import resolve_cost_model
from hyper_parallel.distributed_data.planner import DynamicPackingPlanner


def _backbone_config() -> BackboneFlopsConfig:
    """Return a small one-layer dense backbone with explicit dimensions."""
    return BackboneFlopsConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        mlp_layer_types=("dense",), kv_lora_rank=4, q_lora_rank=None,
        qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=4,
    )


class TestCostModel(unittest.TestCase):
    """Verify explicit model configuration, overrides, and backbone estimates."""

    def test_causal_square_cost_and_relative_gain_exclude_linear_terms(self) -> None:
        """Feature: Normalized causal Attention cost.
        Description: Plan unequal document groups using the square-cost model.
        Expectation: Rank costs and relative gain include PAD squares and exclude linear terms.
        """
        lengths = (6, 6, 2, 2, 2, 2)
        samples = tuple(SampleMetadata(length, key=SampleKey(0, index))
                        for index, length in enumerate(lengths))
        reference = ((samples[0].key, samples[1].key), tuple(item.key for item in samples[2:]))
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=12, cost_model=CausalAttentionCostModel(),
        )
        plan = planner.plan(samples, reference_bins=reference, step=0)
        self.assertEqual([cost.llm for cost in plan.rank_costs], [48, 48])
        self.assertEqual(planner.last_balance_decision["original_objective"], 72)
        self.assertEqual(planner.last_balance_decision["accepted_objective"], 48)
        self.assertAlmostEqual(planner.last_balance_decision["relative_gain"], (72 - 48) / 72)
        with self.assertRaisesRegex(ValueError, "causal text"):
            CausalAttentionCostModel()(SampleMetadata(4, features={"P": 2, "D": 2}))

    def test_gain_gate_rejects_document_gain_that_increases_padding_cost(self) -> None:
        """Feature: Padding-aware acceptance gate.
        Description: Propose a layout with lower source cost but higher padded cost.
        Expectation: The planner rejects the layout and reports negative relative gain.
        """
        samples = tuple(SampleMetadata(length, key=SampleKey(0, index))
                        for index, length in enumerate((1, 1, 1, 1, 2)))
        reference = (tuple(item.key for item in samples[2:]), tuple(item.key for item in samples[:2]))
        candidate = ((samples[4].key,), tuple(item.key for item in samples[:4]))
        policy = SimpleNamespace(
            assign=lambda *_, **__: candidate, objective=lambda costs: max(cost.llm for cost in costs),
            objective_name="makespan",
        )
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=1, seq_len=12,
            cost_model=CausalAttentionCostModel(), balancing_algorithm=policy,
        )
        plan = planner.plan(samples, reference_bins=reference, step=0)
        self.assertEqual(tuple(batch[0].sample_keys for batch in plan.local_batches), reference)
        self.assertEqual([cost.llm for cost in plan.rank_costs], [70, 102])
        self.assertEqual(planner.last_balance_decision["candidate_objective"], 104)
        self.assertFalse(planner.last_balance_decision["accepted"])
        self.assertAlmostEqual(planner.last_balance_decision["relative_gain"], -2 / 102)

    def test_padding_is_charged_per_pack_for_multiple_local_packs(self) -> None:
        """Feature: Rank-local multi-pack cost.
        Description: Keep two independently padded packs on each rank.
        Expectation: Each PAD segment contributes its own squared length.
        """
        samples = tuple(SampleMetadata(length, key=SampleKey(0, index))
                        for index, length in enumerate((2, 3, 4, 5)))
        reference = tuple((item.key,) for item in samples)
        policy = SimpleNamespace(
            assign=lambda *_, **__: reference, objective=lambda costs: max(cost.llm for cost in costs),
            objective_name="makespan",
        )
        planner = DynamicPackingPlanner(
            data_parallel_size=2, local_batch_size=2, seq_len=12,
            cost_model=CausalAttentionCostModel(), balancing_algorithm=policy,
        )
        plan = planner.plan(samples, reference_bins=reference, step=0)
        self.assertEqual([cost.llm for cost in plan.rank_costs], [194, 154])

    def test_default_requires_model_configuration(self) -> None:
        """Feature: Explicit default cost configuration.
        Description: Construct the FLOPs model without model dimensions.
        Expectation: Missing dimensions are rejected rather than using arbitrary costs.
        """
        with self.assertRaisesRegex(ValueError, "model_config"):
            DefaultCostModel(None)

    def test_custom_cost_model_does_not_require_model_configuration(self) -> None:
        """Feature: Custom workload model.
        Description: Resolve a metadata-based callback without backbone dimensions.
        Expectation: The callback retains the explicitly provided stage costs.
        """
        cost = WorkloadCost(io=1, encoder=2, llm=3)
        metadata = SampleMetadata(pack_tokens=4, cost=cost)

        model = resolve_cost_model(lambda sample: sample.cost, None)
        self.assertEqual(model(metadata), cost)

    def test_backbone_estimate_preserves_token_footprint(self) -> None:
        """Feature: Backbone workload estimation.
        Description: Evaluate projection, MLP and attention costs from sample dimensions.
        Expectation: Estimated cost is correct without changing the physical token footprint.
        """
        model = DefaultCostModel(_backbone_config())
        metadata = SampleMetadata(pack_tokens=4, features={"P": 2, "D": 2})

        self.assertEqual(model.forward_flops(metadata), 5184.0)
        self.assertEqual(model(metadata), WorkloadCost(llm=15552.0))
        self.assertEqual(metadata.pack_tokens, 4)

    def test_backbone_rejects_boolean_noninteger_and_nonpositive_dimensions(self) -> None:
        """Feature: Backbone configuration validation.
        Description: Supply boolean, noninteger and nonpositive hidden sizes.
        Expectation: Invalid dimensions raise a descriptive ValueError.
        """
        config = _backbone_config()
        for value in (True, False, 8.0, 0, -1):
            with self.subTest(hidden_size=value), self.assertRaisesRegex(ValueError, "hidden_size"):
                replace(config, hidden_size=value)
