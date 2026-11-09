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
"""Independent FLOPs accounting for complete hybrid GDN and gated-attention models."""

import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

from hyper_parallel.trainer.runtime.monitor.flops import FlopsBlock, model_blocks, transformer_blocks


class TestMonitorFlops(unittest.TestCase):
    """Verify independent arithmetic, real module coverage and input geometry."""

    def test_gdn_individual_blocks_match_hand_calculation(self) -> None:
        """Feature: GDN. Description: Unequal head counts. Expectation: Seven complete compute blocks."""
        config = SimpleNamespace(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=16,
            layer_types=["linear_attention"],
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_conv_kernel_dim=3,
            intermediate_size=12,
        )
        estimator = transformer_blocks(config)
        work = estimator.estimate(5, 25)
        actual = {
            name.removeprefix("layers.0.linear_attn."): value
            for name, value in work.items()
            if name.startswith("layers.0.linear_attn.")
        }
        self.assertFalse(estimator.missing)
        self.assertEqual(
            actual,
            {
                "in_proj_qkv": 3840,
                "in_proj_z": 1920,
                "in_proj_b": 480,
                "in_proj_a": 480,
                "conv1d": 1440,
                "delta_rule": 3840,
                "out_proj": 1920,
            },
        )
        # The recurrence uses value heads after Q/K repetition; convolution uses pre-repeat channels.
        self.assertEqual(sum(actual.values()), 13920)
        self.assertEqual(work, estimator.estimate(5, 13))

    @staticmethod
    def _reference_from_real_weights(model: torch.nn.Module, tokens: int, squared: int) -> float:
        config = model.config
        linear_parameters = 0
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2 or name.endswith("embed_tokens.weight"):
                continue
            count = parameter.numel()
            if ".experts." in name:
                count = count * config.num_experts_per_tok / config.num_experts
            linear_parameters += count
        gdn_layers = config.layer_types.count("linear_attention")
        full_layers = config.layer_types.count("full_attention")
        # Standard training MAC work is 6 per weight/token. Non-weight attention work is separate.
        return (
            6 * tokens * linear_parameters
            + 24 * tokens * gdn_layers * config.linear_num_value_heads * config.linear_value_head_dim**2
            + 6 * squared * full_layers * config.num_attention_heads * config.head_dim
        )

    def test_qwen36_full_and_twenty_layer_views_cover_actual_weight_shapes(self) -> None:
        """Feature: Full model. Description: Public 35B-A3B geometry on meta. Expectation: Exact standard sum."""
        for layers in (20, 40):
            with self.subTest(layers=layers):
                # The Qwen3.6 text checkpoint uses this HF architecture and these dimensions.
                config = Qwen3_5MoeTextConfig(num_hidden_layers=layers, attn_output_gate=True, mtp_num_hidden_layers=1)
                with torch.device("meta"):
                    model = Qwen3_5MoeForCausalLM(config)
                estimator = model_blocks(model, config)
                tokens, squared = 8 * 4096, 8 * 4096**2
                work = estimator.estimate(tokens, squared)
                self.assertFalse(estimator.missing)
                self.assertEqual(sum(work.values()), self._reference_from_real_weights(model, tokens, squared))
                self.assertEqual(sum(name.endswith("delta_rule") for name in work), layers * 3 // 4)
                self.assertEqual(sum(name.endswith("attention_gate") for name in work), layers // 4)
                self.assertEqual(sum(name.endswith("ffn.down") for name in work), layers)
                self.assertEqual(sum(name.endswith("shared_gate") for name in work), layers)
                self.assertEqual(len(work), len({block.name for block in estimator.blocks}))
                self.assertIn("lm_head", work)
                self.assertFalse(any("mtp" in name for name, _ in model.named_parameters()))

    def test_real_query_projection_controls_output_gate(self) -> None:
        """Feature: Attention gate. Description: HF config omits flag. Expectation: No missed projection."""
        config = Qwen3_5MoeTextConfig(
            hidden_size=16,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            head_dim=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            attn_output_gate=False,
        )
        with torch.device("meta"):
            model = Qwen3_5MoeForCausalLM(config)
        estimator = model_blocks(model, config)
        work = estimator.estimate(5, 25)
        self.assertEqual(work["layers.0.attention_gate"], 6 * 5 * 16 * 16)
        self.assertEqual(sum(work.values()), self._reference_from_real_weights(model, 5, 25))

    def test_packed_geometry_and_parallel_partition_preserve_total_work(self) -> None:
        """Feature: Geometry. Description: Unequal packed lengths. Expectation: Additive and replica-free."""
        config = Qwen3_5MoeTextConfig(num_hidden_layers=4, attn_output_gate=True)
        estimator = transformer_blocks(config)
        partitions = ([2, 7], [3, 5, 1])
        total = estimator.estimate(18, 88)
        by_rank = [estimator.estimate(sum(lengths), sum(length**2 for length in lengths)) for lengths in partitions]
        self.assertEqual(total, {name: sum(rank[name] for rank in by_rank) for name in total})
        packed_as_dense = estimator.estimate(18, 18**2)
        for name in total:
            if name.endswith(".attention"):
                self.assertGreater(packed_as_dense[name], total[name])
            else:
                self.assertEqual(packed_as_dense[name], total[name])

    def test_unknown_attention_and_invalid_gdn_are_not_partial_totals(self) -> None:
        """Feature: Coverage. Description: Unsupported or invalid metadata. Expectation: Explicit missing blocks."""
        for change in ({"linear_value_head_dim": 64}, {"linear_conv_kernel_dim": 0}, {"layer_types": ["unknown"]}):
            with self.subTest(change=change):
                config = Qwen3_5MoeTextConfig(num_hidden_layers=1, layer_types=["linear_attention"])
                for key, value in change.items():
                    setattr(config, key, value)
                estimator = transformer_blocks(config)
                self.assertTrue(estimator.missing)
                self.assertEqual(estimator.estimate(5, 25), {})
        with self.assertRaises(ValueError):
            FlopsBlock("conv", "depthwise_conv1d", input_size=4, kernel_size=0)
