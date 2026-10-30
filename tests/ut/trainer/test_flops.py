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
"""Unit tests for analytical model-FLOPs estimation."""

from types import SimpleNamespace
import unittest

from hyper_parallel.trainer.runtime.flops import estimate_flops_per_token
from tests.common.mark_utils import arg_mark


def qwen3_moe_config(**overrides: object) -> SimpleNamespace:
    """Build a miniature Qwen3-MoE configuration with hand-checkable sizes.

    Defaults describe two all-MoE layers with ``hidden_size=8``, GQA with
    ``num_attention_heads=2``/``num_key_value_heads=1``/``head_dim=4``, a gated
    ``intermediate_size=16`` dense MLP, four experts with top-1 routing and a
    ``vocab_size=32`` head. Every expectation in this module is derived from
    these numbers by hand.
    """
    values = {
        "model_type": "qwen3_moe",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "moe_intermediate_size": 4,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "vocab_size": 32,
        "hidden_act": "silu",
        "mlp_only_layers": [],
        "decoder_sparse_step": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestEstimateFlopsPerToken(unittest.TestCase):
    """Verify the estimator counts activations, not parameters or experts."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_qwen3_moe_matches_closed_form(self) -> None:
        """Add attention, top-1 MoE and logits terms into one per-token total.

        Feature: Analytical FLOPs estimation.
        Description: Estimate one token of a two-layer Qwen3-MoE at seq_length=3.
        Expectation: 2*1296 attention + 2*576 MoE + 1536 logits = 5280 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(), 3.0)
        self.assertEqual(
            flops,
            5280.0,
            f"expected 2*1296 attention + 2*576 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_causal_attention_scales_with_sequence_length(self) -> None:
        """Charge only the causal-attention term for longer sequences.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same model at seq_length=8 instead of 3.
        Expectation: The total grows by 2 layers * 6 * q * (8 - 3) = 480 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(), 8.0)
        self.assertEqual(
            flops,
            5760.0,
            f"expected the 5280 baseline plus 480 attention FLOPs, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_topk_routing_sizes_the_moe_term(self) -> None:
        """Scale the MoE term by activated experts instead of total experts.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same model with top-2 routing and four experts.
        Expectation: The MoE term doubles to 2*1152, so the total is 6432 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(num_experts_per_tok=2), 3.0)
        self.assertEqual(
            flops,
            6432.0,
            f"expected 2592 attention + 2304 top-2 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_gated_activation_uses_three_gemms(self) -> None:
        """Charge a gated FFN three GEMMs and a plain FFN two.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same model with a non-gated ``gelu`` activation.
        Expectation: The MoE term drops to 2*384, so the total is 4896 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(hidden_act="gelu"), 3.0)
        self.assertEqual(
            flops,
            4896.0,
            f"expected 2592 attention + 768 non-gated moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_mlp_only_layers_use_the_dense_term(self) -> None:
        """Route ``mlp_only_layers`` entries through the dense MLP term.

        Feature: Analytical FLOPs estimation.
        Description: Estimate a two-layer model whose first layer is listed in ``mlp_only_layers``.
        Expectation: One dense layer (2304) and one MoE layer (576) sum to 7008 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(mlp_only_layers=[0]), 3.0)
        self.assertEqual(
            flops,
            7008.0,
            f"expected 2592 attention + 2304 dense + 576 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_decoder_sparse_step_selects_every_nth_layer(self) -> None:
        """Apply ``decoder_sparse_step`` to decide which layers are MoE.

        Feature: Analytical FLOPs estimation.
        Description: Estimate four layers with ``decoder_sparse_step=2``.
        Expectation: Two dense and two MoE layers sum to 12480 FLOPs.
        """
        flops = estimate_flops_per_token(
            qwen3_moe_config(num_hidden_layers=4, decoder_sparse_step=2), 3.0
        )
        self.assertEqual(
            flops,
            12480.0,
            f"expected 4*1296 attention + 2*2304 dense + 2*576 moe + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_zero_experts_falls_back_to_the_dense_term(self) -> None:
        """Treat a zero-expert configuration as a dense decoder.

        Feature: Analytical FLOPs estimation.
        Description: Estimate the same two layers with ``num_experts=0``.
        Expectation: Both layers are dense, so the total is 8736 FLOPs.
        """
        flops = estimate_flops_per_token(qwen3_moe_config(num_experts=0), 3.0)
        self.assertEqual(
            flops,
            8736.0,
            f"expected 2592 attention + 2*2304 dense + 1536 logits, got={flops}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_unsupported_model_type_returns_none(self) -> None:
        """Return ``None`` instead of guessing for an unsupported model family.

        Feature: Analytical FLOPs estimation.
        Description: Estimate a configuration that only declares ``model_type='llama'``.
        Expectation: No estimate is produced and no missing field is read.
        """
        flops = estimate_flops_per_token(SimpleNamespace(model_type="llama"), 3.0)
        self.assertIsNone(
            flops,
            f"unsupported model families must not report FLOPs, got={flops!r}",
        )


if __name__ == "__main__":
    unittest.main()
