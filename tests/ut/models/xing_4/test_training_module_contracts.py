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
"""CPU contracts the Xing4 training model establishes while it is constructed."""

import unittest

import torch

from hyper_parallel.models.xing_4.modeling_xing4_0 import Xing4ForCausalLM, create_config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import Xing4_0DecoderLayer
from tests.common.mark_utils import arg_mark

RECIPE = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=12,
    num_hidden_layers=3, num_nextn_predict_layers=1, num_attention_heads=4, num_key_value_heads=4,
    n_shared_experts=1, n_routed_experts=8, kv_lora_rank=8, q_lora_rank=16,
    qk_rope_head_dim=4, v_head_dim=8, qk_nope_head_dim=8, n_group=2, topk_group=1,
    num_experts_per_tok=2, first_k_dense_replace=1, max_position_embeddings=64,
    rms_norm_eps=1e-5, hc_mult=2, hc_sinkhorn_iters=2,
    mhc_h_res_clamp_min=None, mhc_h_res_clamp_max=None,
    mhc_layernorm_epsilon=2e-5, mhc_init_gating_factor=0.25, mtp_loss_factor=0.3,
    moe_aux_loss_coeff=0.001, moe_router_bias_update_rate=0.001,
    moe_router_enable_expert_bias=True, use_pad_tokens=False,
    chunk_loss_size=8, compute_dtype="float32",
)


def _build_model(**overrides):
    """Build the training model from a small but structurally complete recipe."""
    torch.manual_seed(0)
    return Xing4ForCausalLM(create_config(dict(RECIPE, **overrides)))


class _CaptureExperts(torch.nn.Module):
    """Record the expert call without running any device kernel."""

    def __init__(self, captured: dict) -> None:
        """Keep the dictionary the test inspects after the forward."""
        super().__init__()
        self._captured = captured

    def forward(self, hidden, topk_indices, topk_weights):
        """Capture the dispatch arguments and return a neutral result."""
        self._captured.update(indices=topk_indices, weights=topk_weights)
        return torch.zeros_like(hidden)


class TestTrainingModuleContracts(unittest.TestCase):
    """Construction-time wiring the official modules do not express themselves."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_recipe_epsilon_reaches_the_fused_latent_norms(self) -> None:
        """
        Feature: Xing4 training model construction.
        Description: Compare the fused MLA latent-norm epsilon with the official module default.
        Expectation: Backbone and MTP attention use the recipe epsilon, not the official default.
        """
        model = _build_model()
        config = model.config
        official_default = Xing4_0DecoderLayer(config, 0).self_attn.q_a_layernorm.variance_epsilon
        self.assertNotEqual(official_default, config.rms_norm_eps)
        for attention in [layer.self_attn for layer in model.model.layers] + \
                         [depth.mtp_block.self_attn for depth in model.mtp.layers]:
            self.assertEqual(attention.q_a_layernorm.eps, config.rms_norm_eps)
            self.assertEqual(attention.kv_a_layernorm.eps, config.rms_norm_eps)
        for layer in model.model.layers:
            self.assertEqual(layer.attn_hc.input_norm.eps, config.mhc_layernorm_epsilon)
            self.assertEqual(layer.ffn_hc.input_norm.eps, config.mhc_layernorm_epsilon)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_cold_start_initializes_the_assembled_modules(self) -> None:
        """
        Feature: Xing4 training model construction.
        Description: Inspect modules the official ``_init_weights`` no longer sees after assembly.
        Expectation: mHC gating, the shared RMSNorm scale and the packed experts are initialized.
        """
        model = _build_model()
        config = model.config
        layer = model.model.layers[config.first_k_dense_replace]
        hyper_connection = layer.attn_hc
        self.assertTrue(torch.allclose(
            hyper_connection.hc_scale,
            torch.full_like(hyper_connection.hc_scale, config.mhc_init_gating_factor)))
        self.assertEqual(int(torch.count_nonzero(hyper_connection.hc_base)), 0)
        self.assertTrue(torch.allclose(layer.self_attn.q_a_layernorm.weight,
                                       torch.ones_like(layer.self_attn.q_a_layernorm.weight)))
        packed = layer.mlp.experts.gate_up_proj
        self.assertTrue(torch.isfinite(packed).all())
        self.assertAlmostEqual(packed.std().item(), config.initializer_range,
                               delta=config.initializer_range * 0.4)
        self.assertEqual(int(torch.count_nonzero(layer.mlp.gate.tokens_per_expert)), 0)
        self.assertIs(layer.mlp.gate.e_score_correction_bias.dtype, torch.float32)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_moe_dispatches_indices_then_weights(self) -> None:
        """
        Feature: Xing4 MoE routing.
        Description: Capture the arguments the MoE hands to the grouped experts.
        Expectation: Experts receive integer expert ids first and the matching float weights second.
        """
        model = _build_model()
        config = model.config
        moe = model.model.layers[config.first_k_dense_replace].mlp
        captured = {}
        moe.experts = _CaptureExperts(captured)
        tokens = torch.randn(2, 5, config.hidden_size)
        output = moe(tokens)

        self.assertEqual(output.shape, tokens.shape)
        self.assertIn(captured["indices"].dtype, (torch.int32, torch.int64))
        self.assertTrue(captured["weights"].is_floating_point())
        self.assertEqual(tuple(captured["indices"].shape), (2 * 5, config.num_experts_per_tok))
        self.assertEqual(captured["weights"].shape, captured["indices"].shape)
        self.assertLess(int(captured["indices"].max()), config.n_routed_experts)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_safe_token_padding_carries_no_routing_weight(self) -> None:
        """
        Feature: Xing4 MoE routing.
        Description: Route with safe-token padding enabled and inspect the dispatched rows.
        Expectation: Padding rows exist for every expert, carry zero weight and leave the output length unchanged.
        """
        model = _build_model(use_pad_tokens=True)
        config = model.config
        moe = model.model.layers[config.first_k_dense_replace].mlp
        captured = {}
        moe.experts = _CaptureExperts(captured)
        tokens = torch.randn(2, 5, config.hidden_size)
        output = moe(tokens)

        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(tuple(captured["indices"].shape),
                         (2 * (5 + config.n_routed_experts), config.num_experts_per_tok))
        padded_weights = captured["weights"].view(2, -1, config.num_experts_per_tok)[:, :config.n_routed_experts]
        self.assertEqual(int(torch.count_nonzero(padded_weights)), 0)
        padded_indices = captured["indices"].view(2, -1, config.num_experts_per_tok)[:, :config.n_routed_experts]
        self.assertEqual(sorted(set(padded_indices.reshape(-1).tolist())), list(range(config.n_routed_experts)))


if __name__ == "__main__":
    unittest.main()
