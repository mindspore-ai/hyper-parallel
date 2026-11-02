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
"""CPU tests of the Xing4.0 optimizer: AdamW decay groups and QK clipping."""

import unittest

import torch
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

from hyper_parallel.core.optimizer.muon import Muon as MuonOptimizer
from hyper_parallel.models.xing40.adapter.optim import build_optimizer
from hyper_parallel.models.xing40.modeling_xing40 import (
    Xing40Config,
    Xing40ForCausalLM,
)
from tests.common.mark_utils import arg_mark
from tests.common.xing40_mhc_stubs import stub_mhc_for_cpu

CONFIG = Xing40Config(  # pylint: disable=unexpected-keyword-arg
    vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=2,
    first_k_dense_replace=1, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=16, kv_lora_rank=8,
    qk_rope_head_dim=4, qk_nope_head_dim=6, v_head_dim=5, n_routed_experts=4, num_experts_per_tok=2,
    n_shared_experts=1, n_group=1, topk_group=1, moe_aux_loss_coeff=1e-4, moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=1e-3, use_pad_tokens=True, num_residual_streams=4, mhc_sinkhorn_iterations=20,
    mhc_init_gating_factor=0.01, mhc_layernorm_epsilon=1e-6, num_nextn_predict_layers=1, mtp_loss_factor=0.3,
    chunk_loss_size=8)


class TestQkClip(unittest.TestCase):
    """MuonClip rescaling of the MLA up projections with mHC isolated."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_optimizer_step_runs_the_clip(self):
        """
        Feature: Optimizer integration of QK clipping.
        Description: Isolate mHC with shape-preserving stubs and compare training steps with and without
            clipping; every attention module recorded maxima where only head 1 exceeds the threshold.
        Expectation: The clipped step equals the unclipped one with head 1's query/key no-RoPE rows scaled
            by 1/2 and its query RoPE rows by 1/4, and it clears the maxima.
        """
        input_ids = torch.randint(0, CONFIG.vocab_size, (1, 8), generator=torch.Generator().manual_seed(1))
        stepped = {}
        for threshold in (0.0, 100.0):
            torch.manual_seed(0)
            model = Xing40ForCausalLM(CONFIG)
            optimizer = build_optimizer(
                model=model, muon_config={"muon_lr": 1e-3}, adamw_config={"adamw_lr": 1e-3},
                qk_clip_threshold=threshold).get_optimizer()
            with stub_mhc_for_cpu():
                torch.stack(list(model(input_ids, shift_labels=input_ids.roll(-1, 1)).loss.values())).sum().backward()
            attention = [module for module in model.modules() if isinstance(module, DeepseekV3Attention)]
            for module in attention:
                module.max_logits = torch.tensor([10.0, 400.0, 99.0, 100.0])
            optimizer.step()
            stepped[threshold] = attention
        nope, rope, value = CONFIG.qk_nope_head_dim, CONFIG.qk_rope_head_dim, CONFIG.v_head_dim
        query_scale = torch.ones(4, nope + rope)
        query_scale[1] = torch.tensor([0.5] * nope + [0.25] * rope)
        key_value_scale = torch.ones(4, nope + value)
        key_value_scale[1, :nope] = 0.5
        for clipped, unclipped in zip(stepped[100.0], stepped[0.0]):
            torch.testing.assert_close(clipped.q_b_proj.weight, unclipped.q_b_proj.weight * query_scale.reshape(-1, 1),
                                       msg="case: query rows of the clipped step")
            torch.testing.assert_close(clipped.kv_b_proj.weight,
                                       unclipped.kv_b_proj.weight * key_value_scale.reshape(-1, 1),
                                       msg="case: key/value rows of the clipped step")
            self.assertIsNone(clipped.max_logits, f"uncleared maxima: {clipped.max_logits}")


class TestAdamwDecayGroups(unittest.TestCase):
    """MindFormers weight-decay groups of the AdamW fallback."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_one_dimensional_parameters_are_not_decayed(self):
        """
        Feature: AdamW weight-decay groups.
        Description: Build the optimizer with an AdamW weight decay of 0.1.
        Expectation: Every 1-D parameter (such as norm weights) has no decay; the 2-D AdamW parameters
            (embedding, LM head) decay by 0.1.
        """
        model = Xing40ForCausalLM(CONFIG)
        optimizer = build_optimizer(model=model, muon_config={"muon_lr": 1e-3},
                                    adamw_config={"adamw_lr": 1e-3, "adamw_weight_decay": 0.1}).get_optimizer()
        decay = {parameter.model_name: group["weight_decay"] for leaf in optimizer.chained_optimizers
                 if not isinstance(leaf, MuonOptimizer) for group in leaf.param_groups for parameter in group["params"]}
        expected = {name: 0.0 if parameter.ndim == 1 else 0.1
                    for name, parameter in model.named_parameters() if name in decay}
        self.assertEqual(decay, expected)
        self.assertEqual(decay.get("model.norm.weight"), 0.0, f"norm decay={decay.get('model.norm.weight')}")
        self.assertEqual(decay.get("lm_head.weight"), 0.1, f"lm_head decay={decay.get('lm_head.weight')}")


if __name__ == "__main__":
    unittest.main()
