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
"""CPU tests of the Xing4.0 optimizer: Muon schema, AdamW decay groups and QK clipping."""

import unittest

import pytest
import torch

from hyper_parallel.core.optimizer.muon import Muon as MuonOptimizer
from hyper_parallel.models.xing40.adapter.optim import _ns_transform, build_optimizer
from hyper_parallel.models.xing40.modeling_xing40 import Xing40Config
from tests.common.mark_utils import arg_mark

CONFIG = Xing40Config(  # pylint: disable=unexpected-keyword-arg
    vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=2,
    first_k_dense_replace=1, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=16, kv_lora_rank=8,
    qk_rope_head_dim=4, qk_nope_head_dim=6, v_head_dim=5, n_routed_experts=4, num_experts_per_tok=2,
    n_shared_experts=1, n_group=1, topk_group=1, moe_aux_loss_coeff=1e-4, moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=1e-3, use_pad_tokens=True, num_residual_streams=4, mhc_sinkhorn_iterations=20,
    mhc_init_gating_factor=0.01, mhc_layernorm_epsilon=1e-6, num_nextn_predict_layers=1, mtp_loss_factor=0.3,
    chunk_loss_size=8)



class TestMuonSchema(unittest.TestCase):
    """MindFormers matrix splits for Newton-Schulz."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_packed_mlp_preserves_separate_gate_up_updates(self):
        """
        Feature: Muon updates after the SwiGLUMLP replacement.
        Description: Pack unequal gate/up gradients into the Dense and shared FC1 layouts and
            run two native Muon steps alongside their separate projection parameters.
        Expectation: Packing preserves both logical matrix updates and their FP32 momentum.
        """
        transform = _ns_transform(CONFIG)
        for name, intermediate in (
                ("model.layers.0.mlp.linear_fc1.weight", CONFIG.intermediate_size),
                ("model.layers.1.mlp.shared_experts.linear_fc1.weight", CONFIG.moe_intermediate_size)):
            with self.subTest(name=name):
                values = torch.arange(intermediate * CONFIG.hidden_size, dtype=torch.float32)
                values = values.reshape(intermediate, CONFIG.hidden_size)
                gate_values = (values.remainder(17) - 8) / 32
                up_values = (values.remainder(11) - 5) / 16
                gate = torch.nn.Parameter(gate_values)
                up = torch.nn.Parameter(up_values)
                packed = torch.nn.Parameter(torch.cat((gate_values, up_values)))
                packed.model_name = name
                prefix = name.removesuffix(".linear_fc1.weight")
                gate.model_name = prefix + ".gate_proj.weight"
                up.model_name = prefix + ".up_proj.weight"
                options = {"lr": 1e-3, "weight_decay": 0.1, "momentum": 0.95, "ns_steps": 5,
                           "ns_variant": "legacy", "ns_transform_fn": transform}
                fused_optimizer = MuonOptimizer([packed], **options)
                separate_optimizer = MuonOptimizer([gate, up], **options)
                for step in range(2):
                    gate.grad = (values.remainder(13) - 6 + step) / 64
                    up.grad = (values.remainder(7) - 3 - step) / 8
                    packed.grad = torch.cat((gate.grad, up.grad))
                    fused_optimizer.step()
                    separate_optimizer.step()
                    torch.testing.assert_close(
                        packed, torch.cat((gate, up)), rtol=0, atol=0,
                        msg=f"{name}: packed gate/up parameter update differs at step {step + 1}")
                    expected_momentum = torch.cat((separate_optimizer.state[gate]["momentum_buffer"],
                                                   separate_optimizer.state[up]["momentum_buffer"]))
                    torch.testing.assert_close(
                        fused_optimizer.state[packed]["momentum_buffer"], expected_momentum, rtol=0, atol=0,
                        msg=f"{name}: packed momentum differs at step {step + 1}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_splits_restore_the_original_layout(self):
        """
        Feature: Muon NS input transform.
        Description: Split the fused latent projection into its query/KV/RoPE-key blocks, the MLA up
            projections per head and the fused expert gate/up matrix in halves, in the Transformers
            [E, 2I, H] layout and in the transposed GroupedExperts [E, H, 2I] layout.
        Expectation: Each part has the expected shape and restoring the unmodified parts reproduces the input.
        """
        transform = _ns_transform(CONFIG, frozenset({"model.layers.2.mlp.experts"}))
        heads, nope, rope, value = 4, CONFIG.qk_nope_head_dim, CONFIG.qk_rope_head_dim, CONFIG.v_head_dim
        q_lora, kv_lora = CONFIG.q_lora_rank, CONFIG.kv_lora_rank
        expert = CONFIG.moe_intermediate_size
        cases = {
            "model.layers.0.self_attn.linear_qkv.weight": (
                (q_lora + kv_lora + rope, 32), [(q_lora, 32), (kv_lora, 32), (rope, 32)]),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": (
                (kv_lora + rope, 32), [(kv_lora, 32), (rope, 32)]),
            "model.layers.0.self_attn.q_b_proj.weight": (
                (heads * (nope + rope), 16), [(heads * nope, 16), (heads * rope, 16)]),
            "model.layers.0.self_attn.kv_b_proj.weight": (
                (heads * (nope + value), 8), [(heads * nope, 8), (heads * value, 8)]),
            "model.layers.1.mlp.experts.gate_up_proj": (
                (4, 2 * expert, 32), [(4, expert, 32), (4, expert, 32)]),
            "model.layers.2.mlp.experts.gate_up_proj": (
                (4, 32, 2 * expert), [(4, 32, expert), (4, 32, expert)]),
        }
        for name, (shape, part_shapes) in cases.items():
            with self.subTest(name=name):
                update = torch.randn(shape)
                split = transform(name, update)
                self.assertEqual([tuple(part.shape) for part in split.tensors], part_shapes)
                expected_parts = ()
                if name.endswith(".kv_b_proj.weight"):
                    per_head = update.reshape(heads, nope + value, -1)
                    expected_parts = (per_head[:, :nope].reshape(heads * nope, -1),
                                      per_head[:, nope:].reshape(heads * value, -1))
                elif name == "model.layers.1.mlp.experts.gate_up_proj":
                    expected_parts = (update[:, :expert, :], update[:, expert:, :])
                elif name == "model.layers.2.mlp.experts.gate_up_proj":
                    expected_parts = (update[:, :, :expert], update[:, :, expert:])
                for part, expected_part in zip(split.tensors, expected_parts):
                    torch.testing.assert_close(part, expected_part, rtol=0, atol=0)
                restored = torch.empty_like(update)
                split.restore(split.tensors, restored)
                torch.testing.assert_close(restored, update, rtol=0, atol=0)
        query = torch.randn(heads * (nope + rope), 16)
        query_parts = transform("model.layers.0.self_attn.q_b_proj.weight", query).tensors
        per_head = query.view(heads, nope + rope, 16)
        torch.testing.assert_close(query_parts[0], per_head[:, :nope].reshape(-1, 16), rtol=0, atol=0)
        torch.testing.assert_close(query_parts[1], per_head[:, nope:].reshape(-1, 16), rtol=0, atol=0)
        self.assertIsNone(transform("model.layers.0.self_attn.o_proj.weight", torch.randn(8, 8)))


def test_builder_logical_split_is_opt_in_and_conflict_checked():
    """The actual Muon leaf receives the optional schema and other options."""
    model = torch.nn.Linear(2, 2)
    model.config = CONFIG
    for enabled in (False, True):
        builder = build_optimizer(model=model, muon_config={"ns_epsilon": 1e-7}, adamw_config={},
                                  logical_matrix_splits=enabled)
        leaf = next(opt for opt in builder.get_optimizer().chained_optimizers if isinstance(opt, MuonOptimizer))
        assert (leaf.ns_transform_fn is not None) == enabled, (
            f"expected transform_enabled={enabled}, got={leaf.ns_transform_fn is not None}"
        )
        if enabled:
            assert callable(leaf.ns_transform_fn), (
                f"expected callable=True, got={callable(leaf.ns_transform_fn)}"
            )
        assert all(group["ns_epsilon"] == 1e-7 for group in leaf.param_groups), (
            f"expected ns_epsilon=1e-7, got={[group['ns_epsilon'] for group in leaf.param_groups]}"
        )
    with pytest.raises(ValueError, match="conflicts"):
        build_optimizer(model=model, muon_config={"ns_transform_fn": lambda *_: None},
                        adamw_config={}, logical_matrix_splits=True)
