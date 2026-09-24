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
"""Complete standalone model construction and optional acceleration boundaries."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from transformers import DeepseekV32Config

from hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3 import (
    JTDeepseekV3ForCausalLM, JTDeepseekV3Decoder, JTDeepseekV3MoE,
    JTDeepseekV3Attention, JTDeepseekV3MLAAttention,
)
from hyper_parallel.components.modules.mtp import DeepseekV3MTPExecution, MultiTokenPredictionLayer
from hyper_parallel.models.jt_deepseek_v3.adapter.conversion.jt_mtp import JTDeepseekV3MTPExecution
from hyper_parallel.models.replacement import compile_module_replacements, apply_module_replacements
from hyper_parallel.models.jt_deepseek_v3.adapter.jt_builder import _load_reference_state
from hyper_parallel.models.registry import get_model_adapter
from hyper_parallel.trainer.config import entries_to_module_replacements
from hyper_parallel.trainer.config.manager import parse_training_args
from tests.common.mark_utils import arg_mark


def small_config() -> DeepseekV32Config:
    """Build a CPU-sized fixture without changing the production validation recipe."""
    config = DeepseekV32Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
        mlp_layer_types=["dense", "sparse"], max_position_embeddings=32, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        architectures=["JTDeepseekV3ForCausalLM"],
        num_nextn_predict_layers=1, use_pad_tokens=True, norm_topk_prob=True,
        routed_scaling_factor=1.0, moe_aux_loss_coeff=0.01, mtp_loss_factor=0.3)
    config.rope_interleave = True
    return config


class TestCompleteModel(unittest.TestCase):
    """Exercise model semantics without a builder, EP adapter or replacement pass."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_standalone_forward_backward(self):
        """Feature: Complete model.

        Description: Construct directly while replacement compilation is forbidden.
        Expectation: Trunk, MTP and router execute and backpropagate without an EP adapter.
        """
        torch.manual_seed(11)
        with patch("hyper_parallel.models.replacement.compile_module_replacements", side_effect=AssertionError):
            model = JTDeepseekV3ForCausalLM(small_config())
        self.assertIsInstance(model.model.layers[0], JTDeepseekV3Decoder)
        self.assertIsInstance(model.model.layers[1].mlp, JTDeepseekV3MoE)
        self.assertEqual(model.model.layers[1].mlp.ep_compute, model.model.layers[1].mlp.local_routed_forward)
        self.assertIsInstance(model.mtp.layers[0], MultiTokenPredictionLayer)
        self.assertIs(type(model.mtp.execution), DeepseekV3MTPExecution)
        tokens = torch.arange(8).unsqueeze(0)
        output = model(tokens, (tokens + 1) % 32, torch.ones(1, 8))
        self.assertTrue(torch.isfinite(output.loss))
        metrics = model.get_logging_metrics()
        self.assertEqual(set(metrics), {"training/lm_loss", "training/mtp_loss", "training/aux_loss",
                                       "training/load_balancing_loss"})
        combined = (metrics["training/lm_loss"] + metrics["training/aux_loss"]) + metrics["training/mtp_loss"]
        self.assertTrue(torch.equal(output.loss.detach(), combined))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        self.assertEqual(model.get_logging_metrics(), {})
        output.loss.backward()
        for name in ["model.embed_tokens.weight", "model.layers.1.mlp.gate.weight", "mtp.layers.0.eh_proj.weight"]:
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_replacement_selects_attention_and_mtp_execution(self):
        """Feature: Acceleration boundary.

        Description: Apply the optional high-performance replacement to a complete model.
        Expectation: Attention and MTP execution change; decoder, MLP, norm and MTP weights survive.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        previous = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3Attention) for m in previous.values()), 3)
        original_q = model.model.layers[0].self_attn.q_a_proj.weight.detach().clone()
        original_kv = model.model.layers[0].self_attn.kv_a_proj_with_mqa.weight.detach().clone()
        recipe_path = Path(__file__).resolve().parents[4] / (
            "hyper_parallel/models/jt_deepseek_v3/recipes/jt_deepseek_v3.yaml")
        recipe = parse_training_args([str(recipe_path)])
        rules = entries_to_module_replacements(recipe.plan_overrides)
        self.assertEqual(len(rules), 2)
        plan = compile_module_replacements(model, rules)
        apply_module_replacements(model, plan, weights_mapping=[])
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.linear_qkv.weight,
                                    torch.cat((original_q, original_kv))))
        self.assertIsInstance(model.mtp.execution, JTDeepseekV3MTPExecution)
        self.assertEqual(dict(model.mtp.execution.named_parameters()), {})
        current = dict(model.named_modules())
        self.assertEqual(sum(isinstance(m, JTDeepseekV3MLAAttention) for m in current.values()), 3)
        for name in ["model.layers.0", "model.layers.1.mlp", "model.norm", "mtp.layers.0"]:
            self.assertIs(current[name], previous[name])
        self.assertIs(model.model.layers[0].self_attn.q_a_layernorm,
                      previous["model.layers.0.self_attn.q_a_layernorm"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_specialized_mtp_rounds_only_its_fusion(self):
        """Feature: Specialized MTP.

        Description: Use rounding-sensitive states and independent identity components.
        Expectation: BF16 hidden-then-embedding fusion and both input gradients are preserved.
        """
        layer = MultiTokenPredictionLayer(embedding_norm=nn.Identity(), hidden_norm=nn.Identity(),
                                      projection=nn.Identity(), decoder=nn.Identity(), output_norm=nn.Identity())
        hidden = torch.tensor([[[1.001, 2.003]]], requires_grad=True)
        embedding = torch.tensor([[[3.005, 4.007]]], requires_grad=True)
        execution = JTDeepseekV3MTPExecution(module=DeepseekV3MTPExecution())
        result = execution.fuse_inputs(layer, hidden, embedding)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(result, torch.cat((hidden.bfloat16(), embedding.bfloat16()), dim=-1)))
        result.float().sum().backward()
        self.assertTrue(torch.equal(hidden.grad, torch.ones_like(hidden)))
        self.assertTrue(torch.equal(embedding.grad, torch.ones_like(embedding)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_meta_replacement_loads_complete_reference_state(self):
        """Feature: Checkpoint conversion.

        Description: Materialize the recipe-selected model and load every original tensor.
        Expectation: Fused projections round-trip and every loaded parameter is exact.
        """
        config = small_config()
        source = JTDeepseekV3ForCausalLM(config)
        arrays = {name: value.detach().numpy().copy() for name, value in source.state_dict().items()}
        recipe_path = Path(__file__).resolve().parents[4] / (
            "hyper_parallel/models/jt_deepseek_v3/recipes/jt_deepseek_v3.yaml")
        rules = entries_to_module_replacements(parse_training_args([str(recipe_path)]).plan_overrides)
        with torch.device("meta"):
            candidate = JTDeepseekV3ForCausalLM(config)
            plan = compile_module_replacements(candidate, rules)
            apply_module_replacements(candidate, plan, weights_mapping=[])
        candidate.to_empty(device="cpu")
        loaded, groups = _load_reference_state(candidate, arrays)
        self.assertEqual(len(groups), 3)
        self.assertEqual(set(loaded), set(arrays))
        for group in groups:
            fused = candidate.state_dict()[group["storage"]].detach().numpy()
            original = [source.state_dict()[name].detach().numpy() for name in group["logical_parameters"]]
            np.testing.assert_array_equal(fused, np.concatenate(original, axis=0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_family_registration_does_not_replace_standard_deepseek(self):
        """Feature: Independent family registration.

        Description: Resolve both families and the custom architecture identity.
        Expectation: JT owns its spec; the original V3/V2 identities stay registered independently.
        """
        standard = get_model_adapter("deepseek_v3")
        custom = get_model_adapter("jt_deepseek_v3")
        self.assertIs(get_model_adapter("JTDeepseekV3ForCausalLM"), custom)
        self.assertIsNot(standard, custom)
        self.assertEqual(standard.architecture, "DeepseekV3ForCausalLM")
        self.assertEqual(custom.model_type, "jt_deepseek_v3")
        self.assertEqual(small_config().model_type, "deepseek_v32")
        self.assertIs(get_model_adapter(small_config().architectures[0]), custom)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_recipe_config_roundtrip(self):
        """Feature: Native model configuration.

        Description: Build and serialize the recipe's HF configuration without a reference YAML.
        Expectation: Model dimensions, JT options and independent adapter identity survive.
        """
        recipe_path = Path(__file__).resolve().parents[4] / (
            "hyper_parallel/models/jt_deepseek_v3/recipes/jt_deepseek_v3.yaml")
        recipe = parse_training_args([str(recipe_path)])
        config = DeepseekV32Config(**recipe.model.config)
        self.assertIs(type(config), DeepseekV32Config)
        self.assertFalse(hasattr(recipe.model, "reference_yaml"))
        self.assertFalse(hasattr(config, "jt_config"))
        restored = DeepseekV32Config.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(restored.mlp_layer_types, ["dense", "sparse"])
        self.assertEqual(restored.num_nextn_predict_layers, 1)
        self.assertEqual(restored.mtp_loss_factor, 0.3)
        restored.n_group = 2
        with self.assertRaisesRegex(ValueError, "n_group=topk_group=1"):
            JTDeepseekV3ForCausalLM(restored)
        self.assertEqual(restored.moe_aux_loss_coeff, 0.0001)
        self.assertEqual(restored.rope_parameters["rope_theta"], 5000000)
        self.assertIs(get_model_adapter(restored.architectures[0]), get_model_adapter("jt_deepseek_v3"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_aux_monitor_uses_trunk_moe_count_and_handles_zero_scale(self):
        """Feature: JT monitoring semantics.

        Description: Use two trunk routers and a separate MTP router, then disable auxiliary loss.
        Expectation: The monitor averages by two trunk layers; zero scale produces a finite zero.
        """
        config = small_config()
        config.mlp_layer_types = ["sparse", "sparse"]
        model = JTDeepseekV3ForCausalLM(config)
        model._step_loss_metrics = torch.tensor([2.0, 0.3, 0.06])
        model._metric_micro_batches = 1
        metrics = model.get_logging_metrics()
        torch.testing.assert_close(metrics["training/load_balancing_loss"], torch.tensor(3.0))
        self.assertEqual(model.get_logging_metrics(), {})
        config.moe_aux_loss_coeff = 0.0
        model = JTDeepseekV3ForCausalLM(config)
        model._step_loss_metrics = torch.tensor([2.0, 0.3, 0.0])
        model._metric_micro_batches = 1
        self.assertEqual(model.get_logging_metrics()["training/load_balancing_loss"].item(), 0.0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_public_batch_fields_reach_model_without_mapping(self):
        """Feature: Native Trainer input contract.

        Description: Pass bookkeeping labels, shifted labels, mask and positions directly.
        Expectation: Shifted targets and positions reach the objective unchanged; missing targets fail.
        """
        model = JTDeepseekV3ForCausalLM(small_config())
        tokens = torch.arange(8).unsqueeze(0)
        shifted, mask = tokens + 1, torch.ones(1, 8)
        positions = tokens + 2
        values = {"loss": torch.tensor(3.), "lm_loss": torch.tensor(2.),
                  "mtp_loss": torch.tensor(0.9), "aux_loss": torch.tensor(0.1)}
        with patch.object(model, "compute_jt_losses", return_value=values) as compute:
            model(input_ids=tokens, labels=tokens, shift_labels=shifted, loss_mask=mask,
                  position_ids=positions, attention_mask=None)
        self.assertIs(compute.call_args.args[1], shifted)
        self.assertIs(compute.call_args.kwargs["position_ids"], positions)
        with self.assertRaisesRegex(ValueError, "attention_mask=None"):
            model(input_ids=tokens, shift_labels=shifted, loss_mask=mask, attention_mask=torch.ones_like(tokens))
        with self.assertRaisesRegex(ValueError, "explicit shift_labels"):
            model(input_ids=tokens, labels=tokens, loss_mask=mask)
