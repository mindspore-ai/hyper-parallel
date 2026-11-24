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
"""Complete Qwen3 VL MoE image fusion, deepstack and source-gradient parity."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

from __future__ import annotations

import copy
import unittest

import torch
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeTextConfig, Qwen3VLMoeVisionConfig
from transformers import Qwen3VLMoeForConditionalGeneration

from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.qwen3_vl_moe.qwen3_vl_moe_spec import MODEL_SPEC
from hyper_parallel.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration as TrainingModel,
)


class TestQwen3VLMoeSpec(unittest.TestCase):
    """Exercise actual media placeholders, vision deepstack and routed text experts together."""

    text_config_class = Qwen3VLMoeTextConfig
    vision_config_class = Qwen3VLMoeVisionConfig
    config_class = Qwen3VLMoeConfig
    vlm_model_class = Qwen3VLMoeForConditionalGeneration
    vlm_spec = MODEL_SPEC
    extra_text_fields = {"num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16}

    def test_complete_vlm_vision_replacement_and_fusion(self) -> None:
        """Feature: Complete VLM default replacement.
        Description: Execute actual media encoding, placeholder fusion, MRoPE and the hybrid decoder.
        Expectation: Outputs and all vision/language gradients match HF without any feature bridge.
        """
        text = self.text_config_class(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [3, 3, 2],
            },
            **self.extra_text_fields,
        )
        vision = self.vision_config_class(
            depth=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0, 1],
        )
        config = self.config_class(
            text_config=text,
            vision_config=vision,
            image_token_id=32,
            vision_start_token_id=33,
            vision_end_token_id=34,
            video_token_id=35,
        )
        config._attn_implementation = "eager"
        source = self.vlm_model_class(config).eval()
        target = TrainingModel(copy.deepcopy(config)).eval()
        target.load_state_dict(source.state_dict())
        target, _ = prepare_model_components(target, build_module_context(model_spec=self.vlm_spec, config=config))
        self.assertIsNotNone(ShardingPlanner().plan(target, LogicalDeviceMesh(("dp",), (1,))))
        ids = torch.tensor([[1, 33, 32, 32, 32, 32, 34, 2, 3]])
        pixels = torch.randn(16, 24)
        grid = torch.tensor([[1, 4, 4]])
        types = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0]])
        inputs = {
            "input_ids": ids,
            "pixel_values": pixels,
            "image_grid_thw": grid,
            "mm_token_type_ids": types,
            "use_cache": False,
        }
        expected, actual = source(**inputs).logits, target(**inputs).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=1e-4, atol=1e-6)

        source.zero_grad(set_to_none=True)
        target.zero_grad(set_to_none=True)
        targets = torch.tensor([[33, 32, -100, -100, -100, 34, 2, 3, -100]])
        oracle = source(**inputs, output_router_logits=True)
        expected_loss = torch.nn.functional.cross_entropy(oracle.logits.float().flatten(0, 1), targets.flatten())
        expected_loss = expected_loss + text.router_aux_loss_coef * oracle.aux_loss
        chunked = target(**inputs, chunk_loss_targets=targets, chunk_loss_chunk_size=3, output_router_logits=True)
        actual_loss = chunked.loss_sum / chunked.valid_token_count + chunked.aux_loss_coef * chunked.aux_loss
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
        expected_loss.backward()
        actual_loss.backward()
        for name, parameter in target.named_parameters():
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=1e-4, atol=1e-6)
