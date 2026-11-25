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
"""Real native K2.6 media fusion through complete public vision and projector families."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F
from transformers import DeepseekV3Config
from transformers.models.kimi_k25.configuration_kimi_k25 import Kimi_K25Config, Kimi_K25VisionConfig
from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25ForConditionalGeneration

from hyper_parallel.models.kimi_k26.kimi_k26_spec import MODEL_SPEC
from hyper_parallel.models.kimi_k26.modeling_kimi_k26 import Kimi_K25ForConditionalGeneration as ChunkedModel
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.models.module_build import build_module_context, prepare_model_components


class TestKimiK26Spec(unittest.TestCase):
    """Keep native media placeholder fusion and independent text/vision matrix boundaries."""

    def test_complete_image_source_parity(self) -> None:
        """Feature: Native K2.6 full media composition.
        Description: Replace complete encoders/projector and a dense-to-MoE MLA decoder with actual image patches.
        Expectation: Source state identities, fused logits and all vision/language gradients agree.
        """
        torch.set_num_threads(2)
        text = DeepseekV3Config(
            hidden_size=32,
            intermediate_size=48,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            first_k_dense_replace=1,
            n_routed_experts=4,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            vocab_size=64,
        )
        vision = Kimi_K25VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=2,
            pos_emb_height=4,
            pos_emb_width=4,
            pos_emb_time=4,
            merge_kernel_size=[2, 2],
        )
        config = Kimi_K25Config(
            text_config=text,
            vision_config=vision,
            projection_hidden_size=16,
            image_token_id=32,
            video_token_id=35,
            vision_start_token_id=33,
            vision_end_token_id=34,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        initial_rng = torch.get_rng_state().clone()
        source = Kimi_K25ForConditionalGeneration(config).eval()
        source_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        native = ChunkedModel(copy.deepcopy(config)).eval()
        torch.testing.assert_close(torch.get_rng_state(), source_rng, rtol=0, atol=0)
        self.assertEqual(set(native.state_dict()), set(source.state_dict()))
        for name, value in source.state_dict().items():
            torch.testing.assert_close(native.state_dict()[name], value, rtol=0, atol=0)
        adopted = copy.deepcopy(source)
        parameters = dict(adopted.named_parameters())
        target, _ = prepare_model_components(adopted, build_module_context(model_spec=MODEL_SPEC, config=config))
        self.assertIsNotNone(ShardingPlanner().plan(target, LogicalDeviceMesh(("dp",), (1,))))
        for name, parameter in parameters.items():
            self.assertIs(parameter, target.get_parameter(name))
        ids = torch.tensor([[1, 33, 32, 32, 32, 32, 34, 2, 3]])
        inputs = {
            "input_ids": ids,
            "pixel_values": torch.randn(16, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "use_cache": False,
        }
        expected, actual = source(**inputs).logits, target(**inputs).logits
        torch.testing.assert_close(native(**inputs).logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=1e-4, atol=1e-6)

        source.zero_grad(set_to_none=True)
        chunked = ChunkedModel(config).eval()
        chunked.load_state_dict(source.state_dict())
        chunked, _ = prepare_model_components(chunked, build_module_context(model_spec=MODEL_SPEC, config=config))
        targets = torch.tensor([[2, -100, -100, -100, -100, -100, 2, 3, -100]])
        result = chunked(**inputs, chunk_loss_targets=targets, chunk_loss_chunk_size=2)
        expected_loss = F.cross_entropy(source(**inputs).logits.flatten(0, 1), targets.flatten(), reduction="sum")
        torch.testing.assert_close(result.loss_sum, expected_loss, rtol=1e-6, atol=1e-6)
        self.assertIsNone(result.logits)
        self.assertEqual(result.valid_token_count.item(), 3)
        expected_loss.backward()
        result.loss_sum.backward()
        reference = dict(source.named_parameters())
        for name, parameter in chunked.named_parameters():
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=2e-4, atol=1e-6)
