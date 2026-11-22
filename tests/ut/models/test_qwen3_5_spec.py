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
"""Real hybrid text-model parity for the dense Qwen3.5 default specification."""
# Transformers @strict generates keyword configuration constructors at runtime.
# pylint: disable=unexpected-keyword-arg

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP, init_device_mesh
from hyper_parallel.distributed import ShardingPlanner
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderLocalInputs
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.distributed.encoder_parallel.transport import EncoderParallelBatchTransport
from hyper_parallel.distributed.mesh import LogicalDeviceMesh
from hyper_parallel.components.modules.module_family import get_family_build_info
from hyper_parallel.models.module_build import build_module_context, prepare_model_components
from hyper_parallel.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as DenseChunkedModel
from hyper_parallel.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as DenseNativeVLM
from hyper_parallel.models.qwen3_5.qwen3_5_spec import MODEL_SPEC, TEXT_SPEC
from hyper_parallel.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM as MoeChunkedModel
from hyper_parallel.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration as MoeNativeVLM
from hyper_parallel.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    bind_qwen_vlm_parallel,
)
from hyper_parallel.models.qwen3_5_moe.qwen3_5_moe_spec import MODEL_SPEC as MOE_SPEC
from hyper_parallel.models.qwen3_5_moe.qwen3_5_moe_spec import TEXT_SPEC as MOE_TEXT_SPEC
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMCPInputs


def _convert_gdn_state(model, state):
    """Apply the upstream family's declared reversible row conversion to a source state."""
    converted = dict(state)
    for prefix, module in model.named_modules():
        info = get_family_build_info(module)
        if info is None or info.resolved.family.name != "gated_delta_net":
            continue
        local = {name.removeprefix(prefix + "."): value for name, value in state.items()
                 if name.startswith(prefix + ".")}
        for transform in module.make_transforms():
            local.update(transform.operations[0].convert(local, transform.source_patterns, transform.target_patterns))
        converted.update({prefix + "." + name: value for name, value in local.items()})
    return converted


class TestQwen35Spec(unittest.TestCase):
    """Verify hybrid-layer roles, parameter identity, MRoPE and complete decoder gradients."""

    text_config_class = Qwen3_5TextConfig
    vision_config_class = Qwen3_5VisionConfig
    config_class = Qwen3_5Config
    chunked_model_class = DenseChunkedModel
    text_model_class = Qwen3_5ForCausalLM
    vlm_model_class = Qwen3_5ForConditionalGeneration
    native_vlm_class = DenseNativeVLM
    text_spec, vlm_spec = TEXT_SPEC, MODEL_SPEC
    extra_text_fields = {}

    def test_hybrid_text_source_parity(self) -> None:
        """Feature: Qwen3.5 public default families.
        Description: Run a real four-layer mixed GDN/gated-attention model, including padding and MRoPE.
        Expectation: Logical state, logits and every parameter gradient agree with HF without adapters.
        """
        torch.set_num_threads(2)
        config = self.text_config_class(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=64,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 1.0,
                "mrope_section": [3, 3, 2],
            },
            **self.extra_text_fields,
        )
        config._attn_implementation = "eager"
        config.use_cache = False
        source = self.text_model_class(config).eval()
        target_source = copy.deepcopy(source)
        parameters = dict(target_source.named_parameters())
        target, _ = prepare_model_components(
            target_source, build_module_context(model_spec=self.text_spec, config=config)
        )
        self.assertIsNotNone(ShardingPlanner().plan(target, LogicalDeviceMesh(("dp",), (1,))))
        converted = _convert_gdn_state(target, parameters)
        for name, parameter in parameters.items():
            if converted[name] is parameter:
                self.assertIs(parameter, target.get_parameter(name))
            else:
                torch.testing.assert_close(target.get_parameter(name), converted[name], rtol=0, atol=0)
        ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        expected = source(input_ids=ids, attention_mask=mask, use_cache=False).logits
        actual = target(input_ids=ids, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        expected_gradients = _convert_gdn_state(target, {name: p.grad for name, p in source.named_parameters()})
        for name, parameter in target.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=1e-4, atol=1e-6)

    def test_complete_vlm_vision_replacement_and_fusion(self) -> None:
        """Feature: Complete VLM default replacement.
        Description: Execute actual media encoding, placeholder fusion, MRoPE and the hybrid decoder.
        Expectation: Outputs and all vision/language gradients match HF without any feature bridge.
        """
        text = self.text_config_class(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=64,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
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
        target = copy.deepcopy(source)
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
        native = self.native_vlm_class(copy.deepcopy(config)).eval()
        native.load_state_dict(source.state_dict())
        bind_qwen_vlm_parallel(native, cp_enabled=False)
        self.assertNotIn("forward", vars(native))
        torch.testing.assert_close(native(**inputs).logits, expected, rtol=0, atol=0)
        geometry = QwenVLMCPInputs(0, ids.shape[1], ids, torch.ones_like(ids), types,
                                  torch.full_like(ids, -100), 1)
        owned = native(**inputs, vlm_cp_inputs=geometry).logits
        torch.testing.assert_close(owned, expected, rtol=0, atol=0)
        native, _ = prepare_model_components(native, build_module_context(model_spec=self.vlm_spec, config=config))
        mesh = init_device_mesh("cpu", (1, 1, 1), mesh_dim_names=("dp", "cp", "tp"),
                                rank_list=(0,), init_backend=False)
        context = SimpleNamespace(device_mesh=mesh, cp_size=1, tp_size=1, cp_rank=0, tp_rank=0,
                                  pp_size=1, sequence_parallel=False, loss_parallel=False)
        runtime = EncoderParallelRuntime(
            config=EncoderParallelConfig(enabled=True), mesh_context=context, model=native,
            spec=self.vlm_spec.encoder_parallel(), execution_options={"micro_batch_size": 1},
        )
        transport = EncoderParallelBatchTransport(runtime=runtime)
        for invalid in (dict(inputs, pixel_values=torch.randn(16, 23)),
                        dict(inputs, pixel_values_videos=torch.randn(16, 24))):
            with self.subTest(invalid_fields=tuple(invalid)):
                prepared = transport._prepare_source(invalid)
                self.assertIsNotNone(prepared["_encoder_transport_metadata"][2])
        with patch("hyper_parallel.models." + self.vlm_spec.model_type + "." + self.vlm_spec.model_type
                   + "_spec.HSDPModule", type(native.model)):
            with self.assertRaisesRegex(NotImplementedError, "complete model core"):
                self.vlm_spec.encoder_parallel().bind(native)
        plan = runtime.finalize_plan(*runtime.preflight_batch(inputs))
        encoded = native(**inputs, encoder_parallel_plan=plan).logits
        torch.testing.assert_close(encoded, expected, rtol=1e-5, atol=1e-6)
        transported = {key: value for key, value in inputs.items()
                       if key not in ("pixel_values", "image_grid_thw")}
        payload = EncoderLocalInputs(runtime.bridge.pack_payload(inputs, plan.local_segments))
        restored = native(**transported, encoder_parallel_plan=plan, encoder_parallel_inputs=payload).logits
        torch.testing.assert_close(restored, expected, rtol=1e-5, atol=1e-6)
        self.assertNotIn("forward", vars(native.model.visual))
        _DEVICE_MESH_MAP.clear()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().mean().backward()
        actual.square().mean().backward()
        reference = dict(source.named_parameters())
        for name, parameter in target.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, reference[name].grad, rtol=1e-4, atol=1e-6)


class TestQwen35MoeSpec(TestQwen35Spec):
    """Apply the same complete text/VLM parity checks to natural MoE routes and shared experts."""

    text_config_class = Qwen3_5MoeTextConfig
    vision_config_class = Qwen3_5MoeVisionConfig
    config_class = Qwen3_5MoeConfig
    chunked_model_class = MoeChunkedModel
    text_model_class = Qwen3_5MoeForCausalLM
    vlm_model_class = Qwen3_5MoeForConditionalGeneration
    native_vlm_class = MoeNativeVLM
    text_spec, vlm_spec = MOE_TEXT_SPEC, MOE_SPEC
    extra_text_fields = {
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24,
    }
