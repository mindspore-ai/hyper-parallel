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
"""Real HF source precision/state contracts and native vision/shared integration."""
import unittest
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3MLP
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm, Qwen3_5MLP
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeMLP
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm, DeepseekV3MLP
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32RMSNorm, DeepseekV32MLP
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm, DeepseekV4MLP

from examples.torch.module_family.config import RMSNormConfig, SharedExpertConfig, SwiGLUMLPConfig
from hyper_parallel.components.modules import RMSNorm, OffsetRMSNorm, SharedExpert, SwiGLUMLP, get_family_build_info
from hyper_parallel.components.modules.mlp import swiglu_activation, W1W2SwiGLUMLPModule
from hyper_parallel.components.modules.common.rms_norm import RMSNormModule
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower
from hyper_parallel.models._transformers.model_builder import validate_model_init_dtype


def _mlp_config():
    """Supply the actual attributes read by each HF MLP constructor."""
    return SimpleNamespace(hidden_size=8, intermediate_size=12, hidden_act="silu", mlp_bias=False, swiglu_limit=0.3)


def _source_gradients(source, member, values, *, atol=2e-2, rtol=2e-2):
    """Check independent source execution with a shared upstream gradient."""
    left = values.detach().clone().requires_grad_()
    right = values.detach().clone().requires_grad_()
    expected, actual = source(left), member(right)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(left.grad, right.grad, atol=atol, rtol=rtol)
    return expected, actual


class TestFamilySourceContracts(unittest.TestCase):
    """Exercise real source behavior rather than synthetic attribute-only wrappers."""

    def setUp(self):
        """Keep source precision cases reproducible across environments."""
        torch.manual_seed(20261117)

    def test_hf_direct_norm_cast_and_all_gradients(self):
        """Direct text sources round before multiplication, including FP32 scale promotion."""
        for source_type in (Qwen3RMSNorm, DeepseekV3RMSNorm, DeepseekV32RMSNorm, DeepseekV4RMSNorm):
            for weight_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(source=source_type.__name__, dtype=weight_dtype):
                    source = source_type(8, eps=1e-6).to(dtype=weight_dtype)
                    nn.init.uniform_(source.weight, 0.1, 1.9)
                    reference = deepcopy(source)
                    member = RMSNorm(module=source)
                    self.assertEqual(member.rms_norm_cast_order, "before_weight")
                    self.assertIs(member.weight, source.weight)
                    expected, actual = _source_gradients(reference, member, torch.randn(3, 8).bfloat16(),
                                                         atol=0, rtol=0)
                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(member.weight.grad, reference.weight.grad, atol=0, rtol=0)

    def test_offset_source_retains_low_precision_offsets_and_state(self):
        """Offset scales add one in FP32 without storing a rounded direct-scale replacement."""
        source = Qwen3_5RMSNorm(8).bfloat16()
        nn.init.uniform_(source.weight, -0.3, 0.3)
        reference = deepcopy(source)
        member = OffsetRMSNorm(module=source)
        self.assertIs(member.weight, source.weight)
        self.assertEqual(member.make_transforms(), [])
        _source_gradients(reference, member, torch.randn(3, 8).bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(member.weight.grad, reference.weight.grad, atol=0, rtol=0)
        restored = OffsetRMSNorm(config=RMSNormConfig(8)).bfloat16()
        restored.load_state_dict(member.state_dict())
        values = torch.randn(3, 8).bfloat16()
        torch.testing.assert_close(restored(values), member(values), atol=0, rtol=0)
        adopted = OffsetRMSNorm(module=member)
        self.assertIs(adopted.weight, member.weight)

    def test_fp32_scale_conversion_and_meta(self):
        """Model dtype conversion retains loaded FP32 weights and gradients without rounding."""
        config = RMSNormConfig(8, fp32_weight=True)
        member = RMSNorm(config=config)
        nn.init.uniform_(member.weight, 0.1, 1.9)
        member.weight.grad = torch.rand_like(member.weight)
        weight, gradient = member.weight.detach().clone(), member.weight.grad.clone()
        member.bfloat16()
        torch.testing.assert_close(member.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(member.weight.grad, gradient, rtol=0, atol=0)
        validate_model_init_dtype(member, "bfloat16")
        with torch.device("meta"):
            meta = RMSNorm(config=config)
        meta.to_empty(device="cpu")
        meta.reset_parameters()
        self.assertEqual(meta.weight.dtype, torch.float32)
        self.assertFalse(meta.weight.is_meta)
        with self.assertRaises(ValueError):
            RMSNorm(module=nn.RMSNorm(8, eps=1e-6).bfloat16(), fp32_weight=True)

    def test_hf_mlp_and_shared_sources(self):
        """Actual Qwen/DeepSeek projection and activation contracts survive fusion."""
        for source_type in (Qwen2MoeMLP, Qwen3MLP, Qwen3_5MLP, Qwen3_5MoeMLP,
                            DeepseekV3MLP, DeepseekV32MLP, DeepseekV4MLP):
            for factory in (SwiGLUMLP, SharedExpert):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(source=source_type.__name__, factory=factory.__name__, dtype=dtype):
                        config = _mlp_config()
                        # The same extra config attribute must not invent clamp in other HF MLPs.
                        source = (source_type(config, 12) if source_type in (Qwen3_5MLP, Qwen3_5MoeMLP)
                                  else source_type(config)).to(dtype=dtype)
                        reference = deepcopy(source)
                        member = factory(module=source)
                        _source_gradients(reference, member, torch.randn(2, 3, 8).to(dtype),
                                          atol=2e-2 if dtype == torch.bfloat16 else 2e-6,
                                          rtol=2e-2 if dtype == torch.bfloat16 else 2e-5)
                        expected_gate = torch.cat((reference.gate_proj.weight.grad, reference.up_proj.weight.grad))
                        torch.testing.assert_close(member.linear_fc1.weight.grad, expected_gate, atol=2e-2, rtol=2e-2)
                        torch.testing.assert_close(member.linear_fc2.weight.grad, reference.down_proj.weight.grad,
                                                   atol=2e-2, rtol=2e-2)

    def test_v41_shared_separate_state_and_fp32_no_clamp(self):
        """V4.1 override keeps projection names and FP32 activation even with clamp disabled."""
        for limit in (0.0, -1.0, 0.3):
            with self.subTest(limit=limit):
                source = DeepseekV4MLP(_mlp_config()).bfloat16()
                source.limit = limit
                member = SharedExpert(module=source, preserve_source_layout=True,
                                      swiglu_limit=limit, activation_dtype="float32")
                self.assertIs(member.gate_proj.weight, source.gate_proj.weight)
                self.assertEqual(tuple(member.state_dict()), tuple(source.state_dict()))
                self.assertEqual(member.make_transforms(), [])
                x = torch.randn(3, 8).bfloat16().requires_grad_()
                gate, up = source.gate_proj(x).float(), source.up_proj(x).float()
                if limit > 0:
                    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
                expected = source.down_proj((torch.nn.functional.silu(gate) * up).to(x.dtype))
                torch.testing.assert_close(member(x), expected, rtol=0, atol=0)
                member(x).float().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
        gate, up = torch.randn(2, 8).bfloat16(), torch.randn(2, 8).bfloat16()
        self.assertEqual(swiglu_activation(gate, up, activation_dtype="float32").dtype, torch.float32)

    def test_native_vision_family_calls_and_state(self):
        """Real vision reuses public members while tracked provenance awaits the external copy fix."""
        config = SimpleNamespace(v41_vision_hidden_size=8, v41_vision_patch_size=2,
                                 v41_vision_num_attention_heads=2, v41_vision_rope_theta=10000.0,
                                 v41_vision_intermediate_size=12, v41_vision_num_hidden_layers=1)
        vision = DeepseekV41VisionTower(config)
        self.assertIsInstance(vision.norm, RMSNormModule)
        self.assertIsInstance(vision.blocks[0].mlp, W1W2SwiGLUMLPModule)
        self.assertIsNone(get_family_build_info(vision.norm))
        self.assertIsNone(get_family_build_info(vision.blocks[0].mlp))
        cloned = deepcopy(vision)
        self.assertIsNot(cloned.norm.weight, vision.norm.weight)
        torch.testing.assert_close(cloned.norm.weight, vision.norm.weight)
        self.assertIn("blocks.0.mlp.w1.weight", vision.state_dict())
        self.assertNotIn("blocks.0.mlp.linear_fc1.weight", vision.state_dict())
        vision.bfloat16()
        self.assertEqual(vision.norm.weight.dtype, torch.float32)
        self.assertEqual(vision.blocks[0].mlp.w1.weight.dtype, torch.bfloat16)
        validate_model_init_dtype(vision, "bfloat16")

    def test_explicit_config_and_invalid_contracts(self):
        """Incompatible policies fail rather than silently changing activation/state semantics."""
        for factory in (SharedExpert, SwiGLUMLP):
            with self.subTest(factory=factory.__name__), self.assertRaises(ValueError):
                factory(config=SharedExpertConfig(8, 12), activation_dtype="float16")
        with self.assertRaises(ValueError):
            RMSNorm(config=RMSNormConfig(8), cast_order="unknown")
        with self.assertRaises(ValueError):
            RMSNorm(module=Qwen3_5RMSNorm(8))
        with self.assertRaises(ValueError):
            OffsetRMSNorm(module=Qwen3RMSNorm(8))
        low_precision = Qwen3_5RMSNorm(8).bfloat16()
        retained = OffsetRMSNorm(module=low_precision)
        self.assertIs(retained.weight, low_precision.weight)
        converted = OffsetRMSNorm(module=Qwen3_5RMSNorm(8), fp32_weight=True)
        adopted = OffsetRMSNorm(module=converted)
        self.assertIs(adopted.weight, converted.weight)
        self.assertEqual(adopted.make_transforms(), [])
        adopted.bfloat16()
        self.assertEqual(adopted.weight.dtype, torch.float32)
        eager = OffsetRMSNorm(module=converted)
        self.assertIs(eager.weight, converted.weight)
        source = DeepseekV4MLP(_mlp_config())
        source.limit = 0
        with self.assertRaises(ValueError):
            SwiGLUMLP(module=source)
        with self.assertRaises(ValueError):
            SharedExpert(config=SharedExpertConfig(8, 12, return_bias_tuple=True), preserve_source_layout=True)
        with torch.device("meta"):
            packed = SwiGLUMLP(config=SwiGLUMLPConfig(8, 12), state_layout="w1_w2")
        packed.to_empty(device="cpu")
        packed.reset_parameters()
        self.assertEqual(tuple(packed.state_dict()), ("w1.weight", "w2.weight"))
