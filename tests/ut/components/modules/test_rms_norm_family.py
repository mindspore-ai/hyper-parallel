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
"""RMSNormModule family construction and state tests without NPU execution."""

import unittest
from types import SimpleNamespace

import torch

from examples.torch.module_family.config import RMSNormConfig
from hyper_parallel.components.modules import (
    OffsetRMSNormModule, RMSNormModule, RMSNorm, get_family_build_info,
    OffsetRMSNorm,
)
from hyper_parallel.components.checkpoint.conversion_ops import AddScalar
from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention import (
    GroupedQueryAttentionModule,
)
from hyper_parallel.components.modules.attention.multi_latent_attention.multi_latent_attention import (
    MultiLatentAttentionModule,
)
from hyper_parallel.components.modules.family import get_module_family
from hyper_parallel.components.modules.common.rms_norm import NativeRMSNormModule, NativeOffsetRMSNormModule
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementSpec


class DirectSource(torch.nn.RMSNorm):
    """HF-style direct-scale source call contract."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use the source's named hidden-state argument."""
        return torch.nn.functional.rms_norm(hidden_states, self.normalized_shape, self.weight, self.eps)


class OffsetSource(torch.nn.RMSNorm):
    """Unit-offset source call contract."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize with the source unit-offset scale."""
        return torch.nn.functional.rms_norm(x, self.normalized_shape, self.weight + 1, self.eps)


class TestRMSNormFamily(unittest.TestCase):
    """Validate production constructors and provenance."""

    def test_attention_config_uses_fused_family_norms(self):
        """Master attention constructors retain fused norms through the config-based family API."""
        config = SimpleNamespace(
            hidden_size=16, num_attention_heads=2, num_key_value_heads=1, qk_norm=True,
            q_lora_rank=8, kv_lora_rank=4, qk_nope_head_dim=4, qk_rope_head_dim=4,
            v_head_dim=8, rms_norm_eps=1e-5, dtype=torch.bfloat16,
        )
        for device in ("cpu", "meta"):
            with self.subTest(device=device), torch.device(device):
                gqa = GroupedQueryAttentionModule(config=config)
                mla = MultiLatentAttentionModule(config=config)
                for norm, width in ((gqa.q_norm, 8), (gqa.k_norm, 8),
                                    (mla.q_a_layernorm, 8), (mla.kv_a_layernorm, 4)):
                    self.assertIs(type(norm), RMSNormModule)
                    self.assertEqual(norm.weight.shape, (width,))
                    self.assertEqual(norm.weight.dtype, torch.bfloat16)
                    self.assertEqual(norm.weight.device.type, device)
                    self.assertEqual(norm.eps, config.rms_norm_eps)
                    self.assertEqual(get_family_build_info(norm).origin, "direct")

    def test_model_owned_config(self):
        """Both normalization contracts accept model-owned configuration attributes."""
        for factory in (RMSNorm, OffsetRMSNorm):
            with self.subTest(factory=factory.__name__):
                member = factory(config=SimpleNamespace(hidden_size=8, eps=1e-6))
                torch.testing.assert_close(member.weight, torch.ones(8) if factory is RMSNorm else torch.zeros(8))
                self.assertEqual(member.eps, 1e-6)
                with self.assertRaises(ValueError):
                    factory(config=SimpleNamespace(hidden_size=0, eps=1e-6))
                with self.assertRaises(ValueError):
                    factory(config=SimpleNamespace(hidden_size=8, eps=float("nan")))

    def test_direct_and_legacy(self):
        """Tracked construction retains the original concrete type and layout."""
        member = RMSNorm(config=RMSNormConfig(8, 1e-6))
        self.assertIsInstance(member, RMSNormModule)
        self.assertEqual(get_family_build_info(member).origin, "direct")
        torch.testing.assert_close(member.weight, RMSNormModule(8).weight)
        self.assertEqual(member.eps, 1e-6)

    def test_source_identity(self):
        """Replacement preserves loaded parameters, freezing and eval state."""
        source = torch.nn.RMSNorm(8, eps=1e-6)
        source.weight.requires_grad_(False)
        source.eval()
        member = RMSNorm(module=source)
        self.assertIs(member.weight, source.weight)
        self.assertFalse(member.training)
        self.assertFalse(member.weight.requires_grad)
        self.assertEqual(get_family_build_info(member).origin, "replacement")

    def test_meta_and_checkpoint(self):
        """Materialization reset and state restoration preserve direct scale."""
        with torch.device("meta"):
            member = RMSNorm(config=RMSNormConfig(8))
        self.assertTrue(member.weight.is_meta)
        member.to_empty(device="cpu")
        member.reset_parameters()
        torch.testing.assert_close(member.weight, torch.ones(8))
        restored = RMSNorm(config=RMSNormConfig(8))
        restored.load_state_dict(member.state_dict())
        torch.testing.assert_close(restored.weight, member.weight)

    def test_production_replacement_and_idempotent_prepare(self):
        """Actual model preparation installs tracked members without double conversion."""
        for factory, offset in ((RMSNorm, False), (OffsetRMSNorm, True)):
            with self.subTest(offset=offset):
                source = (OffsetSource if offset else DirectSource)(8, eps=1e-6)
                original = torch.clone(source.weight).detach()
                model = torch.nn.Sequential(source)
                context = {"module_replacements": (ModuleReplacementSpec(("0",), factory),)}
                model, _ = prepare_model_components(model, context)
                torch.testing.assert_close(model[0].weight, original)
                self.assertIs(model[0].weight, source.weight)
                self.assertEqual(get_family_build_info(model[0]).origin, "replacement")
                parameter = model[0].weight
                model, _ = prepare_model_components(model, context)
                self.assertIs(model[0].weight, parameter)
                torch.testing.assert_close(model[0].weight, original)

    def test_offset_config_meta(self):
        """Direct offset construction initializes offsets to zero on either backend."""
        with torch.device("meta"):
            member = OffsetRMSNorm(config=RMSNormConfig(8))
        member.to_empty(device="cpu")
        member.reset_parameters()
        torch.testing.assert_close(member.weight, torch.zeros(8))
        self.assertEqual(get_family_build_info(member).origin, "direct")

    def test_invalid_configuration(self):
        """Invalid sizes, epsilons and conflicting sources fail early."""
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                RMSNormConfig(size)
        for eps in (0, -1, float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                RMSNormConfig(8, eps)
        with self.assertRaises(ValueError):
            RMSNorm(config=RMSNormConfig(8), module=torch.nn.RMSNorm(8))

    def test_offset_loaded_state_and_roundtrip(self):
        """Ordinary offset sources retain loaded values and parameter identity."""
        source = torch.nn.RMSNorm(8, eps=1e-6)
        with torch.no_grad():
            source.weight.copy_(torch.linspace(-0.2, 0.3, 8))
        source.weight.requires_grad_(False)
        source.eval()
        member = NativeOffsetRMSNormModule(module=source)
        self.assertIs(member.weight, source.weight)
        torch.testing.assert_close(member.weight, source.weight)
        self.assertFalse(member.training)
        self.assertFalse(member.weight.requires_grad)
        converted = AddScalar(1).convert({"weight": source.weight}, ["weight"], ["weight"])
        exported = AddScalar(1).reverse_op.convert(converted, ["weight"], ["weight"])
        torch.testing.assert_close(exported["weight"], source.weight)
        restored = NativeOffsetRMSNormModule(module=source)
        restored.load_state_dict(member.state_dict())
        torch.testing.assert_close(restored.weight, member.weight)

    def test_native_and_npu_members_preserve_parameter_representation(self):
        """Legacy members share Parameters and checkpoints while using OpSelect."""
        pairs = ((RMSNorm, NativeRMSNormModule, RMSNormModule),
                 (OffsetRMSNorm, NativeOffsetRMSNormModule, OffsetRMSNormModule))
        for factory, native_type, npu_type in pairs:
            with self.subTest(factory=factory.__name__):
                native = factory(config=RMSNormConfig(8))
                fused = factory(module=native, use_fused_ops=True)
                adopted = factory(module=fused, use_fused_ops=False)
                self.assertIs(type(native), native_type)
                self.assertIs(type(fused), npu_type)
                self.assertIs(type(adopted), native_type)
                self.assertIs(fused.weight, native.weight)
                self.assertIs(adopted.weight, native.weight)
                self.assertEqual(get_family_build_info(fused).resolved.variant, "npu")
                self.assertEqual(get_family_build_info(adopted).resolved.variant, "local")
                self.assertEqual(set(get_module_family(factory).variants), {"local", "npu"})
                self.assertEqual(tuple(native.state_dict()), tuple(fused.state_dict()))
                values = torch.randn(2, 8)
                torch.testing.assert_close(native(values), adopted(values))
                torch.testing.assert_close(fused(values), native(values))
                for backend in ("auto", "torch_npu", "cann", "unknown"):
                    with self.subTest(use_fused_ops=backend), self.assertRaises((TypeError, ValueError)):
                        factory(config=RMSNormConfig(8), use_fused_ops=backend)

    def test_explicit_legacy_scale_conversion(self):
        """Legacy direct-scale values are converted only with an explicit representation declaration."""
        source = torch.nn.RMSNorm(8, eps=1e-6).eval()
        source.weight.data.copy_(torch.linspace(0.8, 1.2, 8))
        source.weight.requires_grad_(False)
        member = OffsetRMSNorm(module=source, source_representation="scale")
        torch.testing.assert_close(member.weight, source.weight - 1)
        self.assertFalse(member.training)
        self.assertFalse(member.weight.requires_grad)
        conversion = member.make_transforms()[0].operations[0]
        converted = conversion.convert({"weight": source.weight}, ["weight"], ["weight"])
        torch.testing.assert_close(converted["weight"], member.weight)
        exported = conversion.reverse_op.convert(converted, ["weight"], ["weight"])
        torch.testing.assert_close(exported["weight"], source.weight)
        values = torch.randn(3, 8)
        torch.testing.assert_close(member(values), source(values))
        target = OffsetRMSNorm(config=RMSNormConfig(8, 1e-6))
        target.load_state_dict(member.state_dict())
        torch.testing.assert_close(target(values), member(values))
        self.assertEqual(target.make_transforms(), [])
        with self.assertRaises(ValueError):
            OffsetRMSNorm(module=target, source_representation="scale")

    def test_functional_cast_policies_and_all_gradients(self):
        """An independent RMS formula checks both rounding contracts and input/scale gradients."""
        for order in ("before_weight", "after_weight"):
            for dtype in (torch.float32, torch.bfloat16):
                for weight_dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(order=order, dtype=dtype, weight_dtype=weight_dtype):
                        x = torch.randn(3, 8, dtype=dtype, requires_grad=True)
                        scale = torch.randn(8, dtype=weight_dtype, requires_grad=True)
                        weight = torch.nn.Parameter(scale.detach().clone())
                        other = x.detach().clone().requires_grad_()
                        normalized = other.float() / (other.float().square().sum(-1, keepdim=True) / 8 + 1e-6).sqrt()
                        reference = (scale * normalized.to(dtype) if order == "before_weight"
                                     else (normalized * scale.float()).to(dtype))
                        member = NativeRMSNormModule(8, eps=1e-6, cast_order=order)
                        member.weight = weight
                        actual = member(x)
                        tolerance = 3e-6 if dtype == weight_dtype == torch.float32 else 3e-2
                        torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
                        upstream = torch.randn_like(reference)
                        actual.backward(upstream)
                        reference.backward(upstream)
                        torch.testing.assert_close(x.grad, other.grad, atol=tolerance, rtol=tolerance)
                        torch.testing.assert_close(weight.grad, scale.grad, atol=tolerance, rtol=tolerance)
