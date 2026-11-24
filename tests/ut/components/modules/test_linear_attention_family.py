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
"""Essential linear-attention family ownership, promotion and packed-lifetime contracts."""
import unittest
from copy import deepcopy
from functools import partial
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from hyper_parallel.components.checkpoint.weight_conversion import revert_weight_conversion
from hyper_parallel.components.modules import GatedDeltaNet, KimiDeltaAttention
from hyper_parallel.components.modules.module_family import FamilyShardingContext, get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.models.module_build import prepare_model_components
from hyper_parallel.models.replacement import ModuleReplacementRequest
from tests.common.mark_utils import arg_mark


def _layer(kind, **options):
    if kind == "kda":
        return KimiDeltaAttention(hidden_size=16, num_heads=2, head_k_dim=4, head_v_dim=3,
                                  chunk_size=4, **options)
    return GatedDeltaNet(config=SimpleNamespace(hidden_size=16, linear_num_key_heads=2,
                         linear_num_value_heads=4, linear_key_head_dim=4, linear_value_head_dim=3),
                         chunk_size=4, **options)


class TestLinearAttentionFamily(unittest.TestCase):
    """Test contracts introduced by structure migration, reusing existing mathematical tests."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_source_identity_alias_and_promotion(self):
        """Feature: Linear-attention module and CP compatibility.
        Description: Local-to-CP selection preserves frozen/shared parameters and adds no source prefix.
        Expectation: Parameters, aliases and frozen flags survive local-to-CP promotion.
        """
        for kind, factory in (("kda", KimiDeltaAttention), ("gdn", GatedDeltaNet)):
            with self.subTest(kind=kind):
                source = _layer(kind)
                source.A_log.requires_grad_(False)
                source.source_activation = nn.SiLU()
                if kind == "kda":
                    source.k_proj = source.q_proj
                local = factory(module=source, chunk_size=4, ulysses_degree=None)
                self.assertIs(local.source_activation, source.source_activation)
                names = dict(source.named_parameters(remove_duplicate=False))
                self.assertEqual(tuple(local.state_dict()), tuple(source.state_dict()))
                for name, parameter in local.named_parameters(remove_duplicate=False):
                    self.assertIs(parameter, names[name])
                model = nn.Module()
                model.attention = local
                model.alias = local
                model, _ = prepare_model_components(model, {"cp_size": 2})
                self.assertIs(model.attention, model.alias)
                self.assertTrue(type(model.attention).__name__.startswith("CP"))
                self.assertFalse(model.attention.A_log.requires_grad)
                for name, parameter in model.attention.named_parameters(remove_duplicate=False):
                    self.assertIs(parameter, names[name])
                with self.assertRaisesRegex(RuntimeError, "Bind"):
                    model.attention(torch.randn(1, 9, 16))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_flat_token_layout_and_capability_checks(self):
        """Feature: Linear-attention module and CP compatibility.
        Description: Flat TND shards tokens on axis zero; unsupported degrees/backends fail before binding.
        Expectation: Flat tokens use axis zero and unsupported execution fails before binding.
        """
        for kind in ("kda", "gdn"):
            for layout, axis in (("BSH", 1), ("TH", 0)):
                with self.subTest(kind=kind, layout=layout):
                    module = _layer(kind, input_layout=layout, context={"cp_size": 2}, ulysses_degree=None)
                    context = FamilyShardingContext(("cp",), {"cp": 2, "tp": 1}, {})
                    member = get_family_build_info(module).resolved
                    spec = member.family.sharding(module=module, module_fqn="", member=member, context=context)
                    self.assertEqual(spec.in_src["hidden_states"]["cp"], Shard(axis))
                    self.assertEqual(spec.in_src, spec.in_dst)
                    self.assertEqual(spec.out_src, spec.out_dst)
                    self.assertEqual(set(spec.params), set(dict(module.named_parameters())))
        for kind in ("kda", "gdn"):
            for layout, axis in (("BSH", 1), ("TH", 0)):
                for sp in (False, True):
                    module = _layer(kind, input_layout=layout, context={"tp_size": 2})
                    context = FamilyShardingContext(("tp",), {"tp": 2, "cp": 1}, {}, sequence_parallel=sp)
                    member = get_family_build_info(module).resolved
                    spec = member.family.sharding(module=module, module_fqn="", member=member, context=context)
                    self.assertEqual(spec.in_src["hidden_states"]["tp"], Shard(axis) if sp else Replicate())
                    self.assertEqual(spec.in_dst["hidden_states"]["tp"], Replicate())
                    self.assertEqual(spec.out_src["output"]["tp"], Partial())
                    self.assertEqual(spec.out_dst["output"]["tp"], Shard(axis) if sp else Replicate())
                    self.assertIn("num_k_heads", spec.tp_divide_attrs)
        with self.assertRaisesRegex(ValueError, "TP times Ulysses"):
            _layer("gdn", context={"tp_size": 2, "cp_size": 2}, ulysses_degree=2)
        with self.assertRaisesRegex(NotImplementedError, "GDN state CP"):
            GatedDeltaNet(config=SimpleNamespace(linear_num_key_heads=8, linear_num_value_heads=8),
                          backend="ascendc", context={"cp_size": 4})
        with self.assertRaisesRegex(ValueError, "ulysses_degree"):
            _layer("kda", context={"cp_size": 4}, ulysses_degree=3)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_gdn_grouped_source_conversion_and_gradients(self):
        """Feature: TP-compatible GDN weight layout.
        Description: Compare grouped QKV/conv with raw HF GDN, including packed resets and export.
        Expectation: Outputs, dX, parameter gradients and inverse checkpoint conversion agree.
        """
        # Qwen3.5 uses value/key head ratios 1, 2, 3 and 4 across model sizes.
        for ratio, offsets in product((1, 2, 3, 4), ((0, 9), (0, 2, 6, 9))):
            config = qwen.Qwen3_5TextConfig.from_dict({
                "hidden_size": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2 * ratio,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 3,
            })
            with self.subTest(ratio=ratio, offsets=offsets), patch.object(qwen, "FusedRMSNormGated", None, create=True):
                torch.manual_seed(319)
                reference = qwen.Qwen3_5GatedDeltaNet(config, 0)
                reference.causal_conv1d_fn = None
                reference.chunk_gated_delta_rule = partial(qwen.torch_chunk_gated_delta_rule, chunk_size=4)
                reference.in_proj_qkv.bias = nn.Parameter(torch.randn(reference.conv_dim) * .01)
                reference.conv1d.bias = nn.Parameter(torch.randn(reference.conv_dim) * .01)
                reference.A_log.requires_grad_(False)
                source = deepcopy(reference)
                original = {name: value.clone() for name, value in source.state_dict().items()}
                model = nn.ModuleDict({"attention": source})
                request = ModuleReplacementRequest(("attention",), GatedDeltaNet, {"chunk_size": 4})
                model, transforms = prepare_model_components(model, {"module_replacement_requests": (request,)})
                model.config = config
                # The checkpoint loader retains the applied mapping for source-format export.
                model._weight_conversions = transforms  # pylint: disable=protected-access
                candidate = model.attention
                with self.assertRaisesRegex(TypeError, "hidden_states"):
                    candidate()
                with self.assertRaisesRegex(TypeError, "positional arguments"):
                    candidate(torch.randn(1, 9, 16), None, None, None)
                promoted = GatedDeltaNet(module=candidate, chunk_size=4, context={"cp_size": 2},
                                         ulysses_degree=None)
                self.assertIs(promoted.in_proj_qkv.weight, candidate.in_proj_qkv.weight)
                self.assertFalse(candidate.A_log.requires_grad)
                for name, tensor in source.state_dict().items():
                    torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
                restored = revert_weight_conversion(model, model.state_dict())
                for name, tensor in original.items():
                    torch.testing.assert_close(restored[f"attention.{name}"], tensor, rtol=0, atol=0)
                inputs = (torch.randn(1, 9, 16) * .1).requires_grad_()
                copied = inputs.detach().clone().requires_grad_()
                expected = torch.cat([reference(inputs[:, start:end])
                                      for start, end in zip(offsets, offsets[1:])], dim=1)
                actual = (candidate(copied, None, None, cu_seqlens=offsets) if len(offsets) == 2 else
                          candidate(hidden_states=copied, cu_seqlens=offsets))
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                gradient = torch.randn_like(actual)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.testing.assert_close(copied.grad, inputs.grad, rtol=3e-5, atol=3e-6)
                gradients = {name: parameter.grad for name, parameter in reference.named_parameters()}
                expected_gradients = dict(gradients)
                for transform in candidate.make_transforms():
                    expected_gradients.update(transform.operations[0].convert(
                        gradients, transform.source_patterns, transform.target_patterns))
                for name, parameter in candidate.named_parameters():
                    torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=5e-5, atol=5e-6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_packed_two_live_calls_checkpoint_and_optimizer(self):
        """Feature: Linear-attention module and CP compatibility.
        Description: Different packed metadata and reverse backward preserve the actual AdamW update.
        Expectation: Two live calls and checkpoint recomputation preserve AdamW updates.
        """
        # CPU-only inputs must not inherit torch-npu's checkpoint device default.
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        for kind in ("kda", "gdn"):
            with self.subTest(kind=kind):
                torch.manual_seed(318)
                baseline = _layer(kind)
                candidate = deepcopy(baseline)
                values = [torch.randn(9, 16) * .1, torch.randn(9, 16) * .1]
                upstream = [torch.randn_like(value) for value in values]
                offsets = [(0, 3, 9), (0, 1, 5, 9)]
                # torch-npu's automatic foreach discovery queries hardware even for CPU parameters.
                optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
                              for model in (baseline, candidate)]
                for step in range(2):
                    for model, optimizer, live in zip((baseline, candidate), optimizers, (False, True)):
                        optimizer.zero_grad(set_to_none=True)
                        if live:
                            outputs = [checkpoint(model, value, use_reentrant=False, cu_seqlens=boundary)
                                       for value, boundary in zip(values, offsets)]
                            for index in (1, 0):
                                outputs[index].backward(upstream[index])
                        else:
                            for index in (1, 0):
                                model(values[index], cu_seqlens=offsets[index]).backward(upstream[index])
                        optimizer.step()
                    for first, second in zip(baseline.parameters(), candidate.parameters()):
                        torch.testing.assert_close(first, second, rtol=0, atol=0, msg=f"{kind}/step={step}")
                    self.assertEqual(get_family_build_info(candidate).resolved.variant, "local")
