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
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from tests.common.mark_utils import arg_mark

from hyper_parallel.components.modules import KimiDeltaAttention, GatedDeltaNet
from hyper_parallel.components.modules.module_family import FamilyShardingContext, get_family_build_info
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.models.module_build import prepare_model_components


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
