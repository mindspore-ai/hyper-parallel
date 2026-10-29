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
"""QAT Family selects the existing strategies through real configuration."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch
import yaml
from torch import nn

from hyper_parallel.components.modules.low_precision_grouped_experts import LowPrecisionGroupedExperts
from hyper_parallel.components.modules.mxfp8_grouped_swiglu import FusedMXFP8GroupedSwiGLU
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import W4A8GroupedLinear
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import FakeW4A8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import MXFP8LinearStrategy
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.trainer.config.resolver import replace_override_path, resolve_config
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from tests.common.grouped_experts_family import TinyMoE
from tests.common.mark_utils import arg_mark
from tests.common.linear_family import linear_config
from tests.common.qat_family import deepseek_experts_model, deepseek_qat_config, install_family, qat_config


class TestQATGroupedExperts(unittest.TestCase):
    """Mock device capability checks only; registration and strategy selection stay real."""

    def setUp(self) -> None:
        """Isolate NPU availability without replacing the selected computation strategy."""
        for name in ("validate_npu_gmm_runtime", "validate_w4a8_gmm_runtime", "validate_fake_w4a8_gmm_runtime"):
            probe = patch("hyper_parallel.components.modules.low_precision_grouped_experts." + name)
            probe.start()
            self.addCleanup(probe.stop)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_named_scheme_selects_strategy_and_preserves_weights(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Named schemes choose their native/fake strategy and retain canonical parameters.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        cases = [(False, 32, "mxfp8", MXFP8GroupedLinear)]
        cases += [(fake, block, "mxfp4", FakeW4A8GroupedLinear if fake else W4A8GroupedLinear)
                  for fake in (False, True) for block in (32, 128)]
        for fake, block, fmt, expected in cases:
            with self.subTest(fake=fake, block=block, fmt=fmt):
                model = TinyMoE(canonical=True)
                weight = model.mlp.experts.gate_up_proj
                weight.requires_grad_(False)
                install_family(model, qat_config(fake=fake, block_size=block, weight_format=fmt))
                self.assertIsInstance(model.mlp.experts, LowPrecisionGroupedExperts)
                self.assertIs(type(model.mlp.experts.grouped_linear), expected)
                self.assertIs(model.mlp.experts.gate_up_proj, weight)
                self.assertFalse(weight.requires_grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scheme_conflicts_and_unknown_arguments(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Invalid choices fail in plan construction before any device computation.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        for args in ({"use_mxfp8": True}, {"use_fused_mxfp8": True}, {"fused": True}, {"unknown_option": 1}):
            with self.subTest(args=args):
                raw = qat_config().to_dict()
                raw["plan_overrides"][0]["replace_module"].update(args)
                config = resolve_config(raw)
                with self.assertRaises((ValueError, TypeError)):
                    entries_to_module_replacements(config.plan_overrides,
                                                   low_precision_config=config.training.low_precision)

    @patch("hyper_parallel.components.modules.mxfp8_grouped_swiglu.validate_npu_swiglu_runtime")
    @patch("hyper_parallel.components.modules.mxfp8_grouped_swiglu.validate_npu_gmm_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_fused_yaml_and_legacy_alias_overrides(self, gmm_probe, swiglu_probe) -> None:
        """
        Feature: Format-independent fusion selector.
        Description: YAML and CLI-style overrides accept both spellings and serialize the canonical name.
        Expectation: Fusion selects the existing implementation; disabling it returns to the scheme strategy.
        """
        for spelling in ("fused", "use_fused_mxfp8"):
            with self.subTest(spelling=spelling):
                raw = qat_config(weight_format="mxfp8").to_dict()
                raw["plan_overrides"][0]["replace_module"][spelling] = True
                config = resolve_config(yaml.safe_load(yaml.safe_dump(raw)))
                target = config.plan_overrides[0].replace_module
                self.assertTrue(target.to_dict()["fused"])
                self.assertNotIn("use_fused_mxfp8", target.to_dict())
                model = install_family(TinyMoE(), config)
                self.assertIs(type(model.mlp.experts), FusedMXFP8GroupedSwiGLU)
                for override in ("fused", "use_fused_mxfp8"):
                    config.plan_overrides[0].replace_module = replace_override_path(
                        target, [override], False, path="replace_module")
                    model = install_family(TinyMoE(), resolve_config(config.to_dict()))
                    self.assertIs(type(model.mlp.experts.grouped_linear), MXFP8GroupedLinear)
        self.assertEqual(gmm_probe.call_count, 2)
        self.assertEqual(swiglu_probe.call_count, 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scheme_name_roundtrip_and_override(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Names serialize unchanged; changing the reference selects by its contents.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        raw = qat_config().to_dict()
        raw["training"]["low_precision"]["dtype_schemes"]["different_name"] = {
            "weight_format": "mxfp8", "act_format": "mxfp8"}
        config = resolve_config(raw)
        config.plan_overrides[0].replace_module = replace_override_path(
            config.plan_overrides[0].replace_module, ["low_precision_dtype_scheme"],
            "different_name", path="replace_module")
        config = resolve_config(config.to_dict())
        self.assertFalse(config.training.low_precision.enabled)
        self.assertIsNone(config.plan_overrides[0].when)
        self.assertEqual(config.plan_overrides[0].replace_module.low_precision_dtype_scheme, "different_name")
        model = install_family(TinyMoE(), config)
        self.assertIs(type(model.mlp.experts.grouped_linear), MXFP8GroupedLinear)

    @patch("hyper_parallel.components.modules.low_precision_linear.validate_npu_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mixed_dense_and_expert_schemes(self, probe) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: One configuration independently binds a Dense and a grouped scheme.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        raw = qat_config(fake=True).to_dict()
        dense = linear_config(quantized=True).to_dict()
        raw["training"]["low_precision"]["dtype_schemes"].update(
            dense["training"]["low_precision"]["dtype_schemes"])
        entry = dense["plan_overrides"][0]
        entry["match"] = "head"
        raw["plan_overrides"].append(entry)
        model = TinyMoE()
        model.head = nn.Linear(128, 256)
        install_family(model, resolve_config(raw))
        self.assertIs(type(model.head), LowPrecisionLinear)
        self.assertIs(type(model.head.strategy), MXFP8LinearStrategy)
        self.assertIs(type(model.mlp.experts.grouped_linear), FakeW4A8GroupedLinear)
        probe.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_transposed_layout_preserves_loaded_values(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Noncanonical source weights are copied by the existing high-performance parent.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        model = TinyMoE()
        before = model.mlp.experts.gate_up_proj.detach().clone()
        install_family(model, qat_config())
        torch.testing.assert_close(model.mlp.experts.gate_up_proj, before.transpose(-2, -1))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deepseek_clamp_matches_native_forward_and_gradient(self) -> None:
        """
        Feature: Non-fused V4.1 expert activation.
        Description: Real Family selection preserves native clamp values and gradients at both boundaries.
        Expectation: All three strategies retain dtype and exact activation behavior without retaining source weights.
        """
        for fake, fmt in ((False, "mxfp8"), (False, "mxfp4"), (True, "mxfp4")):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(fake=fake, fmt=fmt, dtype=dtype):
                    model = deepseek_experts_model()
                    source = model.experts
                    source_ref = weakref.ref(source)
                    install_family(model, deepseek_qat_config(fake=fake, weight_format=fmt))
                    projected = torch.tensor(
                        [[-20., -10., -5., 0., 5., 10., 20., -20., -10., -5., 0., 5., 10., 20.]],
                        dtype=dtype, requires_grad=True)
                    reference = projected.detach().clone().requires_grad_()
                    actual = model.experts.activation_func(projected)
                    # The native hook isolates activation from projection and routing math.
                    expected = source._apply_gate(reference)  # pylint: disable=protected-access
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    actual.sum().backward()
                    expected.sum().backward()
                    torch.testing.assert_close(projected.grad, reference.grad, rtol=0, atol=0)
                    self.assertEqual(projected.grad[0, 6].item(), 0)  # gate above its upper bound
                    self.assertEqual(projected.grad[0, 13].item(), 0)  # up above its upper bound
                    self.assertEqual(projected.grad[0, 7].item(), 0)  # up below its lower bound
                    self.assertEqual(set(dict(model.experts.named_parameters())), {"gate_up_proj", "down_proj"})
                    del source
                    gc.collect()
                    self.assertIsNone(source_ref())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_deepseek_fusion_and_baseline_keep_existing_guard(self) -> None:
        """
        Feature: Localized V4.1 QAT support.
        Description: Non-fused QAT support does not broaden the public baseline or fused contracts.
        Expectation: Both unsupported replacements still reject the custom activation.
        """
        for fused in (False, True):
            with self.subTest(fused=fused):
                raw = deepseek_qat_config(fused=fused).to_dict()
                if not fused:
                    del raw["plan_overrides"][0]["replace_module"]["low_precision_dtype_scheme"]
                with self.assertRaisesRegex(ValueError, "custom _apply_gate"):
                    install_family(deepseek_experts_model(), resolve_config(raw))
