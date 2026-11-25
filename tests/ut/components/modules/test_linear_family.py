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
"""Dense Family selection through configuration and the actual replacement engine."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.components.modules.linear import (
    Linear,
    LowPrecisionLinear,
    replace_hif8_linear,
    replace_mxfp8_linear,
)
from hyper_parallel.components.quantization import LowPrecisionLinear as QuantizationLinear
from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.functional.hifloat8_linear_func import HiFloat8LinearStrategy
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import MXFP8LinearStrategy
from hyper_parallel.components.quantization.modules import LowPrecisionLinear as QuantizationModuleLinear
from hyper_parallel.components.quantization.modules.replacement import replace_linear
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import replace_override_path, resolve_config
from tests.common.linear_family import linear_config
from tests.common.qat_family import install_family


class TestLinearFamily(unittest.TestCase):
    """Only isolate the runtime probe; selection and parameter reuse remain real."""

    def test_baseline_preserves_computation_and_parameters(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: The unquantized family retains bias, mode, values and autograd behavior.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        source = nn.Linear(128, 256).eval()
        reference = copy.deepcopy(source)
        model = install_family(nn.Sequential(source), linear_config())
        self.assertIs(type(model[0]), Linear)
        self.assertIs(model[0].weight, source.weight)
        self.assertIs(model[0].bias, source.bias)
        self.assertFalse(model[0].training)
        inputs = torch.randn(4, 128)
        output = model(inputs)
        expected = reference(inputs)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(model[0].weight.grad, reference.weight.grad, rtol=0, atol=0)

    @patch("hyper_parallel.components.quantization.functional.linear_strategy_factory.validate_npu_runtime")
    def test_mxfp8_selected_without_global_enable_or_when(self, probe) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: An explicit scheme selects the existing Dense adapter with shared weights.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        source = nn.Linear(128, 256).eval()
        model = install_family(nn.Sequential(source), linear_config(quantized=True))
        self.assertIs(type(model[0]), LowPrecisionLinear)
        self.assertIs(type(model[0].strategy), MXFP8LinearStrategy)
        self.assertIs(model[0].weight, source.weight)
        self.assertIs(model[0].bias, source.bias)
        self.assertFalse(model[0].training)
        probe.assert_called_once()
        with self.assertRaisesRegex(TypeError, "must be exact nn.Linear"):
            replace_mxfp8_linear(module=type("SubLinear", (nn.Linear,), {})(128, 256), module_fqn="0", context={})
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            replace_mxfp8_linear(module=nn.Linear(30, 64), module_fqn="0", context={})
        with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
            replace_mxfp8_linear(module=nn.Linear(128, 256), module_fqn="0", context={"pp": True})

    @patch("hyper_parallel.components.quantization.functional.linear_strategy_factory.validate_hifloat8_runtime")
    def test_hif8_selected_without_tile_alignment(self, probe) -> None:
        """
        Feature: Unified Linear Module Family.
        Description: An explicit hif8 scheme reuses the shared Dense strategy builder without MX block-32 alignment.
        Expectation: Selection, non-aligned shapes, and parameter retention match the stated contract.
        """
        raw = linear_config(quantized=True).to_dict()
        raw["training"]["low_precision"]["dtype_schemes"]["dense"] = {
            "weight_format": "hif8",
            "act_format": "hif8",
        }
        source = nn.Linear(17, 13).eval()
        model = install_family(nn.Sequential(source), resolve_config(raw))
        self.assertIs(type(model[0]), LowPrecisionLinear)
        self.assertIs(type(model[0].strategy), HiFloat8LinearStrategy)
        self.assertIs(model[0].weight, source.weight)
        self.assertIs(model[0].bias, source.bias)
        self.assertFalse(model[0].training)
        probe.assert_called_once()
        converted = replace_hif8_linear(module=nn.Linear(19, 11), module_fqn="0", context={})
        self.assertIsInstance(converted.strategy, HiFloat8LinearStrategy)

    @patch("hyper_parallel.components.quantization.functional.linear_strategy_factory.validate_npu_runtime")
    @patch("hyper_parallel.components.quantization.functional.linear_strategy_factory.validate_hifloat8_runtime")
    def test_policy_and_family_preserve_the_same_linear_contract(self, hif8_probe, mxfp8_probe) -> None:
        """
        Feature: Shared policy and Family Linear replacement.
        Description: Both entry points use one shell while preserving loaded parameters, mode and registration.
        Expectation: MXFP8 and non-aligned HiF8 retain the same state without initializing replacement parameters.
        """
        cases = (("mxfp8", 64, 32, MXFP8LinearStrategy), ("hif8", 17, 13, HiFloat8LinearStrategy))
        for format_name, in_features, out_features, strategy_type in cases:
            for bias in (False, True):
                for training in (False, True):
                    with self.subTest(format_name=format_name, bias=bias, training=training):
                        source = nn.Linear(in_features, out_features, bias=bias).train(training)
                        source.weight.requires_grad_(False)
                        scheme = LowPrecisionDtypeScheme(weight_format=format_name, act_format=format_name)
                        raw = linear_config(quantized=True).to_dict()
                        raw["training"]["low_precision"]["dtype_schemes"]["dense"] = {
                            "weight_format": format_name, "act_format": format_name,
                        }
                        rng_state = torch.random.get_rng_state()
                        family = install_family(nn.Sequential(source), resolve_config(raw))[0]
                        policy = replace_linear(module=source, module_fqn="0", context={"low_precision": scheme})

                        for converted in (family, policy):
                            self.assertIs(type(converted), LowPrecisionLinear)
                            self.assertIs(type(converted.strategy), strategy_type)
                            self.assertIs(converted.weight, source.weight)
                            self.assertIs(converted.bias, source.bias)
                            self.assertEqual(converted.training, training)
                            self.assertEqual(converted.in_features, in_features)
                            self.assertEqual(converted.out_features, out_features)
                            self.assertEqual(tuple(converted.state_dict()), tuple(source.state_dict()))
                            self.assertFalse(converted.weight.requires_grad)
                        torch.testing.assert_close(torch.random.get_rng_state(), rng_state, rtol=0, atol=0)
        self.assertEqual(mxfp8_probe.call_count, 8)
        self.assertEqual(hif8_probe.call_count, 8)

    def test_quantization_exports_use_the_canonical_linear(self) -> None:
        """
        Feature: Linear compatibility exports.
        Description: Quantization package imports resolve to the moved module implementation.
        Expectation: Existing imports and the Family return the same LowPrecisionLinear class.
        """
        self.assertIs(QuantizationLinear, LowPrecisionLinear)
        self.assertIs(QuantizationModuleLinear, LowPrecisionLinear)

    def test_dense_w4a8_is_not_silently_treated_as_grouped(self) -> None:
        """
        Feature: Scheme-selected Module Family.
        Description: Grouped-only strategies cannot be selected for an ordinary Dense module.
        Expectation: Selection, validation and parameter behavior match the stated contract.
        """
        raw = linear_config(quantized=True).to_dict()
        raw["training"]["low_precision"]["dtype_schemes"]["dense"]["weight_format"] = "mxfp4"
        config = resolve_config(raw)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            entries_to_module_replacements(config.plan_overrides,
                                           low_precision_config=config.training.low_precision)

    def test_dense_fusion_requires_a_registered_implementation(self) -> None:
        """
        Feature: Format-independent fusion selector.
        Description: Dense MXFP8 has no fused variant registered.
        Expectation: The unsupported request fails during planning without device operations.
        """
        config = linear_config(quantized=True)
        config.plan_overrides[0] = replace_override_path(
            config.plan_overrides[0], ["replace", "implementation", "fused"], True, path="plan_overrides[0]")
        with self.assertRaisesRegex(ValueError, "unsupported selector combination"):
            entries_to_module_replacements(config.plan_overrides,
                                           low_precision_config=config.training.low_precision)
