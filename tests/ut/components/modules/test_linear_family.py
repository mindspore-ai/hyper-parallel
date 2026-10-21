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

from hyper_parallel.components.modules.linear import Linear
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import MXFP8LinearStrategy
from hyper_parallel.components.quantization.modules.linear import LowPrecisionLinear
from hyper_parallel.trainer.config.parallelism import entries_to_module_replacements
from hyper_parallel.trainer.config.resolver import resolve_config
from tests.common.mark_utils import arg_mark
from tests.common.linear_family import linear_config
from tests.common.qat_family import install_family


class TestLinearFamily(unittest.TestCase):
    """Only isolate the runtime probe; selection and parameter reuse remain real."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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

    @patch("hyper_parallel.components.modules.low_precision_linear.validate_npu_runtime")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_dense_fusion_requires_a_registered_implementation(self) -> None:
        """
        Feature: Format-independent fusion selector.
        Description: Dense MXFP8 has no fused variant registered.
        Expectation: The unsupported request fails during planning without device operations.
        """
        config = linear_config(quantized=True)
        config.plan_overrides[0].replace_module = config.plan_overrides[0].replace_module.replace(fused=True)
        with self.assertRaisesRegex(ValueError, "unsupported selector combination"):
            entries_to_module_replacements(config.plan_overrides,
                                           low_precision_config=config.training.low_precision)
