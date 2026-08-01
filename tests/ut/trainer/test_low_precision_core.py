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
"""CPU-only contracts for Dense MXFP8 model conversion."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from hyper_models.components.training.low_precision import (
    LowPrecisionConfig,
    LowPrecisionConversionError,
    NpuQuantLinear,
    apply_low_precision,
)


class _CustomLinear(nn.Linear):
    """A Linear subclass whose forward contract cannot be retained generically."""


class TestLowPrecisionCore(unittest.TestCase):
    """Verify conversion behavior before a real NPU runtime is required."""

    def test_converter_replaces_selected_exact_linear(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=False))

        converted = apply_low_precision(
            model,
            LowPrecisionConfig(enabled=True, include_fqns=["0"]),
        )

        self.assertIs(converted, model)
        self.assertIsInstance(model[0], NpuQuantLinear)

    def test_converter_rejects_selected_linear_subclass(self):
        model = nn.Sequential(_CustomLinear(32, 32, bias=False))

        with self.assertRaisesRegex(
            LowPrecisionConversionError,
            "linear-subclass-is-not-supported",
        ):
            apply_low_precision(
                model,
                LowPrecisionConfig(enabled=True, include_fqns=["0"]),
            )

    def test_converted_linear_preserves_parameter_objects(self):
        linear = nn.Linear(32, 32, bias=False)
        model = nn.Sequential(linear)

        apply_low_precision(
            model,
            LowPrecisionConfig(enabled=True, include_fqns=["0"]),
        )

        self.assertIs(model[0].weight, linear.weight)

    def test_converted_linear_calls_native_quantized_function(self):
        model = nn.Sequential(nn.Linear(32, 32, bias=False))
        apply_low_precision(
            model,
            LowPrecisionConfig(enabled=True, include_fqns=["0"]),
        )

        with patch(
            "hyper_models.components.training.low_precision.modules.linear.npu_quant_linear",
            return_value=torch.zeros(1, 32),
        ) as quant_linear:
            output = model(torch.ones(1, 32))

        self.assertEqual(output.shape, (1, 32))
        quant_linear.assert_called_once()
        inputs, weight, quantizer = quant_linear.call_args.args
        self.assertTrue(torch.equal(inputs, torch.ones(1, 32)))
        self.assertIs(weight, model[0].weight)
        self.assertIs(quantizer, model[0].quantizer)


if __name__ == "__main__":
    unittest.main()
