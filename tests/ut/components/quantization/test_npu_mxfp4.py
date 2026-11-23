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
"""CPU-only operator-call contract for Indexer MXFP4 QDQ."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hyper_parallel.components.quantization.ops.npu_mxfp4 import (
    LowPrecisionCapabilityError,
    _native_qdq,
    _quant_dequant,
)


class TestNpuMxfp4(unittest.TestCase):
    """OCP scale selection and the no-CPU-fallback boundary."""

    def test_operator_uses_ocp_scale_algorithm(self):
        """Q/K requests OCP (scale_alg=0), not an FP4 alternative."""
        values = torch.ones(2, 32, dtype=torch.bfloat16)
        packed = torch.empty(2, 16, dtype=torch.uint8)
        scale = torch.empty(2, 1, 2, dtype=torch.uint8)
        expected = torch.zeros_like(values)
        quant = Mock(return_value=(packed, scale))
        anti_quant = Mock(return_value=expected)
        runtime = SimpleNamespace(
            float4_e2m1fn_x2=296,
            npu_dynamic_mx_quant=quant,
            npu_anti_mx_quant=anti_quant,
        )

        output = _quant_dequant(values, runtime)

        self.assertIs(output, expected)
        self.assertEqual(quant.call_args.kwargs["scale_alg"], 0)
        self.assertEqual(quant.call_args.kwargs["block_size"], 32)
        self.assertEqual(quant.call_args.kwargs["round_mode"], "rint")
        self.assertEqual(anti_quant.call_args.kwargs["dst_type"], 15)
        self.assertIs(anti_quant.call_args.args[0], packed)
        self.assertIs(anti_quant.call_args.args[1], scale)

    def test_cpu_input_fails_instead_of_switching_implementation(self):
        """The production policy cannot quietly choose a CPU approximation."""
        with self.assertRaises(LowPrecisionCapabilityError):
            _native_qdq(torch.zeros(1, 32))

    def test_missing_native_op_reports_capability_error(self):
        """An older torch_npu installation fails with an actionable error."""
        runtime = SimpleNamespace(float4_e2m1fn_x2=296, npu_dynamic_mx_quant=Mock())
        with self.assertRaisesRegex(LowPrecisionCapabilityError, "npu_anti_mx_quant"):
            _quant_dequant(torch.zeros(1, 32), runtime)


if __name__ == "__main__":
    unittest.main()
