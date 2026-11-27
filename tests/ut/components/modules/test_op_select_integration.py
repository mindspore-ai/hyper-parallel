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
"""CPU module acceptance for basic and expert OpSelect call sites."""

import unittest

import torch

from hyper_parallel.components.ops import ops
from tests.common.op_module_precision import check_module_case, module_precision_cases, position_precision_cases
from tests.common.op_precision import check_case


class TestOpSelectIntegration(unittest.TestCase):
    """Compare production members to independent formulas through backward and update."""

    def setUp(self) -> None:
        """Restore process configuration even when a precision case fails."""
        self.previous = ops.get_backends()
        self.addCleanup(ops.set_backends, self.previous)

    def test_native_module_precision(self) -> None:
        """Cover cast policies, clamp policies, empty experts and mixed-precision coefficients."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in module_precision_cases(dtype):
                with self.subTest(dtype=dtype, case=case.name):
                    check_module_case(case)

    def test_recompute_precision(self) -> None:
        """Non-reentrant recomputation preserves all input/parameter gradients and updates."""
        for case in module_precision_cases(torch.float32):
            with self.subTest(case=case.name):
                check_module_case(case, recompute=True)

    def test_compiled_pointwise_members(self) -> None:
        """Dynamo/AOTAutograd captures selectable norm and activation calls with backward."""
        for case in module_precision_cases(torch.float32):
            if case.name in ("rms_False_after_weight", "mlp_0.0_input"):
                with self.subTest(case=case.name):
                    check_module_case(case, compile_module=True)

    def test_explicit_unavailable_backend_is_not_bypassed(self) -> None:
        """A module must honor explicit selection even on CPU."""
        case = module_precision_cases(torch.float32)[0]
        with self.assertRaisesRegex((ValueError, RuntimeError), "torch_npu"):
            check_module_case(case, selections={"rms_norm": "torch_npu"})
        self.assertEqual(ops.get_backends(), self.previous)

    def test_position_adapters(self) -> None:
        """GQA/MLA use the selected RoPE for both layouts, partial widths and identity."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in position_precision_cases(dtype):
                with self.subTest(dtype=dtype, case=case.name):
                    check_case(case)
