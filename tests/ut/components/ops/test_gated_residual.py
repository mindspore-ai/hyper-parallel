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
"""Necessary gated-residual precision and parameter-ownership checks."""
import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.components.modules import GatedResidual, GatedResidualModule
from hyper_parallel.components.ops import ops
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestGatedResidual(unittest.TestCase):
    """Compare both read modes and retain loaded HF-style parameter identities."""

    def test_native_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for case in precision_cases("gated_residual", dtype):
                with self.subTest(dtype=dtype, case=case.name):
                    check_case(case)

    def test_module_parameters_and_residual(self):
        config = SimpleNamespace(hc_count=4, hidden_size=8, hc_lowrank=8, rms_norm_eps=1e-6)
        source = GatedResidual(config=config)
        replacement = GatedResidualModule(module=source)
        self.assertEqual(list(source.state_dict()), list(replacement.state_dict()))
        for name, parameter in source.named_parameters():
            self.assertIs(parameter, dict(replacement.named_parameters())[name])
        previous = ops.get_backends()
        try:
            ops.set_backends({"gated_residual": "native"})
            x = torch.randn(2, 7, 32, requires_grad=True)
            mixed, residual, inject = replacement(x)
            self.assertIs(residual, x)
            replacement.combine(mixed, residual, inject).square().mean().backward()
            for parameter in source.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        finally:
            ops.set_backends(previous)
