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

"""Native kda forward and gradient precision contracts."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestKdaPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("kda", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_native_is_independent_of_model_integration(self) -> None:
        """The standalone contract must work before module and model migration lands."""
        with patch.dict("sys.modules", {"hyper_parallel.components.functional.kimi_delta_attention": None}):
            for case in precision_cases("kda", torch.float32):
                with self.subTest(case=case.name):
                    check_case(case)

    def test_native_grouped_heads_and_unbounded_gate(self) -> None:
        """Retain grouped-value, explicit scaling and negative-softplus gate semantics."""
        case = precision_cases("kda", torch.float32)[0]
        query, key, *rest = case.args
        for lower_bound in (-3.0, None):
            with self.subTest(lower_bound=lower_bound):
                check_case(replace(case, args=(query[:, :, :1], key[:, :, :1], *rest),
                                   kwargs={**case.kwargs, "lower_bound": lower_bound, "scale": 0.25}))
