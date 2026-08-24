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
"""NPU alignment tests for fused SwiGLU."""

# Importing the Hyper interface after importorskip keeps CPU-only collection usable.
# pylint: disable=wrong-import-position

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")

from hyper_parallel.auto_models.ops import swiglu
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _tolerance(dtype: torch.dtype) -> float:
    """Return the tolerance used by fused NPU SwiGLU."""
    return 1e-3 if dtype == torch.float32 else 1e-2


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_matches_formula_and_backward(dtype: torch.dtype) -> None:
    """Fused SwiGLU must match SiLU gating outputs and input gradients."""
    torch.manual_seed(7)
    expected_input = torch.randn(11, 16, device="npu", dtype=dtype, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    gate, up = expected_input.chunk(2, dim=-1)
    expected = F.silu(gate) * up
    actual = swiglu(actual_input, dim=-1)
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )

    output_gradient = torch.randn_like(expected)
    expected_gradient = torch.autograd.grad(
        expected,
        expected_input,
        output_gradient,
    )[0]
    actual_gradient = torch.autograd.grad(
        actual,
        actual_input,
        output_gradient,
    )[0]
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=tolerance,
        atol=tolerance,
    )
