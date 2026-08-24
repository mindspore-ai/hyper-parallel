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
"""NPU alignment tests for grouped matrix multiplication."""

# Importing the Hyper interface after importorskip keeps CPU-only collection usable.
# pylint: disable=wrong-import-position

import pytest
import torch

pytest.importorskip("torch_npu")

from hyper_parallel.auto_models.ops import grouped_matmul
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _tolerance(dtype: torch.dtype) -> float:
    """Return the tolerance used by NPU grouped matrix multiplication."""
    return 1e-3 if dtype == torch.float32 else 1e-2


def _grouped_matmul_reference(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """Apply one regular matrix multiplication for every expert group."""
    outputs = []
    start = 0
    for expert, end in enumerate(group_list.tolist()):
        outputs.append(hidden_states[start:end] @ weights[expert])
        start = end
    return torch.cat(outputs, dim=0)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_matmul_matches_per_expert_forward_and_backward(
    dtype: torch.dtype,
) -> None:
    """Grouped matmul must match per-expert matmul outputs and gradients."""
    torch.manual_seed(3)
    expected_input = torch.randn(7, 8, device="npu", dtype=dtype, requires_grad=True)
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected_weight = torch.randn(3, 8, 6, device="npu", dtype=dtype, requires_grad=True)
    actual_weight = expected_weight.detach().clone().requires_grad_(True)
    group_list = torch.tensor([2, 5, 7], device="npu", dtype=torch.int64)

    expected = _grouped_matmul_reference(
        expected_input,
        expected_weight,
        group_list,
    )
    actual = grouped_matmul(
        actual_input,
        actual_weight,
        group_list=group_list,
        group_type=0,
        group_list_type=0,
    )
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )

    output_gradient = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(
        expected,
        (expected_input, expected_weight),
        output_gradient,
    )
    actual_grads = torch.autograd.grad(
        actual,
        (actual_input, actual_weight),
        output_gradient,
    )
    for actual_gradient, expected_gradient in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=tolerance,
            atol=tolerance,
        )
