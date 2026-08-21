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
"""NPU alignment tests for MoE token permutation and unpermutation."""

# Importing the Hyper interfaces after importorskip keeps CPU-only collection usable.
# pylint: disable=wrong-import-position

import pytest
import torch

pytest.importorskip("torch_npu")

from hyper_models.ops import moe_token_permute, moe_token_unpermute
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is required")


def _tolerance(dtype: torch.dtype) -> float:
    """Return the tolerance used by NPU token routing operations."""
    return 1e-3 if dtype == torch.float32 else 1e-2


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_token_permute_unpermute_matches_torch_oracle(
    dtype: torch.dtype,
) -> None:
    """Token routing must match sorting, weighting, and route accumulation."""
    torch.manual_seed(5)
    expected_tokens = torch.randn(5, 8, device="npu", dtype=dtype, requires_grad=True)
    actual_tokens = expected_tokens.detach().clone().requires_grad_(True)
    indices = torch.tensor(
        [[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]],
        device="npu",
        dtype=torch.long,
    )
    routing_weights = torch.softmax(
        torch.randn(5, 2, device="npu", dtype=torch.float32),
        dim=-1,
    ).to(dtype)
    expected_weights = routing_weights.detach().clone().requires_grad_(True)
    actual_weights = routing_weights.detach().clone().requires_grad_(True)

    flat_indices = indices.reshape(-1)
    _, permutation = torch.sort(flat_indices)
    token_indices = permutation // indices.shape[-1]
    expected_permuted = expected_tokens.index_select(0, token_indices)
    expected_probabilities = expected_weights.reshape(-1).index_select(
        0,
        permutation,
    )
    weighted = expected_permuted * expected_probabilities.unsqueeze(-1)
    expected = torch.zeros_like(expected_tokens)
    expected.index_add_(0, token_indices, weighted)

    actual_permuted, sorted_indices = moe_token_permute(actual_tokens, indices)
    actual = moe_token_unpermute(
        actual_permuted,
        sorted_indices,
        actual_weights,
    )
    tolerance = _tolerance(dtype)
    torch.testing.assert_close(
        actual_permuted,
        expected_permuted,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )

    output_gradient = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(
        expected,
        (expected_tokens, expected_weights),
        output_gradient,
    )
    actual_grads = torch.autograd.grad(
        actual,
        (actual_tokens, actual_weights),
        output_gradient,
    )
    for actual_gradient, expected_gradient in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=tolerance,
            atol=tolerance,
        )
