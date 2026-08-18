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
"""NPU alignment tests for high-performance RMSNorm modules."""

# Importing Hyper interfaces after importorskip keeps CPU-only collection usable.
# pylint: disable=wrong-import-position

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
)

pytest.importorskip("torch_npu")

from hyper_models.modules import OffsetRMSNorm
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU is required"
)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen3_5_moe_rms_norm_matches_transformers(dtype: torch.dtype) -> None:
    """Match Qwen3.5 MoE forward, input gradient, and offset-weight gradient."""
    torch.manual_seed(7)
    source = modeling.Qwen3_5MoeRMSNorm(256, eps=1e-6).to(
        device="npu", dtype=dtype
    )
    replacement = OffsetRMSNorm(module=source).to(device="npu", dtype=dtype)
    torch.testing.assert_close(
        replacement.weight,
        source.weight + 1.0,
        rtol=0.0,
        atol=0.0,
    )
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.reset_parameters()
    torch.testing.assert_close(
        replacement.weight,
        torch.ones_like(replacement.weight),
        rtol=0.0,
        atol=0.0,
    )
    transform = replacement.make_transforms()[0]
    converted = transform.operations[0].convert(
        {"weight": source.weight},
        transform.source_patterns,
        transform.target_patterns,
    )
    with torch.no_grad():
        replacement.weight.copy_(converted["weight"])

    expected_input = torch.randn(
        2, 17, 256, device="npu", dtype=dtype, requires_grad=True
    )
    actual_input = expected_input.detach().clone().requires_grad_(True)
    expected = source(expected_input)
    actual = replacement(actual_input)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    expected.sum().backward()
    actual.sum().backward()
    input_grad_tolerance = 1.1920928955078125e-7 if dtype == torch.float32 else 0.0
    weight_grad_tolerance = 3.814697265625e-6 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(
        actual_input.grad,
        expected_input.grad,
        rtol=0.0,
        atol=input_grad_tolerance,
    )
    torch.testing.assert_close(
        replacement.weight.grad,
        source.weight.grad,
        rtol=0.0,
        atol=weight_grad_tolerance,
    )

    restored = transform.operations[0].reverse_op.convert(
        {"weight": replacement.weight}, ["weight"], ["weight"]
    )
    torch.testing.assert_close(
        restored["weight"], source.weight, rtol=0.0, atol=0.0
    )
