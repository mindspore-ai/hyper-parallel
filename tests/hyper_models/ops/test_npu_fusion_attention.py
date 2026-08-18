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
"""NPU regression tests for packed fusion attention."""

# Importing the interface after importorskip keeps CPU-only test collection usable.
# pylint: disable=wrong-import-position

from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from hyper_models.ops import npu_fusion_attention_forward
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
def test_packed_attention_matches_individual_sequences() -> None:
    """Packed actual/cu length aliases must preserve segment boundaries."""
    torch.manual_seed(2)
    module = SimpleNamespace(is_causal=True)
    query = torch.randn(
        1, 4, 7, 8, device="npu", dtype=torch.bfloat16, requires_grad=True
    )
    key = torch.randn(
        1, 2, 7, 8, device="npu", dtype=torch.bfloat16, requires_grad=True
    )
    value = torch.randn(
        1, 2, 7, 8, device="npu", dtype=torch.bfloat16, requires_grad=True
    )

    packed, _ = npu_fusion_attention_forward(
        module, query, key, value, None, actual_seq_len=[3, 7]
    )
    individual = []
    for start, end in ((0, 3), (3, 7)):
        output, _ = npu_fusion_attention_forward(
            module,
            query[:, :, start:end],
            key[:, :, start:end],
            value[:, :, start:end],
            None,
        )
        individual.append(output)
    expected = torch.cat(individual, dim=1)
    torch.testing.assert_close(packed, expected, rtol=0.0, atol=0.0)

    cumulative = torch.tensor([0, 3, 7], device="npu", dtype=torch.int32)
    from_cu_lengths, _ = npu_fusion_attention_forward(
        module,
        query,
        key,
        value,
        None,
        cu_seq_lens_q=cumulative,
        cu_seq_lens_k=cumulative,
    )
    torch.testing.assert_close(from_cu_lengths, packed, rtol=0.0, atol=0.0)

    packed.float().sum().backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    )
