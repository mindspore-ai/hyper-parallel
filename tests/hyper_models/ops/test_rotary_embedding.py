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
"""NPU tests for the public Transformers-compatible RoPE functions."""

# Importing the interfaces after importorskip keeps CPU-only test collection usable.
# pylint: disable=wrong-import-position

import pytest
import torch

pytest.importorskip("torch_npu")

from hyper_models.ops import apply_rotary_pos_emb, apply_rotary_pos_emb_interleave
from tests.common.mark_utils import arg_mark


pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU is required"
)


def _rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotary_half_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _rotary_interleave_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    query_even, query_odd = query[..., 0::2], query[..., 1::2]
    key_even, key_odd = key[..., 0::2], key[..., 1::2]
    return (
        torch.cat(
            (query_even * cos - query_odd * sin, query_odd * cos + query_even * sin),
            dim=-1,
        ),
        torch.cat(
            (key_even * cos - key_odd * sin, key_odd * cos + key_even * sin),
            dim=-1,
        ),
    )


@pytest.mark.parametrize(
    ("dtype", "unsqueeze_dim"),
    [(torch.float32, 1), (torch.bfloat16, 2)],
)
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_partial_interleaved_rotary_embedding_matches_pairwise_formula(
    dtype: torch.dtype,
    unsqueeze_dim: int,
) -> None:
    """Partial interleaved RoPE must rotate only its leading dimensions."""
    torch.manual_seed(6)
    query = torch.randn(2, 3, 5, 16, device="npu", dtype=dtype)
    key = torch.randn(2, 1, 5, 16, device="npu", dtype=dtype)
    if unsqueeze_dim == 2:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query.requires_grad_(True)
    key.requires_grad_(True)
    frequencies = torch.randn(2, 5, 4, device="npu", dtype=torch.float32)
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).to(dtype)
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).to(dtype)

    actual_query, actual_key = apply_rotary_pos_emb_interleave(
        query, key, cos, sin, unsqueeze_dim=unsqueeze_dim
    )
    query_rotated, key_rotated = _rotary_interleave_oracle(
        query[..., :8], key[..., :8], cos, sin, unsqueeze_dim
    )
    expected_query = torch.cat((query_rotated, query[..., 8:]), dim=-1)
    expected_key = torch.cat((key_rotated, key[..., 8:]), dim=-1)

    tolerance = 0.0 if dtype == torch.float32 else 0.016
    torch.testing.assert_close(
        actual_query, expected_query, rtol=0.0, atol=tolerance
    )
    torch.testing.assert_close(
        actual_key, expected_key, rtol=0.0, atol=tolerance
    )

    query_grad = torch.randn_like(actual_query)
    key_grad = torch.randn_like(actual_key)
    actual_grads = torch.autograd.grad(
        (actual_query * query_grad).sum() + (actual_key * key_grad).sum(),
        (query, key),
        retain_graph=True,
    )
    expected_grads = torch.autograd.grad(
        (expected_query * query_grad).sum() + (expected_key * key_grad).sum(),
        (query, key),
    )
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=tolerance)


@pytest.mark.parametrize(
    ("unsqueeze_dim", "interleaved"),
    [(1, False), (2, True)],
)
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_rotary_embedding_accepts_sequence_only_frequencies(
    unsqueeze_dim: int,
    interleaved: bool,
) -> None:
    """A ``[sequence, rotary_dim]`` table must broadcast over the batch."""
    torch.manual_seed(8)
    query = torch.randn(2, 3, 5, 8, device="npu")
    key = torch.randn(2, 1, 5, 8, device="npu")
    if unsqueeze_dim == 2:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    frequencies = torch.randn(5, 4, device="npu")
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1)
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1)
    function = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb

    actual = function(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
    expected = function(
        query,
        key,
        cos.unsqueeze(0),
        sin.unsqueeze(0),
        unsqueeze_dim=unsqueeze_dim,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize(
    ("dtype", "unsqueeze_dim", "interleaved", "frequency_batch"),
    [
        (torch.float32, 1, False, 2),
        (torch.bfloat16, 2, False, 1),
        (torch.float32, 2, True, 2),
        (torch.bfloat16, 1, True, 1),
    ],
)
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_rotary_embedding_matches_transformers_contract(
    dtype: torch.dtype,
    unsqueeze_dim: int,
    interleaved: bool,
    frequency_batch: int,
) -> None:
    """Check forward and input gradients against the Transformers formulas."""
    torch.manual_seed(1)
    query = torch.randn(2, 3, 5, 8, device="npu", dtype=dtype)
    key = torch.randn(2, 1, 5, 8, device="npu", dtype=dtype)
    if unsqueeze_dim == 2:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query.requires_grad_(True)
    key.requires_grad_(True)
    frequencies = torch.randn(
        frequency_batch, 5, 4, device="npu", dtype=torch.float32
    )
    cos = torch.cat((frequencies.cos(), frequencies.cos()), dim=-1).to(dtype)
    sin = torch.cat((frequencies.sin(), frequencies.sin()), dim=-1).to(dtype)

    function = apply_rotary_pos_emb_interleave if interleaved else apply_rotary_pos_emb
    oracle = _rotary_interleave_oracle if interleaved else _rotary_half_oracle
    if interleaved:
        actual_query, actual_key = function(
            query, key, cos, sin, unsqueeze_dim=unsqueeze_dim
        )
    else:
        # Transformers exposes unsqueeze_dim as the fifth positional argument.
        actual_query, actual_key = function(query, key, cos, sin, unsqueeze_dim)
    expected_query, expected_key = oracle(
        query, key, cos, sin, unsqueeze_dim
    )

    tolerance = 0.0 if dtype == torch.float32 else 0.016
    torch.testing.assert_close(actual_query, expected_query, rtol=0.0, atol=tolerance)
    torch.testing.assert_close(actual_key, expected_key, rtol=0.0, atol=tolerance)

    query_grad = torch.randn_like(actual_query)
    key_grad = torch.randn_like(actual_key)
    actual_grads = torch.autograd.grad(
        (actual_query * query_grad).sum() + (actual_key * key_grad).sum(),
        (query, key),
        retain_graph=True,
    )
    expected_grads = torch.autograd.grad(
        (expected_query * query_grad).sum() + (expected_key * key_grad).sum(),
        (query, key),
    )
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=tolerance)
