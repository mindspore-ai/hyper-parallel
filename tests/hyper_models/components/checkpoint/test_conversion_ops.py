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
"""Tests for reusable checkpoint conversion operations."""

import torch
from transformers.core_model_loading import WeightConverter

from hyper_models.components.checkpoint import (
    AddScalar,
    ConcatenateWithSections,
    DeinterleaveQKV,
    InterleaveQKV,
    Split,
)


def test_add_scalar_round_trip() -> None:
    """Convert an offset tensor and restore its original values."""
    source = torch.tensor([-0.5, 0.0, 0.5])
    operation = AddScalar(1.0)
    converted = operation.convert({"weight": source}, ["weight"], ["weight"])
    torch.testing.assert_close(
        converted["weight"], torch.tensor([0.5, 1.0, 1.5]), rtol=0.0, atol=0.0
    )
    restored = operation.reverse_op.convert(
        converted, ["weight"], ["weight"]
    )
    torch.testing.assert_close(restored["weight"], source, rtol=0.0, atol=0.0)


def test_concatenate_with_sections_roundtrip_is_exact():
    """Unequal source sections are restored exactly by the reverse operation."""
    query = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    key_value = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    operation = ConcatenateWithSections(sections=(3, 5), dim=0)

    packed = operation.convert(
        {"query": [query], "key_value": [key_value]},
        ["query", "key_value"],
        ["packed"],
    )["packed"]
    restored = operation.reverse_op.convert(
        {"packed": [packed]},
        ["packed"],
        ["query", "key_value"],
    )

    assert torch.equal(packed, torch.cat((query, key_value), dim=0))
    assert torch.equal(restored["query"], query)
    assert torch.equal(restored["key_value"], key_value)


def test_weight_converter_uses_split_for_reverse_conversion():
    """Transformers export reverses the Hyper concatenation with explicit sections."""
    converter = WeightConverter(
        source_patterns=["query", "key_value"],
        target_patterns="packed",
        operations=[ConcatenateWithSections(sections=(3, 5), dim=0)],
    )

    reverse_operation = converter.reverse_transform().operations[0]

    assert isinstance(reverse_operation, Split)
    assert reverse_operation.sections == (3, 5)
    assert reverse_operation.dim == 0


def test_interleave_qkv_roundtrip_is_exact():
    """Separate Q/K/V rows are grouped by KV head and restored exactly."""
    query = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    key = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 100
    value = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3) + 200
    operation = InterleaveQKV(
        num_key_value_heads=2,
        num_key_value_groups=2,
        query_head_dim=2,
        value_head_dim=3,
        source_is_fused=False,
    )

    grouped = operation.convert(
        {"query": query, "key": key, "value": value},
        ["query", "key", "value"],
        ["grouped"],
    )["grouped"]
    restored = operation.reverse_op.convert(
        {"grouped": grouped},
        ["grouped"],
        ["query", "key", "value"],
    )

    expected = torch.cat(
        (
            query[:4],
            key[:2],
            value[:3],
            query[4:],
            key[2:],
            value[3:],
        ),
        dim=0,
    )
    assert torch.equal(grouped, expected)
    assert torch.equal(restored["query"], query)
    assert torch.equal(restored["key"], key)
    assert torch.equal(restored["value"], value)


def test_interleave_fused_qkv_uses_fused_reverse_conversion():
    """A fused source is split for grouping and fused again on export."""
    fused = torch.arange(14, dtype=torch.float32)
    operation = InterleaveQKV(
        num_key_value_heads=2,
        num_key_value_groups=2,
        query_head_dim=2,
        value_head_dim=1,
        source_is_fused=True,
    )

    grouped = operation.convert(
        {"fused": fused}, ["fused"], ["grouped"]
    )["grouped"]
    reverse = operation.reverse_op
    restored = reverse.convert(
        {"grouped": grouped}, ["grouped"], ["fused"]
    )["fused"]

    assert isinstance(reverse, DeinterleaveQKV)
    assert torch.equal(restored, fused)
