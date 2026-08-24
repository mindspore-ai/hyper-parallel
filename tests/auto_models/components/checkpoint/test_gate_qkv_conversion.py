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
"""Tests for gated-query GQA checkpoint conversion."""

import torch

from hyper_parallel.auto_models.components.checkpoint import (
    DeinterleaveGateQKV,
    InterleaveGateQKV,
)


def test_interleave_gate_qkv_roundtrip_preserves_per_head_layout() -> None:
    """Q and gate rows alternate per head and survive a complete round trip."""
    query_gate = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3)
    key = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 100
    value = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3) + 200
    operation = InterleaveGateQKV(
        num_key_value_heads=2,
        num_key_value_groups=2,
        query_head_dim=2,
        value_head_dim=3,
    )

    converted = operation.convert(
        {"query_gate": query_gate, "key": key, "value": value},
        ["query_gate", "key", "value"],
        ["grouped"],
    )
    restored = operation.reverse_op.convert(
        converted,
        ["grouped"],
        ["query_gate", "key", "value"],
    )

    expected_query = torch.cat(
        (query_gate[0:2], query_gate[4:6], query_gate[8:10], query_gate[12:14])
    )
    expected_gate = torch.cat(
        (query_gate[2:4], query_gate[6:8], query_gate[10:12], query_gate[14:16])
    )
    expected_grouped = torch.cat(
        (
            expected_query[:4],
            key[:2],
            value[:3],
            expected_gate[:4],
            expected_query[4:],
            key[2:],
            value[3:],
            expected_gate[4:],
        )
    )
    assert isinstance(operation.reverse_op, DeinterleaveGateQKV)
    assert torch.equal(converted["grouped"], expected_grouped)
    assert torch.equal(restored["query_gate"], query_gate)
    assert torch.equal(restored["key"], key)
    assert torch.equal(restored["value"], value)


def test_interleave_gate_qkv_supports_bias_vectors() -> None:
    """The same conversion also handles one-dimensional projection biases."""
    query_gate = torch.arange(16, dtype=torch.float32)
    key = torch.arange(4, dtype=torch.float32) + 100
    value = torch.arange(6, dtype=torch.float32) + 200
    operation = InterleaveGateQKV(2, 2, 2, 3)

    converted = operation.convert(
        {"q": query_gate, "k": key, "v": value},
        ["q", "k", "v"],
        ["qkv"],
    )
    restored = operation.reverse_op.convert(
        converted, ["qkv"], ["q", "k", "v"]
    )

    assert torch.equal(restored["q"], query_gate)
    assert torch.equal(restored["k"], key)
    assert torch.equal(restored["v"], value)
