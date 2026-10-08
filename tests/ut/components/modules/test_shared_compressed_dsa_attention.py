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
"""Check scalar-sink forward/backward against an independent dense reference."""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    _NpuSparseAttentionWithScalarSink,
    _reference_sparse_attention,
)


def _gather_inputs(query, key, indices, query_lengths, _key_lengths):
    """Expand per-batch sparse addresses for a CPU operator-contract mock."""
    batch = query_lengths.numel()
    length = query.shape[0] // batch
    capacity = key.shape[0] // batch
    positions = indices[:, 0].long()
    if bool((positions < 0).any()):
        raise AssertionError(f"Expected nonnegative sparse addresses, got minimum={positions.min()}")
    if capacity <= positions.shape[-1]:
        raise AssertionError(f"Expected KV capacity > selected width, got {capacity} <= {positions.shape[-1]}")
    offsets = torch.arange(batch).repeat_interleave(length).view(-1, 1) * capacity
    positions = positions + offsets
    return positions, key[positions, 0]


def _forward(query, key, value, indices, scale, **kwargs):
    """Reference the sparse operator, including auxiliary score coordinates."""
    positions, selected_key = _gather_inputs(
        query, key, indices, kwargs["actual_seq_lengths_query"], kwargs["actual_seq_lengths_kv"],
    )
    logits = torch.einsum("thd,tkd->thk", query, selected_key)
    logits += torch.einsum("thr,tkr->thk", kwargs["query_rope"], kwargs["key_rope"][positions, 0])
    logits *= scale
    maximum = logits.amax(-1)
    total = (logits - maximum.unsqueeze(-1)).exp().sum(-1)
    output = torch.einsum("thk,tkd->thd", logits.softmax(-1), value[positions, 0])
    return output, maximum.unsqueeze(0), total.unsqueeze(0)


def _backward(query, key, value, indices, grad, output, maximum, total, scale, **kwargs):
    """Honor externally supplied normalizers instead of recomputing softmax."""
    positions, selected_key = _gather_inputs(
        query, key, indices, kwargs["actual_seq_qlen"], kwargs["actual_seq_kvlen"],
    )
    logits = torch.einsum("thd,tkd->thk", query, selected_key)
    logits += torch.einsum("thr,tkr->thk", kwargs["query_rope"], kwargs["key_rope"][positions, 0])
    probabilities = (logits * scale - maximum.squeeze(0).unsqueeze(-1)).exp()
    probabilities /= total.squeeze(0).unsqueeze(-1)
    grad_scores = probabilities * (
        torch.einsum("thd,tkd->thk", grad, value[positions, 0])
        - (grad * output).sum(-1, keepdim=True)
    )
    grad_query = torch.einsum("thk,tkd->thd", grad_scores, selected_key) * scale
    grad_key_slots = torch.einsum("thk,thd->tkd", grad_scores, query) * scale
    grad_value_slots = torch.einsum("thk,thd->tkd", probabilities, grad)
    grad_key, grad_value = torch.zeros_like(key), torch.zeros_like(value)
    grad_key[:, 0].index_add_(0, positions.flatten(), grad_key_slots.reshape(-1, key.shape[-1]))
    grad_value[:, 0].index_add_(0, positions.flatten(), grad_value_slots.reshape(-1, value.shape[-1]))
    return grad_query, grad_key, grad_value, None, None


class TestScalarSinkBridge(unittest.TestCase):
    """Exercise cancellation, empty rows, mixed padding and arbitrary scale."""

    def _compare(self, query, key, sinks, indices, scale):
        reference_inputs = [tensor.clone().requires_grad_() for tensor in (query, key, sinks)]
        actual_inputs = [tensor.clone().requires_grad_() for tensor in (query, key, sinks)]
        expected = _reference_sparse_attention(*reference_inputs[:2], indices, reference_inputs[2], scale)
        operators = types.SimpleNamespace(
            npu_sparse_flash_attention_enhance=_forward,
            npu_sparse_flash_attention_grad_enhance=_backward,
        )
        with patch.dict(sys.modules, {"omni_training_custom_ops": types.ModuleType("omni_training_custom_ops")}):
            with patch.object(torch.ops, "custom", operators):
                actual = _NpuSparseAttentionWithScalarSink.apply(
                    *actual_inputs[:2], indices, actual_inputs[2], 4, scale,
                )
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                actual.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        for value, reference in zip(actual_inputs, reference_inputs):
            torch.testing.assert_close(value.grad, reference.grad, rtol=2e-5, atol=2e-5)

    def test_negative_logits_and_empty_selection(self):
        """Padding cannot erase valid mass, and empty rows have zero gradients."""
        torch.manual_seed(17)
        query = torch.zeros(1, 2, 3, 8)
        key = torch.zeros(1, 1, 4, 8)
        query[..., 0] = 1
        key[..., 0], key[..., 1] = -100, 1
        indices = torch.tensor([[[0, 1, 2, 3], [-1, 0, -1, -1], [-1, -1, -1, -1]]])
        self._compare(query, key, torch.full((2,), -100.0), indices, 1.0)

    def test_batch_padding_and_nonunit_scale(self):
        """The bridge preserves batch-local indices and per-head scalar sinks."""
        torch.manual_seed(19)
        query = torch.randn(2, 3, 4, 8)
        key = torch.randn(2, 1, 5, 8)
        indices = torch.tensor([
            [[0, 2, -1, 4], [-1, 1, 3, -1], [0, -1, -1, -1], [-1, -1, -1, -1]],
            [[4, 1, -1, 2], [0, 2, 4, -1], [-1, -1, 3, -1], [0, 1, 2, 3]],
        ])
        self._compare(query, key, torch.tensor([-20.0, 0.37, 20.0]), indices, 8 ** -0.5)
