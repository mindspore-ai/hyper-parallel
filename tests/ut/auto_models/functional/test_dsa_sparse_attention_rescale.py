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
"""CPU contract tests for sparse and sink attention kernel dispatch."""

import importlib
import unittest
from unittest.mock import patch

import torch
from einops import rearrange

attention = importlib.import_module("hyper_parallel.components.functional.dsa_sparse_attention_rescale")


def _sparse_forward(*args, **kwargs):
    del kwargs
    return args[0] * 2, torch.zeros(1, 6, 2), torch.full((1, 6, 2), 2.0)


def _sink_forward(*args, **kwargs):
    del kwargs
    return args[0] * 3, torch.zeros(2, 2, 3, 1), torch.full((2, 2, 3, 1), 3.0)


def _sparse_grad(*args, **kwargs):
    return tuple(
        torch.full_like(tensor, index + 1)
        for index, tensor in enumerate((args[0], args[1], args[2], kwargs["query_rope"], kwargs["key_rope"]))
    )


def _sink_grad(*args, **kwargs):
    del kwargs
    return tuple(torch.full_like(tensor, (index + 1) * 10) for index, tensor in enumerate(args[:3]))


def _inputs(rope_dim=2):
    return [
        torch.randn(shape, requires_grad=True)
        for shape in ((2, 3, 2, 4), (2, 3, 1, 4), (2, 3, 2, rope_dim), (2, 3, 1, rope_dim),
                      (2, 1, 2, 4 + rope_dim), (2, 1, 2, 4 + rope_dim))
    ]


class TestDSASparseAttentionRescale(unittest.TestCase):
    """Check the six differentiable inputs without launching NPU kernels."""

    def test_forward_backward_contract(self):
        """Feature: DSA sparse attention rescaling.

        Description: Run both branches with mocked operators on CPU tensors.
        Expectation: Preserve output weights, gradient merging and kernel options.
        """
        tensors = _inputs()
        lengths = torch.tensor([3, 6], dtype=torch.int32)
        indices = torch.zeros(2, 3, 1, dtype=torch.int32)
        with patch.object(attention.torch.ops.custom, "npu_sparse_flash_attention_enhance",
                          side_effect=_sparse_forward), \
                patch.object(attention.torch_npu, "npu_fusion_attention", side_effect=_sink_forward), \
                patch.object(attention.torch.ops.custom, "npu_sparse_flash_attention_grad_enhance",
                             side_effect=_sparse_grad) as sparse, \
                patch.object(attention.torch_npu, "npu_fusion_attention_grad", side_effect=_sink_grad) as sink:
            output, _, _ = attention.dsa_sparse_attention_rescale(
                *tensors, indices, 2, 3, 2, 0.5, 0.9, lengths, lengths,
            )
            expected = torch.cat((tensors[0].detach() * 2.6, tensors[2].detach() * 1.8), dim=-1)
            torch.testing.assert_close(output, expected)
            output.sum().backward()
        for tensor, expected_grad in zip(tensors, (11, 5, 14, 5, 20, 30)):
            torch.testing.assert_close(tensor.grad, torch.full_like(tensor, expected_grad))
        self.assertEqual(sparse.call_count, 1)
        self.assertEqual(sink.call_count, 1)
        torch.testing.assert_close(sparse.call_args.args[4], torch.full((6, 2, 4), 0.4))
        torch.testing.assert_close(sink.call_args.args[3], torch.full((3, 2, 12), 0.6))
        torch.testing.assert_close(
            sparse.call_args.args[5], rearrange(output.detach()[..., :4], "b s n d -> (b s) n d"),
        )
        self.assertIs(sparse.call_args.kwargs["actual_seq_qlen"], lengths)
        self.assertIs(sparse.call_args.kwargs["actual_seq_kvlen"], lengths)
        self.assertEqual(sparse.call_args.args[8], 0.5)
        self.assertEqual(sink.call_args.kwargs["scale_value"], 0.5)
        self.assertEqual(sink.call_args.kwargs["keep_prob"], 0.9)
        self.assertEqual(sink.call_args.args[5], "SBH")
