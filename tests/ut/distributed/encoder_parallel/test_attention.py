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
"""Attention fallback, padding isolation, and exact-map auditing."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.encoder_parallel.attention import (
    EncoderAttentionContext,
    activate_encoder_attention,
    current_encoder_attention_context,
    encoder_attention_checkpoint_contexts,
)
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderSegment, EncoderWorkItem
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import DeepseekV41EncoderParallelAttention
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionAttention


class _OneRankMesh:
    @staticmethod
    def size() -> int:
        """Return the degenerate Encoder-CP degree."""
        return 1


class TestEncoderAttention(unittest.TestCase):
    """Compare fallback and active attention with a one-rank Ulysses oracle."""

    def test_fallback_preserves_forward_and_gradient(self):
        """Feature: inactive DeepSeek vision attention replacement.

        Description: Compare source and replacement on the same hidden input.
        Expectation: Output and input gradients are equal.
        """
        torch.manual_seed(2)
        original = DeepseekV41VisionAttention(8, 2)
        replacement = DeepseekV41EncoderParallelAttention.from_source(original, module_fqn="vision.attn")
        self.assertEqual(original.state_dict().keys(), replacement.state_dict().keys())
        self.assertIs(original.wqkv.weight, replacement.wqkv.weight)
        source = torch.randn(4, 8, requires_grad=True)
        target = source.detach().clone().requires_grad_(True)
        cosine = torch.ones(4, 1, 2)
        sine = torch.zeros_like(cosine)
        expected = original(source, cosine, sine)
        actual = replacement(target, cosine, sine)
        self.assertTrue(torch.allclose(expected, actual))
        expected.sum().backward()
        expected_grad = source.grad.clone()
        source.grad = None
        actual.sum().backward()
        self.assertTrue(torch.allclose(expected_grad, target.grad))

    @patch("hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel.ulysses_seq_to_head",
           side_effect=lambda tensor, **kwargs: tensor)
    @patch("hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel.ulysses_head_to_seq",
           side_effect=lambda tensor, **kwargs: tensor)
    def test_segments_do_not_cross_attend(self, unused_head_to_seq, unused_seq_to_head):
        """Feature: segmented Encoder-CP attention.

        Description: Attend two packed images through the active replacement.
        Expectation: Each segment matches an independent native attention call.
        """
        torch.manual_seed(3)
        source = DeepseekV41VisionAttention(8, 2)
        replacement = DeepseekV41EncoderParallelAttention.from_source(source, module_fqn="vision.attn")
        items = (EncoderWorkItem(0, 0, 2, 1, 0, 2, 0, 2, {}),
                 EncoderWorkItem(1, 0, 2, 1, 2, 2, 2, 4, {}))
        segments = (EncoderSegment(items[0], 0, 2, 2, 0),
                    EncoderSegment(items[1], 0, 2, 2, 2))
        hidden = torch.randn(4, 8)
        cosine = torch.ones(4, 1, 2)
        sine = torch.zeros_like(cosine)
        expected = torch.cat((source(hidden[:2], cosine[:2], sine[:2]),
                              source(hidden[2:], cosine[2:], sine[2:])), dim=0)
        context = EncoderAttentionContext(
            segments, _OneRankMesh(), {name: 2 for name in ("query", "key", "value")},
            frozenset({"vision.attn"}),
        )
        with activate_encoder_attention(context):
            actual = replacement(hidden, cosine, sine)
        context.assert_exactly_once()
        self.assertIsNone(current_encoder_attention_context())
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))

    def test_audit_rejects_duplicate_call(self):
        """Feature: exact visual attention execution audit.

        Description: Call one expected block twice and omit another.
        Expectation: The audit reports an execution mismatch.
        """
        context = EncoderAttentionContext((), _OneRankMesh(), {}, frozenset({"first", "second"}))
        with activate_encoder_attention(context):
            context.record("first")
            context.record("first")
        with self.assertRaisesRegex(RuntimeError, "vision attention execution mismatch"):
            context.assert_exactly_once()

    def test_checkpoint_retains_its_own_packed_context(self):
        """Feature: encoder checkpoint metadata ownership.

        Description: Replay one microbatch while another microbatch is active.
        Expectation: Image lengths come from the saved context.
        """
        first = EncoderAttentionContext((), _OneRankMesh(), {}, frozenset(),
                                        packed_sequence_lengths=(3, 5), overlap=True)
        second = EncoderAttentionContext((), _OneRankMesh(), {}, frozenset(),
                                         packed_sequence_lengths=(7,))
        with activate_encoder_attention(first):
            _, replay = encoder_attention_checkpoint_contexts()
        with activate_encoder_attention(second):
            with replay:
                active = current_encoder_attention_context()
                self.assertEqual(active.packed_sequence_lengths, (3, 5))
                self.assertTrue(active.overlap)
                self.assertIsNot(active.seen, first.seen)
            self.assertIs(current_encoder_attention_context(), second)
        self.assertIsNone(current_encoder_attention_context())
