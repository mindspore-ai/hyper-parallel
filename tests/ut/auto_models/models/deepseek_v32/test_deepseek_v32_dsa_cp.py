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
"""Unit tests for DeepSeek-V3.2 DSA Context Parallel length conversion."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel import (
    _resolve_cp_sequence_lengths,
    _reduce_dense_softmax_target,
    _dense_cp_attention,
    _run_cp_dsa_kernels,
    _validate_cp_attention_mask,
)


class TestDeepseekV32DSAContextParallel(unittest.TestCase):
    """Validate hardware-independent CP metadata projection."""

    def test_dense_lengths_project_to_local_query_and_causal_prefix(self):
        """Dense batches retain one CP-local segment for every input row."""
        query, key = _resolve_cp_sequence_lengths(
            None,
            {},
            batch_size=2,
            local_seq_len=4,
            cp_size=2,
            cp_rank=1,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(query, torch.tensor([4, 8], dtype=torch.int32))
        torch.testing.assert_close(key, torch.tensor([8, 16], dtype=torch.int32))

    def test_packed_lengths_preserve_empty_rank_local_segments(self):
        """Packed documents crossing a CP shard keep aligned Q/KV numbering."""
        rank_zero_query, rank_zero_key = _resolve_cp_sequence_lengths(
            [7, 21, 32],
            {},
            batch_size=1,
            local_seq_len=16,
            cp_size=2,
            cp_rank=0,
            device=torch.device("cpu"),
        )
        rank_one_query, rank_one_key = _resolve_cp_sequence_lengths(
            [7, 21, 32],
            {},
            batch_size=1,
            local_seq_len=16,
            cp_size=2,
            cp_rank=1,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(rank_zero_query, torch.tensor([7, 16, 16], dtype=torch.int32))
        torch.testing.assert_close(rank_zero_key, torch.tensor([7, 16, 16], dtype=torch.int32))
        torch.testing.assert_close(rank_one_query, torch.tensor([0, 5, 16], dtype=torch.int32))
        torch.testing.assert_close(rank_one_key, torch.tensor([7, 21, 32], dtype=torch.int32))

    def test_incomplete_packed_boundaries_are_rejected(self):
        """Global boundaries must cover the complete unsharded token stream."""
        with self.assertRaisesRegex(ValueError, "final query sequence length"):
            _resolve_cp_sequence_lengths(
                [3, 7],
                {},
                batch_size=1,
                local_seq_len=4,
                cp_size=2,
                cp_rank=0,
                device=torch.device("cpu"),
            )

    def test_attention_mask_accepts_local_or_global_dimensions(self):
        """The CP boundary accepts Trainer-global and pre-sharded causal masks."""
        for query_length, key_length in ((4, 4), (4, 8), (8, 8)):
            with self.subTest(query_length=query_length, key_length=key_length):
                _validate_cp_attention_mask(
                    torch.zeros(2, 1, query_length, key_length),
                    batch_size=2,
                    local_seq_len=4,
                    global_seq_len=8,
                )

    def test_attention_mask_rejects_unrelated_sequence_dimensions(self):
        """Mask dimensions outside the local/global CP contracts fail early."""
        with self.assertRaisesRegex(ValueError, "query dimension"):
            _validate_cp_attention_mask(
                torch.zeros(2, 1, 6, 8),
                batch_size=2,
                local_seq_len=4,
                global_seq_len=8,
            )

    def test_dense_attention_target_reduces_fa_probabilities_over_heads(self):
        """Dense KL teacher is reduced from FA output rather than recomputed."""
        probabilities = torch.tensor(
            [[1., 0.], [0.5, 0.5], [0., 1.], [0.5, 0.5]],
        ).reshape(1, 2, 2, 2)
        target = _reduce_dense_softmax_target(
            probabilities, query_tokens=2, batch_size=1,
        )
        torch.testing.assert_close(target, torch.full((1, 2, 1, 2), 0.5))

        packed = probabilities.squeeze(0)
        packed_target = _reduce_dense_softmax_target(
            packed, query_tokens=2, batch_size=1,
        )
        torch.testing.assert_close(packed_target, target.reshape(2, 1, 2))

    def test_dense_attention_output_uses_fa_and_skips_target_without_loss(self):
        """The dense attention result uses FA; inference does not build dense targets."""
        module = SimpleNamespace()
        states = tuple(torch.zeros(1, seq, heads, dim) for seq, heads, dim in (
            (2, 2, 3), (4, 1, 3), (2, 2, 1), (4, 1, 1),
        ))
        lengths = (torch.tensor([2]), torch.tensor([4]))
        expected = torch.ones(1, 2, 2, 3)
        with patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel.npu_fusion_attention_forward",
            return_value=(expected, None),
        ) as fa:
            output, probabilities = _dense_cp_attention(
                module, states, lengths, 0.5, need_target=False,
            )

        self.assertIs(output, expected)
        self.assertIsNone(probabilities)
        self.assertEqual(fa.call_args.args[1].shape, (1, 2, 2, 4))
        self.assertEqual(fa.call_args.kwargs["sparse_mode"], 3)
        self.assertFalse(fa.call_args.kwargs["return_softmax_stats"])

    def test_dense_attention_reuses_flash_attention_softmax_output(self):
        """Training requests probabilities from the same FA invocation."""
        module = SimpleNamespace()
        states = tuple(torch.zeros(1, seq, heads, dim) for seq, heads, dim in (
            (2, 2, 3), (2, 1, 3), (2, 2, 1), (2, 1, 1),
        ))
        lengths = (torch.tensor([2]), torch.tensor([2]))
        expected_output = torch.ones(1, 2, 2, 3)
        softmax_out = torch.tensor(
            [[1., 0.], [0.5, 0.5], [0., 1.], [0.5, 0.5]],
        ).reshape(1, 2, 2, 2)
        with patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel.npu_fusion_attention_forward",
            return_value=(expected_output, None, None, None, softmax_out),
        ) as fa:
            output, target = _dense_cp_attention(module, states, lengths, 0.5, need_target=True)

        self.assertIs(output, expected_output)
        torch.testing.assert_close(target, torch.full((1, 2, 1, 2), 0.5))
        self.assertTrue(fa.call_args.kwargs["return_softmax_stats"])

    def test_dense_cp_branch_uses_dense_ops_without_sparse_indexer(self):
        """The wrapper's dense selection bypasses sparse indexer/attention calls."""
        module = SimpleNamespace(
            scaling=1.0,
            training=True,
            freeze_dsa=False,
            dsa_loss_coeff=0.1,
        )
        index_states = tuple(torch.zeros(1, 2, 2, 4) for _ in range(2)) + (torch.zeros(1, 2, 2),)
        attention_states = tuple(torch.zeros(1, 2, heads, dim) for heads, dim in (
            (2, 3), (1, 3), (2, 1), (1, 1),
        ))
        lengths = (torch.tensor([2]), torch.tensor([2]))
        dense_output = torch.ones(1, 2, 2, 3)
        dense_target = torch.ones(2, 1, 2)
        lse = torch.ones(1, 2)
        dense_loss = torch.tensor(2.0)
        with patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel."
            "dsa_dense_indexer_lse",
            return_value=lse,
        ) as dense_lse, patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel."
            "_dense_cp_attention",
            return_value=(dense_output, dense_target),
        ), patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel.dsa_dense_indexer_kl_loss",
            return_value=dense_loss,
        ) as dense_kl, patch(
            "hyper_parallel.models.deepseek_v32.adapter.distributed.context_parallel.aux_loss_auto_scale",
            return_value=dense_output,
        ) as attach_loss:
            output = _run_cp_dsa_kernels(
                module, index_states, attention_states, lengths, use_dense=True,
            )

        self.assertIs(output, dense_output)
        dense_lse.assert_called_once()
        dense_kl.assert_called_once()
        attach_loss.assert_called_once_with(dense_output, dense_loss)


if __name__ == "__main__":
    unittest.main()
