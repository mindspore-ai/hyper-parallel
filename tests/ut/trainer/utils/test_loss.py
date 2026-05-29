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
"""Unit tests for ``hyper_parallel.trainer.utils.loss``.

The math under test is short but load-bearing: every micro-batch's loss
weight is computed as ``raw_loss * (micro_tokens / total_tokens) * fsdp_size``.
A regression in either factor silently corrupts loss scaling under FSDP +
gradient accumulation, so the assertions pin down both the formula and
its boundary conditions (zero total, missing labels, invalid fsdp_size).
"""
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch

from hyper_parallel.trainer.utils.loss import count_loss_token, mean_global_loss


class TestCountLossToken(unittest.TestCase):
    """``count_loss_token`` counts non-ignore tokens (``labels != -100``)."""

    def test_counts_non_padding_tokens(self):
        """Valid tokens (label != -100) are counted; -100 entries excluded."""
        labels = torch.tensor([[1, 2, -100, 3], [-100, -100, 4, 5]])
        count = count_loss_token({"labels": labels})
        self.assertEqual(count, 5, f"Expected 5 valid tokens, got {count}")

    def test_returns_zero_when_labels_missing(self):
        """Missing ``labels`` key returns 0 (do not raise)."""
        count = count_loss_token({"input_ids": torch.tensor([[1, 2, 3]])})
        self.assertEqual(count, 0, f"Expected 0 when labels missing, got {count}")

    def test_all_padding_returns_zero(self):
        """A batch made entirely of -100 returns 0 (avoids divide-by-zero upstream)."""
        labels = torch.full((2, 4), -100, dtype=torch.long)
        count = count_loss_token({"labels": labels})
        self.assertEqual(count, 0, f"Expected 0 for all-padding labels, got {count}")

    def test_returns_python_int(self):
        """Return must be a Python ``int`` so callers can reduce across ranks."""
        labels = torch.tensor([[1, 2, 3]])
        count = count_loss_token({"labels": labels})
        self.assertIsInstance(count, int, f"Return type should be int, got {type(count)}")


class TestMeanGlobalLoss(unittest.TestCase):
    """``mean_global_loss`` implements the token-weighted FSDP-corrected formula."""

    def test_token_weighted_formula(self):
        """``loss * (micro / total) * fsdp_size`` — pin the formula."""
        loss = torch.tensor(2.0)
        out = mean_global_loss(loss, micro_batch_tokens=10, total_tokens=40, fsdp_size=2)
        # 2.0 * (10/40) * 2 = 1.0
        self.assertTrue(torch.allclose(out, torch.tensor(1.0)), (
            f"Expected token-weighted loss=1.0, got {out.item()}"
        ))

    def test_zero_total_tokens_returns_loss_unchanged(self):
        """``total_tokens == 0`` must not divide by zero; return loss as-is."""
        loss = torch.tensor(3.14)
        out = mean_global_loss(loss, micro_batch_tokens=0, total_tokens=0, fsdp_size=4)
        self.assertTrue(out is loss, (
            f"Expected the same loss object back when total_tokens=0, got {out!r}"
        ))

    def test_zero_micro_tokens_zero_contribution(self):
        """A micro-batch with no valid tokens contributes 0."""
        loss = torch.tensor(5.0)
        out = mean_global_loss(loss, micro_batch_tokens=0, total_tokens=20, fsdp_size=2)
        self.assertTrue(torch.allclose(out, torch.tensor(0.0)), (
            f"Expected 0 contribution, got {out.item()}"
        ))

    def test_invalid_fsdp_size_raises(self):
        """``fsdp_size <= 0`` is a config bug and must raise loudly."""
        loss = torch.tensor(1.0)
        with self.assertRaises(ValueError):
            mean_global_loss(loss, micro_batch_tokens=4, total_tokens=8, fsdp_size=0)
        with self.assertRaises(ValueError):
            mean_global_loss(loss, micro_batch_tokens=4, total_tokens=8, fsdp_size=-1)

    def test_sum_of_micro_batches_recovers_fsdp_corrected_mean(self):
        """
        The whole point of the formula: summing per-micro-batch contributions
        with total_tokens=global_tokens reproduces the FSDP-correction factor.
        """
        # Two micro-batches with 6 and 4 valid tokens, raw losses 2.0 and 4.0.
        # Without FSDP correction: weighted_avg = (2*6 + 4*4)/10 = 2.8.
        # With fsdp_size=2 correction: total = 2.8 * 2 = 5.6.
        loss_a = torch.tensor(2.0)
        loss_b = torch.tensor(4.0)
        total = 10
        fsdp = 2
        out_a = mean_global_loss(loss_a, micro_batch_tokens=6, total_tokens=total, fsdp_size=fsdp)
        out_b = mean_global_loss(loss_b, micro_batch_tokens=4, total_tokens=total, fsdp_size=fsdp)
        summed = out_a + out_b
        self.assertTrue(torch.allclose(summed, torch.tensor(5.6)), (
            f"Expected sum of weighted micro-losses=5.6, got {summed.item()}"
        ))

    def test_preserves_dtype(self):
        """Output dtype must follow input loss dtype (bf16 stays bf16)."""
        loss = torch.tensor(1.0, dtype=torch.bfloat16)
        out = mean_global_loss(loss, micro_batch_tokens=2, total_tokens=4, fsdp_size=2)
        self.assertEqual(out.dtype, torch.bfloat16, (
            f"Expected bfloat16 preserved, got {out.dtype}"
        ))


if __name__ == "__main__":
    unittest.main()
