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
"""CPU unit tests for OpenCompass-compatible causal-LM scoring."""

import math
import unittest

import torch

from hyper_parallel.integration.opencompass.scoring import causal_lm_ppl_scores


class TestCausalLMPPLScores(unittest.TestCase):
    """Validate the exact normalization contract used by MMLU PPL."""

    def test_uniform_logits_use_opencompass_full_token_denominator(self):
        """Three predicted tokens are normalized by four non-padding input tokens."""
        logits = torch.zeros(1, 4, 5)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        scores = causal_lm_ppl_scores(logits, input_ids, pad_token_id=0)

        self.assertAlmostEqual(scores.item(), 3.0 * math.log(5.0) / 4.0, places=6)

    def test_padding_tokens_are_excluded_from_loss_and_denominator(self):
        """Right padding contributes neither token loss nor valid-token count."""
        logits = torch.zeros(1, 4, 5)
        input_ids = torch.tensor([[1, 2, 3, 0]])

        scores = causal_lm_ppl_scores(logits, input_ids, pad_token_id=0)

        self.assertAlmostEqual(scores.item(), 2.0 * math.log(5.0) / 3.0, places=6)

    def test_mask_length_scores_only_unmasked_continuation(self):
        """A two-token prefix leaves two shifted continuation targets."""
        logits = torch.zeros(1, 4, 5)
        input_ids = torch.tensor([[1, 2, 3, 4]])

        scores = causal_lm_ppl_scores(
            logits,
            input_ids,
            pad_token_id=0,
            mask_length=[2],
        )

        self.assertAlmostEqual(scores.item(), math.log(5.0), places=6)

    def test_rejects_mask_that_leaves_no_scored_token(self):
        """Mask length must be smaller than each sequence's valid length."""
        logits = torch.zeros(1, 4, 5)
        input_ids = torch.tensor([[1, 2, 3, 0]])

        with self.assertRaisesRegex(ValueError, "leave at least one scored token"):
            causal_lm_ppl_scores(
                logits,
                input_ids,
                pad_token_id=0,
                mask_length=[3],
            )

    def test_rejects_misaligned_shapes(self):
        """Sequence dimensions must align before shifting labels and logits."""
        with self.assertRaisesRegex(ValueError, "identical batch and sequence"):
            causal_lm_ppl_scores(
                torch.zeros(1, 3, 5),
                torch.ones(1, 4, dtype=torch.long),
                pad_token_id=0,
            )


if __name__ == "__main__":
    unittest.main()
