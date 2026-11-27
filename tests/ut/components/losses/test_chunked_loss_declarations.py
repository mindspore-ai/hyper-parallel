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
"""Evaluation declarations of the chunked causal-LM objective."""

import unittest

import torch

from hyper_parallel.components.losses.chunked_cross_entropy import (
    ChunkedCausalLMLoss,
    ChunkedCausalLMOutput,
)


class TestChunkedLossEvaluationDeclarations(unittest.TestCase):
    """The loss reports how evaluation must weight and classify its value."""

    def test_normalization_is_one_local_token_mean(self):
        """Evaluation needs the declared normalization to build a numerator."""
        self.assertEqual(ChunkedCausalLMLoss.loss_normalization, "token_mean")

    def test_aux_folding_marks_foundation_as_non_nll(self):
        """A folded auxiliary term must disable perplexity, not contaminate it."""
        loss = ChunkedCausalLMLoss()
        self.assertIs(loss.statistic_is_lm_nll("foundation_loss"), True)
        output = ChunkedCausalLMOutput(
            loss_sum=torch.tensor(4.0),
            valid_token_count=torch.tensor(2),
            aux_loss=torch.tensor(0.5),
            aux_loss_coef=0.1,
        )
        value = loss(model_output=output, labels=None)
        self.assertAlmostEqual(float(value), 4.0 / 2.0 + 0.1 * 0.5, places=6)
        self.assertIs(loss.statistic_is_lm_nll("foundation_loss"), False)
        self.assertIsNone(loss.statistic_is_lm_nll("image_decoder_loss"))

    def test_pure_nll_without_aux_keeps_perplexity_enabled(self):
        """Without an auxiliary term the value stays a genuine NLL."""
        loss = ChunkedCausalLMLoss()
        output = ChunkedCausalLMOutput(
            loss_sum=torch.tensor(4.0),
            valid_token_count=torch.tensor(2),
        )
        value = loss(model_output=output, labels=None)
        self.assertAlmostEqual(float(value), 2.0, places=6)
        self.assertIs(loss.statistic_is_lm_nll("foundation_loss"), True)

    def test_disabled_aux_keeps_pure_nll_even_with_nonfinite_aux(self):
        """Zero or absent coefficients cannot contaminate the value or disable PPL."""
        for coefficient in (None, 0.0):
            for auxiliary in (0.5, float("nan")):
                with self.subTest(coefficient=coefficient, auxiliary=auxiliary):
                    loss = ChunkedCausalLMLoss()
                    output = ChunkedCausalLMOutput(
                        loss_sum=torch.tensor(4.0), valid_token_count=torch.tensor(2),
                        aux_loss=torch.tensor(auxiliary), aux_loss_coef=coefficient,
                    )
                    self.assertEqual(float(loss(model_output=output, labels=None)), 2.0)
                    self.assertIs(loss.statistic_is_lm_nll("foundation_loss"), True)

    def test_nll_declaration_tracks_only_the_current_forward(self):
        """Reusing a loss after an aux-bearing call must not retain its old flag."""
        loss = ChunkedCausalLMLoss()
        for coefficient, auxiliary, expected in (
                (0.1, 0.5, False), (0.0, 0.5, True),
                (0.1, 0.5, False), (None, 0.5, True), (0.1, None, True)):
            with self.subTest(coefficient=coefficient, auxiliary=auxiliary):
                output = ChunkedCausalLMOutput(
                    loss_sum=torch.tensor(4.0), valid_token_count=torch.tensor(2),
                    aux_loss=None if auxiliary is None else torch.tensor(auxiliary),
                    aux_loss_coef=coefficient,
                )
                loss(model_output=output, labels=None)
                self.assertIs(loss.statistic_is_lm_nll("foundation_loss"), expected)


if __name__ == "__main__":
    unittest.main()
