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

    def test_aux_folding_reclassifies_foundation_as_objective(self):
        """A folded auxiliary term must disable perplexity, not contaminate it."""
        loss = ChunkedCausalLMLoss()
        self.assertIsNone(loss.statistic_kinds("foundation_loss"))
        output = ChunkedCausalLMOutput(
            loss_sum=torch.tensor(4.0),
            valid_token_count=torch.tensor(2),
            aux_loss=torch.tensor(0.5),
            aux_loss_coef=0.1,
        )
        value = loss(model_output=output, labels=None)
        self.assertAlmostEqual(float(value), 4.0 / 2.0 + 0.1 * 0.5, places=6)
        self.assertEqual(loss.statistic_kinds("foundation_loss"), "objective")
        self.assertIsNone(loss.statistic_kinds("image_decoder_loss"))

    def test_pure_nll_without_aux_keeps_the_language_model_kind(self):
        """Without an auxiliary term the value stays a genuine NLL."""
        loss = ChunkedCausalLMLoss()
        output = ChunkedCausalLMOutput(
            loss_sum=torch.tensor(4.0),
            valid_token_count=torch.tensor(2),
        )
        value = loss(model_output=output, labels=None)
        self.assertAlmostEqual(float(value), 2.0, places=6)
        self.assertIsNone(loss.statistic_kinds("foundation_loss"))

    def test_disabled_aux_keeps_pure_nll_even_with_nonfinite_aux(self):
        """Zero or absent coefficients cannot contaminate the value or its kind."""
        for coefficient in (None, 0.0):
            for auxiliary in (0.5, float("nan")):
                with self.subTest(coefficient=coefficient, auxiliary=auxiliary):
                    loss = ChunkedCausalLMLoss()
                    output = ChunkedCausalLMOutput(
                        loss_sum=torch.tensor(4.0), valid_token_count=torch.tensor(2),
                        aux_loss=torch.tensor(auxiliary), aux_loss_coef=coefficient,
                    )
                    self.assertEqual(float(loss(model_output=output, labels=None)), 2.0)
                    self.assertIsNone(loss.statistic_kinds("foundation_loss"))

    def test_statistic_kind_tracks_only_the_current_forward(self):
        """Reusing a loss after an aux-bearing call must not retain its old kind."""
        loss = ChunkedCausalLMLoss()
        for coefficient, auxiliary, expected in (
                (0.1, 0.5, "objective"), (0.0, 0.5, None),
                (0.1, 0.5, "objective"), (None, 0.5, None), (0.1, None, None)):
            with self.subTest(coefficient=coefficient, auxiliary=auxiliary):
                output = ChunkedCausalLMOutput(
                    loss_sum=torch.tensor(4.0), valid_token_count=torch.tensor(2),
                    aux_loss=None if auxiliary is None else torch.tensor(auxiliary),
                    aux_loss_coef=coefficient,
                )
                loss(model_output=output, labels=None)
                self.assertEqual(loss.statistic_kinds("foundation_loss"), expected)


if __name__ == "__main__":
    unittest.main()
