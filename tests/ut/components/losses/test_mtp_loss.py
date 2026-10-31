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
"""Independent target alignment and gradient checks for the MTP objective."""

import unittest

import torch
from torch.nn import functional as F

from hyper_parallel.components.losses import deepseek_v3_mtp_loss
from hyper_parallel.components.losses.mtp import calculate_mtp_loss


class TestDeepseekV3MTPLoss(unittest.TestCase):
    """Test depth-dependent targets, denominators and ignored positions."""

    def setUp(self) -> None:
        """Use two batches with unequal valid-target counts at each depth."""
        torch.manual_seed(7)
        self.logits = [torch.randn(2, 4, 7, requires_grad=True), torch.randn(2, 3, 7, requires_grad=True)]
        self.targets = torch.tensor([[1, 2, -100, 4, 5], [6, -100, 1, 2, 3]])

    def test_matches_tokenwise_reference_loss_and_gradients(self) -> None:
        """Use individual token CE terms, not the implementation's batched loop."""
        for normalization in ("sequence", "valid_tokens"):
            with self.subTest(normalization=normalization):
                reference_logits = [value.detach().clone().requires_grad_() for value in self.logits]
                total = 0
                for depth, logits in enumerate(reference_logits, start=1):
                    terms = []
                    for batch in range(2):
                        for position in range(5 - depth):
                            target = self.targets[batch, position + depth]
                            if target != -100:
                                terms.append(-logits[batch, position].log_softmax(-1)[target])
                    denominator = 10 if normalization == "sequence" else len(terms)
                    total = total + torch.stack(terms).sum() / denominator
                expected = total * 0.3 / 2
                actual = deepseek_v3_mtp_loss(self.logits, self.targets, normalization=normalization)
                torch.testing.assert_close(actual, expected)
                for got, want in zip(torch.autograd.grad(actual, self.logits),
                                     torch.autograd.grad(expected, reference_logits)):
                    torch.testing.assert_close(got, want)

    def test_all_ignored_is_connected_zero(self) -> None:
        """Masked batches must not produce NaNs or break backward."""
        for normalization in ("sequence", "valid_tokens"):
            loss = deepseek_v3_mtp_loss(self.logits, torch.full_like(self.targets, -100),
                                        normalization=normalization)
            torch.testing.assert_close(loss, torch.tensor(0.))
            for grad in torch.autograd.grad(loss, self.logits):
                torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_single_valid_tail_target_and_empty_deeper_depth(self) -> None:
        """A target valid at depth 1 can be absent at depth 2."""
        labels = torch.full_like(self.targets, -100)
        labels[0, 1] = 2
        loss = deepseek_v3_mtp_loss(self.logits, labels, normalization="valid_tokens")
        expected = -self.logits[0][0, 0].log_softmax(-1)[2] * 0.3 / 2
        torch.testing.assert_close(loss, expected)

    def test_weight_zero_and_linear_scaling(self) -> None:
        """The coefficient controls both the scalar loss and gradients."""
        base = deepseek_v3_mtp_loss(self.logits, self.targets, loss_factor=1.)
        scaled = deepseek_v3_mtp_loss(self.logits, self.targets, loss_factor=0.2)
        torch.testing.assert_close(scaled, base * 0.2)
        zero = deepseek_v3_mtp_loss(self.logits, self.targets, loss_factor=0.)
        for grad in torch.autograd.grad(zero, self.logits):
            torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_low_precision_logits_use_float32_loss(self) -> None:
        """Loss accumulation does not inherit BF16 logits precision."""
        logits = [value.detach().to(torch.bfloat16).requires_grad_() for value in self.logits]
        loss = deepseek_v3_mtp_loss(logits, self.targets)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in logits))

    def test_legacy_loss_api_unchanged(self) -> None:
        """The new objective must not change existing calculate_mtp_loss callers."""
        logits = torch.randn(2, 5, 7)
        actual = calculate_mtp_loss([logits], [], self.targets, torch.nn.CrossEntropyLoss())
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, 7), self.targets[:, 1:].reshape(-1))
        torch.testing.assert_close(actual, expected)

    def test_invalid_options(self) -> None:
        """Reject ambiguous target layouts and invalid objective scales."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            deepseek_v3_mtp_loss([], self.targets)
        with self.assertRaisesRegex(ValueError, "torch.long"):
            deepseek_v3_mtp_loss(self.logits, self.targets.float())
        with self.assertRaisesRegex(ValueError, "normalization"):
            deepseek_v3_mtp_loss(self.logits, self.targets, normalization="unknown")
        with self.assertRaisesRegex(ValueError, "Depth k"):
            deepseek_v3_mtp_loss(self.logits[::-1], self.targets)
        for factor in (-1., float("inf"), float("nan")):
            with self.subTest(factor=factor), self.assertRaisesRegex(ValueError, "finite"):
                deepseek_v3_mtp_loss(self.logits, self.targets, loss_factor=factor)
