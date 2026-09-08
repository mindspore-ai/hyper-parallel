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
"""Unit tests for chunked linear cross-entropy."""
# pylint: disable=not-callable

import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # pylint: disable=wrong-import-position
from torch.nn import functional  # pylint: disable=wrong-import-position

from hyper_parallel.components.losses import (  # pylint: disable=wrong-import-position
    chunked_cross_entropy,
)


def _eager_loss(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return the full-logits FP32 summed cross-entropy reference."""
    logits = functional.linear(hidden_states, head_weight).float()
    return functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


class TestChunkedCrossEntropy(unittest.TestCase):
    """Compare Chunk Loss with ordinary eager autograd."""

    def test_matches_eager_loss_and_scaled_gradients(self) -> None:
        """Match uneven chunks, ignored targets, and upstream scaling."""
        torch.manual_seed(7)
        hidden = torch.randn(2, 7, 5, requires_grad=True)
        weight = torch.randn(13, 5, requires_grad=True)
        targets = torch.randint(0, 13, (2, 7))
        targets[0, 2] = -100
        targets[1, 6] = -100

        loss = chunked_cross_entropy(hidden, targets, weight, chunk_size=3)
        (loss * 0.37).backward()
        hidden_grad = hidden.grad.detach().clone()
        weight_grad = weight.grad.detach().clone()

        ref_hidden = hidden.detach().clone().requires_grad_()
        ref_weight = weight.detach().clone().requires_grad_()
        ref_loss = _eager_loss(ref_hidden, ref_weight, targets)
        (ref_loss * 0.37).backward()

        torch.testing.assert_close(loss, ref_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(hidden_grad, ref_hidden.grad, rtol=1e-6, atol=2e-7)
        torch.testing.assert_close(weight_grad, ref_weight.grad, rtol=1e-6, atol=2e-7)

    def test_all_ignored_targets_produce_zero_loss_and_gradients(self) -> None:
        """Keep an all-padding local shard finite and exactly zero."""
        hidden = torch.randn(2, 5, 4, requires_grad=True)
        weight = torch.randn(9, 4, requires_grad=True)
        targets = torch.full((2, 5), -100, dtype=torch.long)

        loss = chunked_cross_entropy(hidden, targets, weight, chunk_size=2)
        loss.backward()

        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))

    def test_rejects_invalid_core_inputs(self) -> None:
        """Reject invalid chunk size, target shape, and target dtype."""
        hidden = torch.randn(1, 3, 4)
        targets = torch.randint(0, 7, (1, 3))
        weight = torch.randn(7, 4)

        with self.assertRaises(ValueError):
            chunked_cross_entropy(hidden, targets, weight, chunk_size=0)
        with self.assertRaises(ValueError):
            chunked_cross_entropy(hidden, targets[:, :-1], weight)
        with self.assertRaises(TypeError):
            chunked_cross_entropy(hidden, targets.int(), weight)


if __name__ == "__main__":
    unittest.main()
