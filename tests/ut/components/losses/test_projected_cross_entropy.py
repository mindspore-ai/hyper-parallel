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
"""CPU contract tests for the caller-owned projected chunk loss."""
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import DefaultDeviceType
from hyper_parallel.components.losses.projected_cross_entropy_sum import projected_cross_entropy_sum

class WrappedHead(nn.Linear):
    """Single-process boundary simulation, not TP or FSDP."""

    def __init__(self, hidden: int, local_vocab: int) -> None:
        """Validate the projected cross-entropy contract."""
        super().__init__(hidden, local_vocab, bias=False)
        self.register_buffer('other', torch.randn(local_vocab, hidden))
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Validate the projected cross-entropy contract."""
        self.calls += 1
        return torch.cat((F.linear(x, self.weight), F.linear(x, self.other)), dim=-1)

class TestProjectedCrossEntropy(unittest.TestCase):
    """Validate the projection boundary, gradient lifetime and FP32 accumulation."""

    def setUp(self) -> None:
        """Keep CPU checkpoint recomputation independent of installed accelerator plugins."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")

    def test_float_reference_recompute_and_weight_identity(self) -> None:
        """
        Feature: Projection boundary and autograd lifetime.
        Description: Compare a full FP32 objective and verify checkpoint re-entry.
        Expectation: Loss and gradients match; original parameter identity and data stay unchanged.
        """
        torch.manual_seed(137)
        h = torch.randn(2, 7, 5, requires_grad=True)
        head = nn.Linear(5, 13, bias=False)
        y = torch.randint(0, 13, (2, 7))
        y[0, 2] = -100
        y[1, 6] = -100
        original = head.weight
        before = original.detach().clone()
        calls = []
        hook = head.register_forward_hook(lambda _m, _i, _o: calls.append(1))
        loss = projected_cross_entropy_sum(h, y, head, vocab_size=13, chunk_size=3)
        self.assertEqual(len(calls), 3)
        self.assertIs(head.weight, original)
        (loss * 0.37).backward()
        self.assertGreater(len(calls), 3)
        self.assertIs(head.weight, original)
        torch.testing.assert_close(original, before, rtol=0, atol=0)
        hook.remove()
        hr = h.detach().clone().requires_grad_()
        wr = before.clone().requires_grad_()
        ref = F.cross_entropy(F.linear(hr, wr).reshape(-1, 13), y.reshape(-1), ignore_index=-100, reduction='sum')
        (ref * 0.37).backward()
        torch.testing.assert_close(loss, ref, rtol=1e-06, atol=2e-06)
        torch.testing.assert_close(h.grad, hr.grad, rtol=1e-06, atol=2e-07)
        torch.testing.assert_close(head.weight.grad, wr.grad, rtol=1e-06, atol=2e-07)

    def test_bfloat16_accumulates_cancelling_chunk_gradients_in_fp32(self) -> None:
        """
        Feature: Low-precision chunk gradient accumulation.
        Description: Use an exactly representable cancellation with a non-unit upstream scale.
        Expectation: FP32 accumulation retains the analytic residual at the original parameter boundary.
        """
        h = torch.tensor([[[256.0], [0.015625], [-256.0], [0.0]]], dtype=torch.bfloat16)
        h = h.repeat(4, 1, 1).requires_grad_()
        head = nn.Linear(1, 2, bias=False, dtype=torch.bfloat16)
        head.weight.data.zero_()
        y = torch.zeros(4, 4, dtype=torch.long)
        loss = projected_cross_entropy_sum(h, y, head, vocab_size=2, chunk_size=1)
        (loss / (y.numel() * 128)).backward()
        expected = torch.tensor([[-2.0 ** (-16)], [2.0 ** (-16)]], dtype=torch.bfloat16)
        torch.testing.assert_close(head.weight.grad, expected, rtol=0, atol=0)
        torch.testing.assert_close(h.grad, torch.zeros_like(h.grad), rtol=0, atol=0)

    def test_no_grad_hidden_all_ignored_and_shard_wrapper(self) -> None:
        """
        Feature: Optional input gradient and local projection boundaries.
        Description: Exercise ignored targets and a local-weight module that returns complete vocabulary logits.
        Expectation: Zero targets produce zero gradients and the local projection matches the full objective.
        """
        torch.manual_seed(93)
        head = WrappedHead(4, 5)
        orig = head.weight
        h = torch.randn(2, 7, 4)
        y = torch.full((2, 7), -100, dtype=torch.long)
        zero = projected_cross_entropy_sum(h, y, head, vocab_size=10, chunk_size=3)
        zero.backward()
        self.assertEqual(zero.item(), 0.0)
        self.assertIs(head.weight, orig)
        torch.testing.assert_close(head.weight.grad, torch.zeros_like(head.weight.grad), rtol=0, atol=0)
        head.zero_grad()
        h.requires_grad_()
        y = torch.randint(0, 10, (2, 7))
        y[0, 0] = -100
        v = projected_cross_entropy_sum(h, y, head, vocab_size=10, chunk_size=3)
        v.backward()
        hr = h.detach().clone().requires_grad_()
        wr = head.weight.detach().clone().requires_grad_()
        logits = torch.cat((F.linear(hr, wr), F.linear(hr, head.other)), dim=-1)
        ref = F.cross_entropy(logits.reshape(-1, 10), y.reshape(-1), ignore_index=-100, reduction='sum')
        ref.backward()
        torch.testing.assert_close(v, ref, rtol=1e-06, atol=2e-06)
        torch.testing.assert_close(h.grad, hr.grad, rtol=1e-06, atol=2e-07)
        torch.testing.assert_close(head.weight.grad, wr.grad, rtol=1e-06, atol=2e-07)
        self.assertIs(head.weight, orig)

    def test_invalid_inputs_fail_fast(self) -> None:
        """
        Feature: Projected objective input and layout contract.
        Description: Reject unsupported shapes, bias, options, dtypes and distributed weight metadata.
        Expectation: Invalid inputs fail before computing a misleading partial-vocabulary objective.
        """
        head = nn.Linear(4, 7, bias=False)
        h = torch.randn(1, 3, 4)
        y = torch.zeros(1, 3, dtype=torch.long)
        with self.assertRaises(TypeError):
            projected_cross_entropy_sum(h, y, nn.Linear(4, 7), vocab_size=7)
        with self.assertRaises(ValueError):
            projected_cross_entropy_sum(h[:, :0], y[:, :0], head, vocab_size=7)
        with self.assertRaises(ValueError):
            projected_cross_entropy_sum(h, y[:, :2], head, vocab_size=7)
        with self.assertRaises(ValueError):
            projected_cross_entropy_sum(h, y, head, vocab_size=6)
        with self.assertRaises(ValueError):
            projected_cross_entropy_sum(h, y, head, vocab_size=8)
        with self.assertRaises(TypeError):
            projected_cross_entropy_sum(h, y.int(), head, vocab_size=7)
        with self.assertRaises(ValueError):
            projected_cross_entropy_sum(h, y, head, vocab_size=7, chunk_size=0)
        head.weight.placements = None
        local_loss = projected_cross_entropy_sum(h, y, head, vocab_size=7)
        self.assertTrue(bool(torch.isfinite(local_loss)))
        head.weight.placements = ()
        with self.assertRaisesRegex(NotImplementedError, 'local unsharded'):
            projected_cross_entropy_sum(h, y, head, vocab_size=7)
if __name__ == '__main__':
    unittest.main()
