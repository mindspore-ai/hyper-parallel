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
"""Bounded replica traffic and numerical oracle tests without distributed setup."""

import unittest

import torch

from hyper_parallel.core.multicore.modules.routed_experts.reference import (
    backward_values, combine, combine_gradient, forward_values, grouped_gradient, swiglu,
)
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import (
    compile_transfer_rounds, copy_matrix_fragment,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestTransferRounds(unittest.TestCase):
    """Check fanout, split matrices, empty participants and inverse attribution."""

    def test_owner_fanout_stays_bounded(self):
        """An owner may send 120 experts despite having only eight home experts."""
        allocation = torch.zeros(128, 16, dtype=torch.int64)
        allocation[:8] = 1
        for reverse in (False, True):
            rounds = compile_transfer_rounds(allocation, (("gate", 7), ("down", 5)),
                                             reverse=reverse, quota_elements=19)
            coverage = {}
            for item in rounds:
                for rank in range(16):
                    sent, received, send_sizes, receive_sizes = item.local(rank)
                    self.assertLessEqual(sum(send_sizes), 19)
                    self.assertLessEqual(sum(receive_sizes), 19)
                    self.assertEqual(sum(fragment.elements for fragment in sent), sum(send_sizes))
                    self.assertEqual(sum(fragment.elements for fragment in received), sum(receive_sizes))
                    for peer in range(16):
                        self.assertEqual(send_sizes[peer], item.local(peer)[3][rank])
                for fragment in item.fragments:
                    key = fragment.source, fragment.target, fragment.expert, fragment.role
                    self.assertEqual(fragment.offset, coverage.get(key, 0))
                    coverage[key] = fragment.offset + fragment.elements
            self.assertEqual(len(coverage), 240)
            for (source, target, expert, role), elements in coverage.items():
                self.assertEqual(target if reverse else source, 0)
                self.assertLess(expert, 8)
                self.assertEqual(elements, 7 if role == "gate" else 5)

    def test_fragment_copy_handles_transposed_borrowed_weights(self):
        """Partial logical rows copy and accumulate without changing the source."""
        matrix = torch.arange(35).reshape(5, 7).t()
        for offset in range(matrix.numel()):
            for count in range(1, matrix.numel() - offset + 1):
                flat = torch.empty(count, dtype=matrix.dtype)
                copy_matrix_fragment(flat, matrix, offset)
                torch.testing.assert_close(flat, matrix.flatten()[offset:offset + count], rtol=0, atol=0)
                target = torch.zeros_like(matrix)
                copy_matrix_fragment(flat, target, offset, unpack=True)
                copy_matrix_fragment(flat, target, offset, unpack=True, accumulate=True)
                expected = torch.zeros(matrix.numel(), dtype=matrix.dtype)
                expected[offset:offset + count] = 2 * flat
                torch.testing.assert_close(target, expected.reshape_as(matrix), rtol=0, atol=0)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestReferenceMath(unittest.TestCase):
    """Check forward, individual dW readiness and router derivatives against autograd."""

    def test_split_experts_and_gradients(self):
        """Independent matrix references give the same derivative as a dense loop."""
        generator = torch.Generator().manual_seed(233)
        x = torch.randn(9, 4, generator=generator, requires_grad=True)
        gate = [torch.randn(4, 6, generator=generator, requires_grad=True) for _ in range(3)]
        down = [torch.randn(3, 4, generator=generator, requires_grad=True) for _ in range(3)]
        boundaries = torch.tensor([2, 3, 9])
        output, state = forward_values(x, gate, down, boundaries, 3)
        parts, start = [], 0
        for index, end in enumerate(boundaries.tolist()):
            parts.append(swiglu(x[start:end] @ gate[index]) @ down[index])
            start = end
        expected = torch.cat(parts)
        dy = torch.randn(output.shape, generator=generator)
        expected.backward(dy)
        ready = []
        dx, dg, dd = backward_values(dy, state, gate, down, lambda role, value: ready.append(role))
        self.assertEqual(ready, ["down", "gate"])
        for actual, reference in ((output, expected), (dx, x.grad),
                                  (dg, torch.stack([weight.grad for weight in gate])),
                                  (dd, torch.stack([weight.grad for weight in down]))):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    def test_combine_derivatives(self):
        """Products follow ordinary EP's dtype promotion and rounding boundary."""
        for score_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(score_dtype=score_dtype):
                generator = torch.Generator().manual_seed(59)
                values = torch.randn(24, 16, generator=generator, dtype=torch.bfloat16, requires_grad=True)
                scores = torch.randn(6, 4, generator=generator, dtype=score_dtype, requires_grad=True)
                order = torch.randperm(24, generator=generator)
                rows = order // scores.shape[1]
                weighted = values * scores.flatten()[order].unsqueeze(-1)
                expected = weighted.new_zeros((6, 16)).index_add_(0, rows, weighted)
                dy = torch.randn(expected.shape, generator=generator, dtype=expected.dtype)
                expected.backward(dy)

                output = combine(values.detach(), order, scores.detach(), 6)
                dv, ds = combine_gradient(dy, values.detach(), order, scores.detach())

                torch.testing.assert_close(output, expected.detach(), rtol=0, atol=0)
                torch.testing.assert_close(dv, values.grad, rtol=0, atol=0)
                torch.testing.assert_close(ds, scores.grad, rtol=0, atol=0)
                if score_dtype == torch.bfloat16:
                    unrounded_ds = torch.empty_like(scores).flatten()
                    unrounded_ds[order] = (dy[rows].float() * values.detach().float()).sum(-1).to(score_dtype)
                    self.assertFalse(torch.equal(ds, unrounded_ds.view_as(scores)),
                                     f"rounded dscore={ds}, unrounded FP32 dot={unrounded_ds.view_as(scores)}")


    def test_bfloat16_partials_round_only_after_owner_merge(self):
        """A representable final sum must retain a partial at the BF16 midpoint."""
        x = torch.ones((3, 1), dtype=torch.bfloat16)
        dy = torch.tensor([[1.0], [1 / 256], [1 / 256]], dtype=torch.bfloat16)

        partials = grouped_gradient(x, dy, torch.tensor([2, 3]))
        complete = grouped_gradient(x, dy, torch.tensor([3]))[0]
        merged = partials.sum(0)
        early_rounded = partials.to(torch.bfloat16).sum(0)

        torch.testing.assert_close(partials[0], torch.tensor([[1 + 1 / 256]]), rtol=0, atol=0)
        torch.testing.assert_close(merged.double(), x.double().t() @ dy.double(), rtol=0, atol=0)
        torch.testing.assert_close(merged.to(torch.bfloat16), complete.to(torch.bfloat16), rtol=0, atol=0)
        self.assertFalse(torch.equal(early_rounded, complete.to(torch.bfloat16)),
                         f"early BF16 merge={early_rounded}, once-rounded result={complete.to(torch.bfloat16)}")
