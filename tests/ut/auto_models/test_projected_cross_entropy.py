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
"""Projected loss preserves eager CE gradients while bounding retained logits."""

import copy
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import DefaultDeviceType
from transformers.loss.loss_utils import ForCausalLMLoss

from hyper_parallel.components.losses import projected_cross_entropy
from hyper_parallel.core.tensor_parallel.loss_parallel import is_loss_parallel_active


class TestProjectedCrossEntropy(unittest.TestCase):
    """Check numerical equivalence, token weighting and checkpoint tensor lifetime."""

    def setUp(self) -> None:
        """Keep CPU checkpoint recomputation independent of installed accelerators."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")

    def test_loss_and_scaled_gradients(self) -> None:
        """Uneven chunks, empty chunks and frozen inputs keep the eager objective."""
        cases = ((1, False, False), (3, False, False), (20, False, False), (3, True, False), (3, False, True))
        for chunk_size, frozen_hidden, frozen_head in cases:
            with self.subTest(chunk_size=chunk_size,
                              frozen_hidden=frozen_hidden, frozen_head=frozen_head):
                torch.manual_seed(21)
                hidden = torch.randn(2, 7, 5, requires_grad=not frozen_hidden)
                head = nn.Linear(5, 13)
                head.requires_grad_(not frozen_head)
                reference_hidden = hidden.detach().clone().requires_grad_(not frozen_hidden)
                reference_head = copy.deepcopy(head)
                targets = torch.tensor([[-100, -100, -100, 3, 4, -100, 6],
                                        [-100, -100, -100, 8, -100, 9, 1]])
                actual = projected_cross_entropy(hidden, targets, head=head, loss_fn=ForCausalLMLoss,
                                                   vocab_size=13, chunk_size=chunk_size)
                expected = ForCausalLMLoss(logits=reference_head(reference_hidden), labels=None,
                                           vocab_size=13, shift_labels=targets)
                (actual * 0.37).backward()
                (expected * 0.37).backward()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(hidden.grad, reference_hidden.grad)
                for parameter, reference in zip(head.parameters(), reference_head.parameters()):
                    torch.testing.assert_close(parameter.grad, reference.grad)

    def test_all_ignored_and_retained_graph(self) -> None:
        """Ignored targets produce zero gradients, including repeated backward."""
        hidden = torch.randn(1, 5, 3, requires_grad=True)
        head = nn.Linear(3, 11, bias=False)
        loss = projected_cross_entropy(hidden, torch.full((1, 5), -100), head=head,
                                       loss_fn=ForCausalLMLoss, vocab_size=11, chunk_size=2)
        loss.backward(retain_graph=True)
        loss.backward()
        torch.testing.assert_close(loss, torch.zeros_like(loss))
        torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
        torch.testing.assert_close(head.weight.grad, torch.zeros_like(head.weight))

    def test_no_logits_saved_and_projection_is_bounded(self) -> None:
        """Forward retains checkpoint inputs, never [tokens, vocabulary] activations."""
        hidden = torch.randn(1, 11, 5, requires_grad=True)
        head = nn.Linear(5, 97, bias=False)
        projected_lengths = []
        handle = head.register_forward_hook(lambda _module, _args, output: projected_lengths.append(output.shape[1]))
        self.addCleanup(handle.remove)
        saved_shapes = []

        def save(tensor: torch.Tensor) -> torch.Tensor:
            """Record activation geometry without retaining an extra tensor copy."""
            saved_shapes.append(tuple(tensor.shape))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            loss = projected_cross_entropy(hidden, torch.zeros(1, 11, dtype=torch.long), head=head,
                                           loss_fn=ForCausalLMLoss, vocab_size=97, chunk_size=3)
        loss.backward()
        handle.remove()
        self.assertEqual(projected_lengths[:4], [3, 3, 3, 2])
        self.assertGreater(len(projected_lengths), 4)
        self.assertLessEqual(max(projected_lengths), 3)
        self.assertFalse(any(97 in shape for shape in saved_shapes), saved_shapes)

    def test_rejects_invalid_geometry(self) -> None:
        """Fail before a head call if sequence shards cannot align with labels."""
        hidden = torch.randn(1, 5, 3)
        head = nn.Linear(3, 11, bias=False)
        for chunk_size, parts, length in ((0, 1, 5), (True, 1, 5), (3, 2, 10), (4, 2, 9)):
            with self.subTest(chunk_size=chunk_size, parts=parts, length=length), self.assertRaises(ValueError):
                projected_cross_entropy(hidden, torch.zeros(1, length, dtype=torch.long), head=head,
                                         loss_fn=ForCausalLMLoss, vocab_size=11, chunk_size=chunk_size,
                                         sequence_parallel_size=parts)

    def test_sequence_gather_matches_target_order(self) -> None:
        """Mock rank-order gathering with distinct shards and a non-divisible tail."""
        torch.manual_seed(37)
        hidden = torch.randn(2, 5, 3, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_()
        head = nn.Linear(3, 11)
        reference_head = copy.deepcopy(head)
        targets = torch.tensor([[0, 1, -100, 3, 4, 5, -100, 7, 8, 9],
                                [9, -100, 7, 6, 5, 4, 3, 2, -100, 0]])

        def gather_head(states: torch.Tensor) -> torch.Tensor:
            """Emulate two distinct rank shards without distributed UT setup."""
            return head(torch.cat((states, states + 2), dim=1))

        actual = projected_cross_entropy(hidden, targets, head=gather_head, loss_fn=ForCausalLMLoss,
                                         vocab_size=11, chunk_size=4, sequence_parallel_size=2)
        expected = ForCausalLMLoss(
            logits=reference_head(torch.cat((reference_hidden, reference_hidden + 2), dim=1)),
            labels=None, vocab_size=11, shift_labels=targets,
        )
        actual.backward()
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        for parameter, reference in zip(head.parameters(), reference_head.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad)

    def test_loss_parallel_context_is_reentered_for_recomputation(self) -> None:
        """Recompute outside the forward context, without leaking context state."""
        hidden = torch.randn(1, 5, 3, requires_grad=True)
        head = nn.Linear(3, 11)
        contexts = []

        def checked_head(states: torch.Tensor) -> torch.Tensor:
            """Observe both forward and backward checkpoint projection calls."""
            contexts.append(is_loss_parallel_active())
            return head(states)

        loss = projected_cross_entropy(hidden, torch.zeros(1, 5, dtype=torch.long), head=checked_head,
                                       loss_fn=ForCausalLMLoss, vocab_size=11, chunk_size=2, tp_mesh=object())
        self.assertFalse(is_loss_parallel_active())
        self.assertEqual(len(contexts), 3)
        loss.backward()
        self.assertEqual(len(contexts), 6)
        self.assertTrue(all(contexts))
        self.assertFalse(is_loss_parallel_active())

    def test_custom_ignore_index_and_invalid_labels(self) -> None:
        """Custom ignored targets use one denominator; invalid dtypes fail early."""
        hidden = torch.randn(1, 5, 3, requires_grad=True)
        head = nn.Linear(3, 11)
        targets = torch.tensor([[-1, -1, 2, -1, 4]])
        actual = projected_cross_entropy(hidden, targets, head=head, loss_fn=ForCausalLMLoss,
                                         vocab_size=11, chunk_size=2, ignore_index=-1)
        expected = ForCausalLMLoss(logits=head(hidden), labels=None, vocab_size=11,
                                   shift_labels=targets, ignore_index=-1)
        torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(TypeError, "torch.long"):
            projected_cross_entropy(hidden, targets.int(), head=head, loss_fn=ForCausalLMLoss,
                                    vocab_size=11, chunk_size=2)

    def test_rejects_head_that_does_not_gather_sequence(self) -> None:
        """A local-only head cannot consume targets covering all SP shards."""
        with self.assertRaisesRegex(ValueError, "gather sequence shards"):
            projected_cross_entropy(torch.randn(1, 5, 3), torch.zeros(1, 10, dtype=torch.long),
                                    head=nn.Linear(3, 11), loss_fn=ForCausalLMLoss,
                                    vocab_size=11, chunk_size=4, sequence_parallel_size=2)
