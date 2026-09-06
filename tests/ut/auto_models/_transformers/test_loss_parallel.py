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
"""Characterization: ``causal_lm_loss_parallel`` local-first production contract.

Post-rework contract (plan §14.6 item 8 / §15.7 step 4): the adapter passes
the full local logits tensor, the global vocabulary size and the loss-parallel
mesh straight to ``vocab_parallel_cross_entropy_local`` and never calls
``DTensor.from_local``. The pad-and-shift label contract, float32 upcast,
ignore-index denominator, the explicit ``num_items_in_batch`` denominator,
and the fail-fast errors are locked here.

Gate-1: no process group — ``vocab_parallel_cross_entropy_local`` is
replaced by a recording fake whose numerics are the pure-Torch single-rank
reference.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from typing import Any, Optional
from unittest import mock

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
import torch.nn.functional as F

from hyper_parallel.models._transformers import loss_parallel as loss_module
from tests.common.mark_utils import arg_mark

_VOCAB = 11


class _NoDTensor:
    """Stand-in for ``DTensor``: ``from_local`` must never be called."""

    from_local = mock.Mock(
        side_effect=AssertionError("local-first path must not call DTensor.from_local")
    )


class _LocalCeRecorder:
    """Records every ``vocab_parallel_cross_entropy_local`` call."""

    def __init__(self, mesh: Any) -> None:
        """Initialize the recorder for the expected mesh."""
        self.mesh = mesh
        self.calls = []

    def __call__(
        self,
        local_logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        vocab_size: int,
        mesh: Any,
        mesh_dim: Optional[int] = None,
        ignore_index: int,
        reduction: str,
    ) -> torch.Tensor:
        """Single-rank reference: plain cross entropy over the local logits."""
        assert reduction == "sum"
        assert vocab_size == _VOCAB  # global vocabulary size is passed through
        assert mesh is self.mesh
        self.calls.append((local_logits, labels))
        return F.cross_entropy(
            local_logits, labels, ignore_index=ignore_index, reduction="sum"
        )


def _reference_loss(logits, labels, ignore_index=-100, num_items=None, shift_labels=None):
    """Pure-Torch equivalent of the production loss (single rank)."""
    if shift_labels is None:
        padded = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = padded[..., 1:].contiguous()
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_labels = shift_labels.reshape(-1)
    total = F.cross_entropy(
        flat_logits, flat_labels, ignore_index=ignore_index, reduction="sum"
    )
    if num_items is None:
        num_items = (flat_labels != ignore_index).sum().to(total.dtype)
    return total / num_items


class TestCausalLmLossParallel(unittest.TestCase):
    """Local-first production contract for the sharded causal LM loss."""

    def setUp(self) -> None:
        """Create the local CE recorder and patch distributed dependencies."""
        torch.manual_seed(0)
        self.mesh = object()  # sentinel: identity is all the contract needs
        self.recorder = _LocalCeRecorder(self.mesh)
        patches = mock.patch.multiple(
            loss_module,
            _get_loss_parallel_mesh=mock.Mock(return_value=self.mesh),
            DTensor=_NoDTensor,
            vocab_parallel_cross_entropy_local=self.recorder,
        )
        patches.start()
        self.addCleanup(patches.stop)

    def _run(self, logits, labels, **kwargs):
        return loss_module.causal_lm_loss_parallel(
            logits, labels, vocab_size=logits.shape[-1], **kwargs
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_full_logits_matches_reference(self):
        """Verify full-logits loss and gradient correctness.

        Feature: Local-first loss-parallel cross entropy.
        Description: Compute loss and gradients from a complete local logits tensor.
        Expectation: One CE call produces the same loss and gradients as the reference.
        """
        logits = torch.randn(2, 10, _VOCAB, requires_grad=True)
        labels = torch.randint(0, _VOCAB, (2, 10))
        loss = self._run(logits, labels)
        self.assertEqual(len(self.recorder.calls), 1)
        _NoDTensor.from_local.assert_not_called()
        ref_logits = logits.detach().clone().requires_grad_(True)
        reference = _reference_loss(ref_logits, labels)
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-4, atol=1e-5))

        loss.backward()
        reference.backward()
        self.assertTrue(
            torch.allclose(logits.grad, ref_logits.grad, rtol=1e-3, atol=1e-4)
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_full_token_dimension_uses_one_call(self):
        """Verify loss computation does not split the token dimension.

        Feature: Unchunked loss-parallel cross entropy.
        Description: Compute loss for a token dimension larger than the removed chunk size.
        Expectation: The full token dimension reaches CE in one FP32 call without a temporary DTensor.
        """
        tokens = 261
        logits = torch.randn(1, tokens, _VOCAB)
        labels = torch.randint(0, _VOCAB, (1, tokens))
        loss = self._run(logits, labels)
        self.assertEqual(len(self.recorder.calls), 1)
        local_logits, local_labels = self.recorder.calls[0]
        self.assertEqual(local_logits.shape[0], tokens)
        self.assertEqual(local_labels.shape[0], tokens)
        self.assertEqual(local_logits.dtype, torch.float32)
        _NoDTensor.from_local.assert_not_called()
        reference = _reference_loss(logits, labels)
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-4, atol=1e-4))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_ignore_index_excluded_from_denominator(self):
        """Verify ignored labels do not affect loss normalization.

        Feature: Ignore-index normalization.
        Description: Mark a prefix of labels with the default ignore index.
        Expectation: Ignored labels contribute to neither the loss sum nor denominator.
        """
        logits = torch.randn(1, 20, _VOCAB)
        labels = torch.randint(0, _VOCAB, (1, 20))
        labels[0, :7] = -100
        loss = self._run(logits, labels)
        reference = _reference_loss(logits, labels)
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-4, atol=1e-5))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_num_items_in_batch_is_the_denominator(self):
        """Verify explicit item-count normalization.

        Feature: Explicit loss denominator.
        Description: Supply ``num_items_in_batch`` with a value different from the token count.
        Expectation: The supplied item count replaces the inferred non-ignore token count.
        """
        logits = torch.randn(1, 20, _VOCAB)
        labels = torch.randint(0, _VOCAB, (1, 20))
        loss = self._run(logits, labels, num_items_in_batch=7)
        reference = _reference_loss(logits, labels, num_items=7)
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-4, atol=1e-5))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_explicit_shift_labels_skip_padding(self):
        """Verify explicit shifted labels bypass internal shifting.

        Feature: Pre-shifted causal labels.
        Description: Supply labels and a distinct ``shift_labels`` tensor.
        Expectation: The loss uses ``shift_labels`` verbatim without padding or shifting labels.
        """
        logits = torch.randn(1, 20, _VOCAB)
        labels = torch.randint(0, _VOCAB, (1, 20))
        shift_labels = torch.randint(0, _VOCAB, (1, 20))
        loss = self._run(logits, labels, shift_labels=shift_labels)
        reference = _reference_loss(logits, labels, shift_labels=shift_labels)
        self.assertTrue(torch.allclose(loss, reference, rtol=1e-4, atol=1e-5))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_targets_raise(self):
        """Verify empty target tensors fail fast.

        Feature: Empty-target validation.
        Description: Pass logits and labels with a zero-length token dimension.
        Expectation: Loss calculation raises a descriptive ``ValueError``.
        """
        logits = torch.empty(1, 0, _VOCAB)
        labels = torch.empty(1, 0, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "at least one target token"):
            self._run(logits, labels)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_missing_tp_mesh_raises(self):
        """Verify the TP mesh is required.

        Feature: Loss-parallel mesh validation.
        Description: Compute a loss while the loss-parallel context has no TP mesh.
        Expectation: Loss calculation raises a descriptive ``ValueError``.
        """
        loss_module._get_loss_parallel_mesh.return_value = None  # pylint: disable=protected-access
        logits = torch.randn(1, 4, _VOCAB)
        labels = torch.randint(0, _VOCAB, (1, 4))
        with self.assertRaisesRegex(ValueError, "requires a TP mesh"):
            self._run(logits, labels)


if __name__ == "__main__":
    unittest.main()
