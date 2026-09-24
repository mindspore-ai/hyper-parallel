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
"""Public MTP composition and gradient contracts."""

import copy
import unittest

import torch
from torch import nn

from hyper_parallel.components.modules.mtp import DeepseekV3MTP, MultiTokenPrediction, MultiTokenPredictionLayer


from tests.common.mark_utils import arg_mark


class TestMultiTokenPrediction(unittest.TestCase):
    """Exercise injected non-DeepSeek components without distributed state."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_composition_and_gradient(self):
        """Both trunk states and future embeddings receive the expected gradients.

        Feature: mtp.
        Description: Both trunk states and future embeddings receive the expected gradients.
        Expectation: The asserted values and state transitions hold.
        """
        projection = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[1., 0., 2., 0.], [0., 1., 0., 2.]]))
        layer = MultiTokenPredictionLayer(embedding_norm=nn.Identity(), hidden_norm=nn.Identity(),
                                         projection=projection, decoder=nn.Identity(), output_norm=nn.Identity())
        hidden = torch.tensor([[[3., 4.]]], requires_grad=True)
        embedding = torch.tensor([[[5., 6.]]], requires_grad=True)
        result = layer(hidden, embedding)
        torch.testing.assert_close(result, hidden + 2 * embedding)
        result.sum().backward()
        torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
        torch.testing.assert_close(embedding.grad, 2 * torch.ones_like(embedding))
        container = MultiTokenPrediction([layer])
        self.assertIn("layers.0.eh_proj.weight", container.state_dict())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_empty_depth_registers_no_parameters(self):
        """Models without MTP do not allocate a dummy prediction layer.

        Feature: mtp.
        Description: Models without MTP do not allocate a dummy prediction layer.
        Expectation: The asserted values and state transitions hold.
        """
        self.assertEqual(dict(MultiTokenPrediction([]).named_parameters()), {})


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_fusion_preserves_input_precision(self):
        """Preserve precision in the public fusion block.

        Feature: MTP fusion.
        Description: FP32 values survive fusion.
        Expectation: No BF16 rounding.
        """
        layer = MultiTokenPredictionLayer(embedding_norm=nn.Identity(), hidden_norm=nn.Identity(),
                                         projection=nn.Identity(), decoder=nn.Identity(), output_norm=nn.Identity())
        hidden = torch.tensor([[[1.001, 2.003]]], requires_grad=True)
        embedding = torch.tensor([[[3.005, 4.007]]], requires_grad=True)
        result = layer(hidden, embedding)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.cat((hidden, embedding), dim=-1), rtol=0, atol=0)
        self.assertFalse(torch.equal(result, result.bfloat16().float()))


class TestDeepseekV3MTP(unittest.TestCase):
    """Verify the public algorithm without any model-family adapter."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_two_depth_objective_and_shared_gradients(self):
        """Feature: Reusable MTP.

        Description: Compare two depths to an independently expanded V3 formula.
        Expectation: Loss and every gradient match; head and embedding are shared.
        """
        torch.manual_seed(23)
        mtp = DeepseekV3MTP(hidden_size=4, num_layers=2,
                           decoder_factory=lambda index: nn.Linear(4, 4, bias=False),
                           output_norm_factory=nn.LayerNorm)
        embedding, head = nn.Embedding(13, 4), nn.Linear(4, 13, bias=False)
        hidden = torch.randn(1, 5, 4, requires_grad=True)
        reference, ref_embedding, ref_head = copy.deepcopy((mtp, embedding, head))
        ref_hidden = hidden.detach().clone().requires_grad_()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[2, 3, 4, 5, 6]])
        mask = torch.tensor([[0., 1., 1., 1., 1.]])
        result = mtp(hidden, tokens, embedding=embedding, head=head,
                     labels=labels, loss_mask=mask, loss_factor=0.3)
        loss, state = 0, ref_hidden
        for depth, layer in enumerate(reference.layers, 1):
            future = torch.zeros_like(tokens)
            future[:, :-depth] = tokens[:, depth:]
            state = layer.transformer_layer(layer.eh_proj(torch.cat(
                (layer.hnorm(state), layer.enorm(ref_embedding(future))), dim=-1)))
            logits = ref_head(layer.final_layernorm(state))[:, :-depth]
            ce = nn.functional.cross_entropy(logits.flatten(0, 1), labels[:, depth:].flatten(), reduction="none")
            loss = loss + (ce * mask[:, depth:].flatten()).sum() / tokens.numel() * 0.15
        torch.testing.assert_close(result.loss, loss, rtol=0, atol=0)
        torch.testing.assert_close(result.hidden_states, state, rtol=0, atol=0)
        result.loss.backward()
        loss.backward()
        torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0, atol=0)
        for actual, expected in zip((mtp, embedding, head), (reference, ref_embedding, ref_head)):
            for param, ref_param in zip(actual.parameters(), expected.parameters()):
                self.assertIsNotNone(param.grad)
                torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-6, atol=1e-7)
        self.assertFalse(any(id(param) in {id(p) for p in mtp.parameters()}
                             for param in list(embedding.parameters()) + list(head.parameters())))
        self.assertEqual(len(result.depth_losses), 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_shifted_targets_and_exhausted_depths(self):
        """Feature: Target alignment.

        Description: Request more MTP depths than tokens with ignored labels.
        Expectation: Shifts never wrap, masks zero invalid tails, losses stay finite.
        """
        mtp = DeepseekV3MTP(hidden_size=2, num_layers=3, decoder_factory=lambda index: nn.Identity())
        seen = []

        def _capture(logits, labels, mask):
            seen.append((labels.tolist(), mask.tolist()))
            return mtp.execution.token_loss(logits, labels, mask)

        result = mtp(torch.randn(1, 2, 2), torch.tensor([[1, 2]]), embedding=nn.Embedding(4, 2),
                     head=nn.Linear(2, 4), labels=torch.tensor([[2, -100]]), loss_mask=torch.ones(1, 2),
                     loss_factor=0.3, loss_fn=_capture)
        self.assertEqual(seen, [([[-100, 0]], [[1., 0.]]), ([[0, 0]], [[0., 0.]]),
                                ([[0, 0]], [[0., 0.]])])
        self.assertEqual(result.loss.item(), 0.)
        result.loss.backward()
        for param in mtp.parameters():
            self.assertTrue(torch.isfinite(param.grad).all())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_zero_depth_and_invalid_contract(self):
        """Feature: Optional MTP.

        Description: Execute a zero-depth model and reject mismatched targets.
        Expectation: No parameters or hidden-state changes and a descriptive error.
        """
        mtp = DeepseekV3MTP(hidden_size=4, num_layers=0, decoder_factory=lambda index: nn.Identity())
        hidden, tokens = torch.randn(1, 3, 4), torch.ones(1, 3, dtype=torch.long)
        options = {"embedding": nn.Embedding(4, 4), "head": nn.Linear(4, 4), "labels": tokens,
                   "loss_mask": torch.ones(1, 3), "loss_factor": 0.3}
        result = mtp(hidden, tokens, **options)
        self.assertIs(result.hidden_states, hidden)
        self.assertEqual(result.loss.item(), 0.)
        self.assertEqual(list(mtp.parameters()), [])
        options["labels"] = tokens[:, :2]
        with self.assertRaisesRegex(ValueError, "must match"):
            mtp(hidden, tokens, **options)
