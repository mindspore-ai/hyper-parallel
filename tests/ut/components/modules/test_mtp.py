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
"""MTP fusion, causal training, weight sharing and adapter contracts."""

from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from hyper_parallel.components.modules import DeepseekV3MTP, DeepseekV3MTPLayer
from hyper_parallel.components.losses import deepseek_v3_mtp_loss


class TinyCausalDecoder(nn.Module):
    """Small actual attention block to test causality and gradient propagation."""

    def __init__(self, hidden_size: int = 4) -> None:
        """Create learnable QKV and output projections."""
        super().__init__()
        self.projection = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, states: torch.Tensor, *, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Use causal attention; optionally exercise model-owned position metadata."""
        if position_ids is not None:
            if position_ids.shape != states.shape[:2]:
                raise ValueError("position_ids must match the current depth")
            states = states + position_ids.unsqueeze(-1).to(states) * 0.01
        query, key, value = self.projection(states).chunk(3, dim=-1)
        scores = (query @ key.transpose(-1, -2)) / query.shape[-1] ** 0.5
        future_mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=states.device).triu(1)
        attention = scores.masked_fill(future_mask, float("-inf")).softmax(-1) @ value
        return states + self.output(attention)


class TestMTP(unittest.TestCase):
    """Test behavior rather than only the module wiring."""

    def setUp(self) -> None:
        """Build two independent MTP depths and shared model parameters."""
        torch.manual_seed(42)
        self.embedding = nn.Embedding(13, 4)
        self.head = nn.Linear(4, 13, bias=False)
        self.model = DeepseekV3MTP([DeepseekV3MTPLayer(4, TinyCausalDecoder()) for _ in range(2)])
        self.tokens = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.hidden = torch.randn(2, 5, 4, requires_grad=True)

    def test_embedding_first_and_projection_conversion(self) -> None:
        """Swapping input blocks requires swapping weight columns, not norm roles."""
        layer = DeepseekV3MTPLayer(2, nn.Identity(), norm_factory=lambda _: nn.Identity())
        with torch.no_grad():
            layer.eh_proj.weight.copy_(torch.tensor([[1., 2., 10., 20.], [3., 4., 30., 40.]]))
        hidden = torch.tensor([[[1., 2.]]])
        embedding = torch.tensor([[[3., 4.]]])
        torch.testing.assert_close(layer(hidden, embedding), torch.tensor([[[61., 135.]]]))
        swapped_weight = torch.cat(layer.eh_proj.weight.chunk(2, dim=1)[::-1], dim=1)
        converted = torch.cat((hidden, embedding), dim=-1) @ swapped_weight.T
        torch.testing.assert_close(layer(hidden, embedding), converted)

    def test_depths_use_future_ground_truth(self) -> None:
        """Depth 1 uses t[i+1], depth 2 uses t[i+2], without generated tokens."""
        observed = []
        handle = self.embedding.register_forward_pre_hook(lambda _, args: observed.append(args[0].clone()))
        try:
            output = self.model(self.hidden, self.tokens, embedding=self.embedding, head=self.head)
        finally:
            handle.remove()
        torch.testing.assert_close(observed[0], self.tokens[:, 1:])
        torch.testing.assert_close(observed[1], self.tokens[:, 2:])
        self.assertEqual([tuple(x.shape) for x in output.logits], [(2, 4, 13), (2, 3, 13)])

    def test_uses_post_normalized_state_for_next_depth(self) -> None:
        """A changed post_norm scale propagates to the next depth's input."""
        with torch.no_grad():
            self.model.layers[0].post_norm.weight.copy_(torch.tensor([0.5, 1., 2., 3.]))
        observed = []
        handle = self.model.layers[1].register_forward_pre_hook(lambda _, args: observed.append(args[0]))
        try:
            output = self.model(self.hidden, self.tokens, embedding=self.embedding)
        finally:
            handle.remove()
        torch.testing.assert_close(observed[0], output.hidden_states[0][:, :-1])

    def test_future_tokens_cannot_leak_into_earlier_prediction(self) -> None:
        """Perturb the last token and trunk position without changing earlier logits."""
        baseline = self.model(self.hidden, self.tokens, embedding=self.embedding, head=self.head)
        tokens = self.tokens.clone()
        tokens[:, -1] = 12
        hidden = self.hidden.detach().clone()
        hidden[:, -1] += 100
        changed = self.model(hidden, tokens, embedding=self.embedding, head=self.head)
        for original, modified in zip(baseline.logits, changed.logits):
            torch.testing.assert_close(original[:, :-1], modified[:, :-1])

    def test_shared_head_embedding_and_trunk_receive_gradients(self) -> None:
        """All objectives update the actual shared parameters, not detached copies."""
        output = self.model(self.hidden, self.tokens, embedding=self.embedding, head=self.head)
        labels = (self.tokens + 1) % 13
        loss = deepseek_v3_mtp_loss(output.logits, labels)
        loss.backward()
        for tensor in [self.hidden, self.embedding.weight, self.head.weight, *self.model.parameters()]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        registered = {id(param) for param in self.model.parameters()}
        self.assertNotIn(id(self.embedding.weight), registered)
        self.assertNotIn(id(self.head.weight), registered)

    def test_hidden_only_path_matches_shared_head_path(self) -> None:
        """An adapter may compute chunk loss without materializing logits here."""
        hidden_output = self.model(self.hidden, self.tokens, embedding=self.embedding)
        full_output = self.model(self.hidden, self.tokens, embedding=self.embedding, head=self.head)
        self.assertEqual(hidden_output.logits, ())
        for states, logits in zip(hidden_output.hidden_states, full_output.logits):
            torch.testing.assert_close(self.head(states), logits)

    def test_per_depth_position_metadata(self) -> None:
        """Do not reuse an unsliced S-position tensor for the shorter depths."""
        kwargs = [{"position_ids": torch.arange(5)[k:].expand(2, -1)} for k in (1, 2)]
        output = self.model(self.hidden, self.tokens, embedding=self.embedding, decoder_kwargs=kwargs)
        self.assertEqual(output.hidden_states[-1].shape, (2, 3, 4))

    def test_layer_norms_and_depth_parameters_are_independent(self) -> None:
        """Only the externally supplied embedding/head should be shared."""
        first, second = self.model.layers
        self.assertIsNot(first.enorm.weight, first.hnorm.weight)
        self.assertIsNot(first.hnorm.weight, first.post_norm.weight)
        self.assertIsNot(first.eh_proj.weight, second.eh_proj.weight)

    def test_checkpoint_roundtrip(self) -> None:
        """Standalone MTP state dict reload preserves the prediction."""
        restored = copy.deepcopy(self.model)
        restored.load_state_dict(self.model.state_dict())
        baseline = self.model(self.hidden, self.tokens, embedding=self.embedding, head=self.head)
        reloaded = restored(self.hidden, self.tokens, embedding=self.embedding, head=self.head)
        for expected, actual in zip(baseline.logits, reloaded.logits):
            torch.testing.assert_close(actual, expected)

    def test_invalid_inputs(self) -> None:
        """Reject incompatible depth, shape and metadata before running decoders."""
        with self.assertRaisesRegex(ValueError, "at least one"):
            DeepseekV3MTP([])
        for hidden_size in (0, -1, True, 2.5):
            with self.subTest(hidden_size=hidden_size), self.assertRaises(ValueError):
                DeepseekV3MTPLayer(hidden_size, nn.Identity())
        for eps in (0., -1., float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "positive and finite"):
                DeepseekV3MTPLayer(4, nn.Identity(), rms_norm_eps=eps)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            self.model.layers[0](self.hidden, self.hidden[:, :-1])
        with self.assertRaisesRegex(ValueError, "aligned"):
            self.model(self.hidden, self.tokens[:, :-1], embedding=self.embedding)
        with self.assertRaisesRegex(ValueError, "greater than"):
            self.model(self.hidden[:, :2], self.tokens[:, :2], embedding=self.embedding)
        with self.assertRaisesRegex(ValueError, "one mapping"):
            self.model(self.hidden, self.tokens, embedding=self.embedding, decoder_kwargs=[{}])
