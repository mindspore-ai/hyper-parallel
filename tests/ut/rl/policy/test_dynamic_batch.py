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
"""CPU contracts for dynamic Actor scheduling and unchanged optimizer objectives."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset.contracts import ExperienceBatch
from rl.roles.policy.actor import Actor
from rl.roles.policy.dynamic_batch import equalize_batch_count, plan_token_batches


class TokenPolicy(torch.nn.Module):
    """Small policy exposing either selected probabilities or standard dense logits."""

    def __init__(self, *, logits: bool = False) -> None:
        """Keep both Actor output contracts on the same differentiable scalars."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.03))
        self.bias = torch.nn.Parameter(torch.tensor(-0.7))
        self.supports_token_log_probs = not logits
        self.supports_token_log_probs_mask = not logits
        self.input_shapes = []

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor | dict:
        """Record head inputs and return the selected output contract."""
        self.input_shapes.append(tuple(input_ids.shape))
        if self.supports_token_log_probs:
            return self.weight * input_ids[:, 1:].float() + self.bias
        vocabulary = torch.arange(13).float()
        features = input_ids.float().unsqueeze(-1) * vocabulary / 13
        return {"logits": self.weight * features + self.bias * vocabulary}


def make_experience(lengths: list[int]) -> ExperienceBatch:
    """Create variable-length rows with distinct tokens, targets and identities."""
    width = max(lengths)
    sequences = torch.arange(len(lengths) * width).reshape(len(lengths), width) % 13
    attention = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
    actions = attention.clone()
    actions[:, :2] = False
    old = 0.03 * sequences[:, 1:].float() - 0.7
    return ExperienceBatch(
        trajectories=tuple(SimpleNamespace(token_ids=sequences[index, :length], metadata={}, worker_policy_version=None)
                           for index, length in enumerate(lengths)),
        sequences=sequences, attention_mask=attention, action_mask=actions,
        rewards=torch.arange(len(lengths)).float(), old_log_probs=old,
        responses=tuple(str(index) for index in range(len(lengths))), generation_seconds=0.0,
        advantages=torch.sin(sequences[:, 1:].float()), reference_log_probs=old - 0.1,
    )


def test_equalization_splits_without_duplicates_or_empty_batches():
    """DP call matching must preserve samples and already-established memory bounds."""
    original = [[0, 1, 2], [3, 4]]
    groups = equalize_batch_count(original, 4)
    assert len(groups) == 4
    assert all(groups)
    assert sorted(index for group in groups for index in group) == list(range(5))
    assert original == [[0, 1, 2], [3, 4]]
    with pytest.raises(ValueError, match="empty"):
        equalize_batch_count(original, 6)


def make_actor(budget: int | None = None, mini_batch: int = 4, epochs: int = 1) -> Actor:
    """Build a CPU Actor sharing the production loss and update loop."""
    model = TokenPolicy()
    return Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean", "kl_coef": 0.01}),
                 2, optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False), device=torch.device("cpu"),
                 response_mini_batch_size=mini_batch, update_epochs=epochs, max_tokens_per_micro_batch=budget)


class CausalLogitsPolicy(torch.nn.Module):
    """Sequence-coupled CPU model to test causal context and padding semantics."""

    def __init__(self) -> None:
        """Build attention and a vocabulary projection without accelerator dependencies."""
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 8)
        self.attention = torch.nn.MultiheadAttention(8, 2, batch_first=True)
        self.projection = torch.nn.Linear(8, 13)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Attend only to the same row's valid causal prefix."""
        hidden = self.embedding(input_ids)
        causal = torch.ones(input_ids.shape[1], input_ids.shape[1], dtype=torch.bool).triu(1)
        hidden, _ = self.attention(hidden, hidden, hidden, attn_mask=causal,
                                   key_padding_mask=~attention_mask.bool(), need_weights=False)
        return {"logits": self.projection(hidden)}


@pytest.mark.parametrize("budget", [4, 24])
def test_dynamic_update_preserves_causal_attention_context_and_loss(budget):
    """Dynamic forwards restore rows without gradients, then updates preserve the causal objective."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        models = [CausalLogitsPolicy(), CausalLogitsPolicy()]
    models[1].load_state_dict(models[0].state_dict())
    actors = [Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}), 2,
                    optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False), device=torch.device("cpu"),
                    response_mini_batch_size=4, update_epochs=2, max_tokens_per_micro_batch=budget)
              for model, budget in zip(models, [None, budget])]
    experience = make_experience([3, 8, 4, 6, 8, 4])
    forward_input = replace(experience, old_log_probs=None, reference_log_probs=None, advantages=None)
    old = actors[0].compute_log_probs(forward_input)
    actors[1]._log_prob_max_tokens_per_micro_batch = 16
    before = [parameter.detach().clone() for parameter in models[1].parameters()]
    actual = actors[1].compute_log_probs(forward_input)
    mask = experience.attention_mask[:, 1:].bool()
    torch.testing.assert_close(actual[mask], old[mask], atol=1e-6, rtol=1e-6)
    assert actual.shape == old.shape and not actual.requires_grad and actors[1].training
    for expected, parameter in zip(before, models[1].parameters()):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        assert parameter.grad is None
    experience = replace(experience, old_log_probs=old, reference_log_probs=old)
    with patch("rl.roles.policy.actor.hsdp_sync_stream"), patch(
        "rl.roles.policy.actor.clip_grad_norm_", torch.nn.utils.clip_grad_norm_,
    ):
        fixed, dynamic = [actor.update(experience) for actor in actors]
    assert dynamic.optimizer_steps == fixed.optimizer_steps == 4
    assert dynamic.total_loss == pytest.approx(fixed.total_loss, abs=2e-7)
    for expected, actual in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)


class TestOversizedSequences(TestCase):
    """Long rows remain complete while ordinary groups retain their token budget."""

    def test_plan_covers_normal_and_oversized_rows(self) -> None:
        """Do not raise the grouping budget for normal rows or lose any long row."""
        for lengths in ([900, 2154, 900, 2208, 3096, 2048], [2154, 2208], [3096], [8, 3, 7, 4], [2048]):
            with self.subTest(lengths=lengths):
                groups = plan_token_batches(lengths, 2048)
                self.assertEqual(sorted(index for group in groups for index in group), list(range(len(lengths))))
                for group in groups:
                    longest = max(lengths[index] for index in group)
                    if longest > 2048:
                        self.assertEqual(len(group), 1)
                    else:
                        self.assertLessEqual(len(group) * longest, 2048)
                self.assertEqual(groups, plan_token_batches(lengths, 2048))
        self.assertEqual(plan_token_batches([900, 2154, 900, 2208], 2048), [[3], [1], [0, 2]])

    def test_old_and_reference_keep_complete_long_rows_and_probability_order(self) -> None:
        """Both frozen and trainable roles preserve valid values through oversized calls."""
        experience = make_experience([900, 2154, 900, 2208, 3096])
        for trainable in (False, True):
            with self.subTest(trainable=trainable):
                model = TokenPolicy(logits=True)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if trainable else None
                actor = Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}),
                              1, optimizer=optimizer,
                              device=torch.device("cpu"))
                expected = actor.compute_log_probs(experience)
                model.input_shapes.clear()
                actor._log_prob_max_tokens_per_micro_batch = 2048
                actual = actor.compute_log_probs(experience)
                mask = experience.attention_mask[:, 1:].bool()
                torch.testing.assert_close(actual[mask], expected[mask], atol=1e-6, rtol=1e-6)
                self.assertEqual(model.input_shapes, [(1, 3096), (1, 2208), (1, 2154), (2, 900)])
                self.assertFalse(actual.requires_grad)


def test_dp_planning_preserves_membership_and_rejects_empty_calls() -> None:
    """Shared forward/update planning splits normal groups while keeping oversized rows whole."""
    actor = make_actor(8)
    actor._dp_size = 2
    actor._dp_group_info = SimpleNamespace(group="dp")
    experience = make_experience([9, 2, 2, 2, 4, 4])
    for peer_count in (3, 5):
        def gather(output: list, local: list, group: str, *, peer_count: int = peer_count) -> None:
            """Supply two peer optimizer minibatches without real collectives."""
            assert group == "dp"
            output[:] = [local, [(peer_count, 4), (1, 2)]]
        with patch("rl.roles.policy.actor.dist.all_gather_object", side_effect=gather):
            if peer_count == 5:
                with pytest.raises(ValueError, match="empty batches"):
                    actor._plan_dynamic_update(experience)
            else:
                plans = actor._plan_dynamic_update(experience)
                assert len(plans[0]) == 3 and [0] in plans[0]
                assert sorted(sum(plans[0], [])) == [0, 1, 2, 3]
                assert sorted(sum(plans[1], [])) == [4, 5]
