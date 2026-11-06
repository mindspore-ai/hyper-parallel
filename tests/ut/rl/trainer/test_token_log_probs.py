# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Role binding and generic token-probability dispatch preserve legacy execution."""

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from rl.algorithm import build_algorithm
from rl.data.contracts import ExperienceBatch
from rl.workers.engine.model import build_role_model
from rl.workers.actor import Actor


class _ProbabilityModel(torch.nn.Module):
    """Expose the generic return protocol without owning model-family mathematics."""

    supports_token_log_probs = True

    def __init__(self, value: torch.Tensor) -> None:
        """Keep the test vector and observed calls on CPU."""
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Record whether the normal module call requested the selected output."""
        self.calls.append(kwargs)
        return self.value


class TestTokenLogProbs(unittest.TestCase):
    """Check the RL boundary independently of the actual HF mathematical tests."""

    @staticmethod
    def _actor(model: torch.nn.Module) -> Actor:
        """Construct the original GRPO role with its required token-mean contract."""
        algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
        return Actor(model, algorithm, micro_batch_size=1)

    def test_role_build_binds_text_model_before_reference_freeze(self) -> None:
        """Use the original public target and preserve parameter objects for both roles."""
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model = torch.nn.Module()
                model.config = SimpleNamespace(model_type="qwen3_5_text")
                model.model = torch.nn.Identity()
                model.lm_head = torch.nn.Linear(4, 8)
                parameters = list(model.parameters())
                target = Mock()
                target.build.return_value = model
                runtime = SimpleNamespace(model=target, activation_checkpoint=SimpleNamespace(mode="full"), peft=None)
                setup = object()
                result = build_role_model(runtime, setup, frozen=frozen)
                self.assertIs(result, model)
                target.build.assert_called_once_with(
                    distributed_setup=setup, activation_checkpoint="full", peft_config=None,
                )
                self.assertTrue(model.supports_token_log_probs)
                self.assertTrue(all(left is right for left, right in zip(parameters, model.parameters())))
                self.assertTrue(all(parameter.requires_grad is not frozen for parameter in model.parameters()))

    def test_actor_uses_normal_module_call(self) -> None:
        """The established consistency path wins; otherwise the generic capability returns its vector."""
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        expected = torch.tensor([[-1.0, -2.0, -3.0]])
        model = _ProbabilityModel(expected)
        actor = self._actor(model)
        hooks = []
        model.register_forward_pre_hook(lambda _module, _inputs: hooks.append(True))
        with nullcontext():
            self.assertIs(actor.sequence_log_probs(ids, mask), expected)
        self.assertEqual(len(hooks), 1)
        self.assertTrue(model.calls[0]["return_token_log_probs"])
        self.assertFalse(model.calls[0]["use_cache"])
        self.assertIs(model.calls[0]["input_ids"], ids)

    def test_actor_rejects_invalid_selected_output(self) -> None:
        """Reject a lossy dtype or unshifted shape instead of falling back to dense logits."""
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        for value in (torch.zeros(1, 3, dtype=torch.bfloat16), torch.zeros(1, 4)):
            with self.subTest(dtype=value.dtype, shape=value.shape):
                actor = self._actor(_ProbabilityModel(value))
                with nullcontext():
                    with self.assertRaisesRegex(ValueError, "FP32"):
                        actor.sequence_log_probs(ids, mask)

    def test_action_mask_is_only_sent_to_models_declaring_mask_support(self) -> None:
        """Legacy models retain their complete-vector interface; capable models receive the actual loss mask."""
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        actions = torch.tensor([[False, True, True]])
        for capable in (False, True):
            model = _ProbabilityModel(torch.zeros(1, 3))
            model.supports_token_log_probs_mask = capable
            actor = self._actor(model)
            with nullcontext():
                actor.sequence_log_probs(ids, mask, actions)
                batch = ExperienceBatch(trajectories=(), sequences=ids, attention_mask=mask,
                                        action_mask=torch.cat((torch.zeros(1, 1, dtype=torch.bool), actions), dim=1),
                                        rewards=torch.zeros(1), old_log_probs=torch.zeros(1, 3), responses=("",),
                                        generation_seconds=0)
                actor.compute_log_probs(batch)
            if capable:
                torch.testing.assert_close(model.calls[0]["token_log_probs_mask"], actions)
            else:
                self.assertNotIn("token_log_probs_mask", model.calls[0])
