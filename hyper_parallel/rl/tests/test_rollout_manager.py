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
"""CPU tests for training-group and batched validation generation."""

from typing import Any, Optional

import pytest
import torch

from rl.contracts import Message, PromptRecord
from rl.roles.rollout import PolicySnapshot
from rl.roles.rollout.hyper_infer import HyperGenerationEngine
from rl.roles.rollout.worker import RolloutManager


class FakeActor(torch.nn.Module):
    """Generate one answer token and EOS for every prompt."""

    def __init__(self) -> None:
        """Initialize generation call tracking."""
        super().__init__()
        self.generation_config: Any = None

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: Any,
    ) -> torch.Tensor:
        """Append deterministic per-row response IDs."""
        del attention_mask
        self.generation_config = generation_config
        answer_ids = torch.arange(
            101,
            101 + input_ids.shape[0],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        eos_ids = torch.full_like(answer_ids, 2)
        responses = torch.stack((answer_ids, eos_ids), dim=-1)
        return torch.cat((input_ids, responses), dim=-1)

    @staticmethod
    def response_log_probs(
        sequences: torch.Tensor,
        prompt_length: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return deterministic frozen-policy log-probabilities."""
        del attention_mask
        responses = sequences[:, prompt_length:]
        return -responses.float() / 100.0


class FakeTokenizer:
    """Decode answer IDs into strict numeric final-answer strings."""

    @staticmethod
    def decode(token_ids: list[int], skip_special_tokens: bool) -> str:
        """Map token 101 to answer 1, token 102 to answer 2, and so on."""
        assert skip_special_tokens is True
        return f"Reasoning #### {token_ids[0] - 100}"


def test_batched_greedy_validation_uses_per_prompt_ground_truths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify batched n=1 evaluation scores each prompt against its own answer."""
    monkeypatch.setattr(HyperGenerationEngine, "_synchronize_device", staticmethod(lambda: None))
    actor = FakeActor()
    actor.train()
    engine = HyperGenerationEngine(actor)
    engine.update_weights(PolicySnapshot(3, "fake", actor))
    manager = RolloutManager(
        engine=engine,
        tokenizer=FakeTokenizer(),
        environment_name="gsm8k",
        num_return_sequences=1,
        max_turns=1,
        max_observation_tokens=0,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=0,
        eos_token_id=2,
        do_sample=False,
    )
    prompts = (
        PromptRecord(
            "p0",
            (Message("user", "question 0"),),
            ground_truth="1",
            metadata={"input_ids": torch.tensor([11, 12])},
        ),
        PromptRecord(
            "p1",
            (Message("user", "question 1"),),
            ground_truth="999",
            metadata={"input_ids": torch.tensor([21, 22, 23])},
        ),
    )
    rollout = manager.generate(prompts, policy_version=3)
    assert rollout.responses == ("Reasoning #### 1", "Reasoning #### 2")
    assert rollout.rewards.tolist() == [1.0, 0.0]
    assert rollout.sequences.tolist() == [
        [11, 12, 101, 0],
        [21, 22, 23, 102],
    ]
    assert rollout.action_mask.tolist() == [
        [False, False, True, False],
        [False, False, False, True],
    ]
    assert rollout.loss_action_mask.tolist() == [
        [False, True, False],
        [False, False, True],
    ]
    assert rollout.old_log_probs is None
    assert [trajectory.policy_version for trajectory in rollout.trajectories] == [3, 3]
    assert [
        trajectory.metadata["extracted_answer"]
        for trajectory in rollout.trajectories
    ] == ["1", "2"]
    assert actor.generation_config.do_sample is False
    assert actor.training is True


def test_training_group_repeats_one_prompt_and_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify n>1 training responses retain group ordering and shared scoring."""
    monkeypatch.setattr(HyperGenerationEngine, "_synchronize_device", staticmethod(lambda: None))
    actor = FakeActor()
    engine = HyperGenerationEngine(actor)
    engine.update_weights(PolicySnapshot(4, "fake", actor))
    manager = RolloutManager(
        engine=engine,
        tokenizer=FakeTokenizer(),
        environment_name="gsm8k",
        num_return_sequences=2,
        max_turns=1,
        max_observation_tokens=0,
        max_new_tokens=2,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=0,
        eos_token_id=2,
        collect_old_log_probs=True,
    )
    prompt = PromptRecord(
        "p0",
        (Message("user", "question"),),
        ground_truth="2",
        metadata={"input_ids": torch.tensor([11, 12])},
    )
    rollout = manager.generate((prompt,), policy_version=4)
    assert rollout.rewards.tolist() == [0.0, 1.0]
    assert rollout.sequences.tolist() == [[11, 12, 101], [11, 12, 102]]
    expected_old_log_probs = torch.tensor([[0.0, -1.01], [0.0, -1.02]])
    assert rollout.old_log_probs is not None
    assert torch.allclose(rollout.old_log_probs, expected_old_log_probs), (
        f"Unexpected rollout old log-probabilities: "
        f"expected={expected_old_log_probs}, got={rollout.old_log_probs}"
    )
    assert [trajectory.group_id for trajectory in rollout.trajectories] == ["p0", "p0"]
    assert [
        trajectory.metadata["extracted_answer"]
        for trajectory in rollout.trajectories
    ] == ["1", "2"]
