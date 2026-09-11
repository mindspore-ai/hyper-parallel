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
"""CPU unit tests for trajectory batching and algorithm target preparation."""
# HyperParallel-RL CPU tests intentionally exercise the verified Torch runtime.
# pylint: disable=forbidden-backend-import

import math

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset import ExperiencePreparer
from rl.dataset.batch_builder import build_experience_batch
from rl.dataset.contracts import Trajectory, Turn
from rl.roles.rollout.base import GenerationSettings


def _settings() -> GenerationSettings:
    """Return deterministic settings that collect rollout log-probabilities."""
    return GenerationSettings(3, 1.0, 1.0, 0, True, 0, 2, collect_log_probs=True)


def _trajectory(
    index: int,
    group_id: str,
    tokens: list[int],
    actions: list[bool],
    reward: float,
) -> Trajectory:
    """Build one committed trajectory with an assistant and optional observation."""
    action_start = actions.index(True)
    turns = (
        Turn("user", f"prompt-{index}", 0, action_start, False),
        Turn("assistant", f"answer-{index}", action_start, len(tokens), True),
    )
    return Trajectory(
        trajectory_id=f"trajectory-{index}",
        prompt_id=str(index),
        group_id=group_id,
        policy_version=2,
        turns=turns,
        token_ids=torch.tensor(tokens),
        attention_mask=torch.ones(len(tokens), dtype=torch.bool),
        action_mask=torch.tensor(actions),
        rollout_log_probs=torch.arange(len(tokens) - 1, dtype=torch.float32) + index,
        reward=reward,
        reward_components={"task": reward},
        done=True,
        truncated=False,
        terminal_reason="completed",
        worker_policy_version=2,
        worker_policy_fingerprint="digest-v2",
    )


def test_trajectories_build_padded_experience_contract() -> None:
    """Variable trajectories preserve action/logprob order and committed identity."""
    first = _trajectory(0, "group", [1, 10, 11], [False, True, True], 0.0)
    second = _trajectory(1, "group", [2, 20, 21, 22], [False, True, True, True], 1.0)

    batch = build_experience_batch(
        (first, second),
        generation_seconds=1.5,
        settings=_settings(),
        metadata={"source": "ut"},
    )

    assert batch.sequences.tolist() == [[1, 10, 11, 0], [2, 20, 21, 22]]
    assert batch.attention_mask.tolist() == [[True, True, True, False], [True, True, True, True]]
    assert batch.action_mask.tolist() == [[False, True, True, False], [False, True, True, True]]
    torch.testing.assert_close(
        batch.old_log_probs,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]]),
    )
    assert batch.responses == ("answer-0", "answer-1")
    assert batch.worker_policy_version == 2
    assert batch.worker_policy_fingerprint == "digest-v2"
    assert batch.metadata == {"source": "ut", "generated_action_tokens": 5}


@pytest.mark.parametrize("algorithm_name", ["grpo", "ppo"])
def test_experience_preparer_builds_algorithm_required_targets(algorithm_name: str) -> None:
    """GRPO and PPO populate only their declared detached training targets."""
    trajectories = tuple(
        _trajectory(
            index,
            "group-a" if index < 2 else "group-b",
            [1, 10 + index, 2],
            [False, True, True],
            (0.0, 2.0, 3.0, 1.0)[index],
        )
        for index in range(4)
    )
    rollout = build_experience_batch(
        trajectories,
        generation_seconds=1.0,
        settings=_settings(),
        metadata={},
    )
    algorithm = build_algorithm({
        "name": algorithm_name, "loss_aggregation": "token-mean",
        "gamma": 0.9, "gae_lambda": 0.8,
    })
    reference = (-torch.arange(1, 9, dtype=torch.float32).reshape(4, 2) / 10).requires_grad_()
    values = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                          requires_grad=True) if algorithm_name == "ppo" else None

    prepared = ExperiencePreparer(algorithm).prepare(
        rollout,
        reference_log_probs=reference,
        values=values,
    )

    assert prepared.reference_log_probs is not None
    assert not prepared.reference_log_probs.requires_grad
    assert prepared.advantages is not None
    assert not prepared.advantages.requires_grad
    torch.testing.assert_close(prepared.reference_log_probs, reference.detach())
    torch.testing.assert_close(prepared.sequences, rollout.sequences)
    torch.testing.assert_close(prepared.action_mask, rollout.action_mask)
    torch.testing.assert_close(prepared.old_log_probs, rollout.old_log_probs)
    if algorithm_name == "grpo":
        assert prepared.values is None
        assert prepared.returns is None
        normalized = 1 / (math.sqrt(2) + 1e-6)
        expected = torch.tensor([[-1, -1], [1, 1], [1, 1], [-1, -1]]) * normalized
    else:
        assert prepared.values is not None and not prepared.values.requires_grad
        assert prepared.returns is not None and not prepared.returns.requires_grad
        torch.testing.assert_close(prepared.values, values.detach())
        # Hand-computed two-action GAE for gamma=.9 and lambda=.8.
        raw = torch.tensor([[-0.064, -0.2], [1.212, 1.6], [1.768, 2.4], [0.164, 0.2]])
        expected = (raw - raw.mean()) / (raw.std(unbiased=False) + 1e-6)
        torch.testing.assert_close(
            prepared.returns, torch.tensor([[0.036, 0.0], [1.512, 2.0], [2.268, 3.0], [0.864, 1.0]])
        )
    torch.testing.assert_close(prepared.advantages, expected)
