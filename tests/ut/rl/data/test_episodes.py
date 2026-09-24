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
"""Episode-level GRPO and zero-loss DP alignment contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.dataset import batch_builder
from rl.dataset.batch_builder import (
    ExperiencePreparer, build_experience_batch, normalize_advantages, pad_agent_call_batch_for_dp,
)
from rl.dataset.contracts import Trajectory, Turn
from rl.dataset.episodes import episode_rows
from rl.roles.rollout.base import GenerationSettings



def _call(episode: str, index: int, count: int, reward: float) -> Trajectory:
    return Trajectory(
        trajectory_id=f"{episode}:{index}", prompt_id="prompt", group_id="prompt", policy_version=1,
        turns=(Turn("user", "real context", 0, 2, False), Turn("assistant", "sampled action", 2, 4, True)),
        token_ids=torch.tensor([11, 12 + index, 20, 21]), attention_mask=torch.ones(4, dtype=torch.bool),
        action_mask=torch.tensor([False, False, True, True]), rollout_log_probs=torch.tensor([0.0, -0.2, -0.3]),
        reward=reward, reward_components={"success": reward}, done=True, truncated=False,
        terminal_reason="completed", worker_policy_version=1,
        metadata={"episode_id": episode, "call_index": index, "call_count": count},
    )


def _batch(rows: tuple[Trajectory, ...] = ()) -> object:
    rows = rows or (_call("a", 0, 1, 1.0), _call("b", 0, 2, 0.0), _call("b", 1, 2, 0.0))
    settings = GenerationSettings(max_new_tokens=2, temperature=1.0, top_p=1.0, top_k=0, do_sample=True,
                                  pad_token_id=0, eos_token_id=2, collect_log_probs=True)
    return build_experience_batch(rows, 1.0, settings, {})


def _prepare(rollout: object) -> object:
    algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    return ExperiencePreparer(algorithm).prepare(rollout, reference_log_probs=torch.zeros_like(rollout.old_log_probs))


def test_episode_advantage_does_not_duplicate_multicall_rewards() -> None:
    """One-call and two-call episodes contribute one reward each to the GRPO group."""
    batch = _batch()
    prepared = _prepare(batch)
    scale = 0.5 / (torch.tensor([1.0, 0.0]).std() + 1.0e-6)
    expected = torch.tensor([[0.0, scale, scale], [0.0, -scale, -scale], [0.0, -scale, -scale]])
    torch.testing.assert_close(prepared.advantages, expected)
    assert episode_rows(batch.trajectories) == [[0], [1, 2]]
    shuffled = (batch.trajectories[2], batch.trajectories[0], batch.trajectories[1])
    assert episode_rows(shuffled) == [[2, 0], [1]]


@pytest.mark.parametrize("field,value", [
    ("call_index", 0), ("call_index", True), ("call_count", 3), ("call_count", 2.0), ("episode_id", ""),
])
def test_episode_rejects_missing_or_duplicate_call_metadata(field: str, value: object) -> None:
    """An episode cannot silently omit, duplicate or coerce a model-call index."""
    rows = [_call("b", 0, 2, 0.0), _call("b", 1, 2, 0.0)]
    rows[1] = replace(rows[1], metadata={**rows[1].metadata, field: value})
    with pytest.raises(ValueError):
        episode_rows(rows)


@pytest.mark.parametrize("changes", [
    {"prompt_id": "other"}, {"group_id": "other"}, {"reward": 1.0}, {"reward_components": {"success": 1.0}},
    {"policy_version": 2, "worker_policy_version": 2}, {"worker_policy_version": None},
])
def test_episode_rejects_inconsistent_identity_or_reward(changes: dict) -> None:
    """All training rows in an episode share one immutable outcome and policy."""
    rows = (_call("b", 0, 2, 0.0), replace(_call("b", 1, 2, 0.0), **changes))
    with pytest.raises(ValueError, match="Inconsistent"):
        episode_rows(rows)


def test_dp_padding_preserves_advantages_and_token_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra DP scheduling rows cannot alter reward grouping or valid-token normalization."""
    batch = _batch()

    def all_reduce(summary: torch.Tensor, *, op: object, group: object) -> None:
        """Simulate a peer with five real call rows."""
        assert summary.tolist() == [1, 3]
        assert op == torch.distributed.ReduceOp.MAX and group == "dp"
        summary[1] = 5

    monkeypatch.setattr(batch_builder.dist, "all_reduce", all_reduce)
    padded = pad_agent_call_batch_for_dp(batch, SimpleNamespace(rank_size=2, group="dp"))
    original, prepared = _prepare(batch), _prepare(padded)
    torch.testing.assert_close(prepared.advantages[:3], original.advantages)
    assert not prepared.loss_action_mask[3:].any()
    assert not prepared.advantages[3:].any()
    assert not prepared.rewards[3:].any()
    assert episode_rows(prepared.trajectories) == [[0], [1, 2]]
    assert prepared.metadata["generated_action_tokens"] == batch.metadata["generated_action_tokens"]
    whitened = normalize_advantages(prepared.advantages, prepared.loss_action_mask)
    baseline = normalize_advantages(original.advantages, original.loss_action_mask)
    torch.testing.assert_close(whitened[:3], baseline)
    assert not whitened[3:].any()


def test_legacy_rank_participates_when_peer_has_segmented_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank cannot skip the collective merely because its own rows are not segmented."""
    batch = _batch(tuple(replace(_call(name, 0, 1, reward), metadata={}) for name, reward in (("a", 1.0), ("b", 0.0))))

    def all_reduce(summary: torch.Tensor, **_kwargs: object) -> None:
        """Advertise a segmented peer without changing the local real episodes."""
        assert summary.tolist() == [0, 2]
        summary.copy_(summary.new_tensor([1, 3]))

    monkeypatch.setattr(batch_builder.dist, "all_reduce", all_reduce)
    padded = pad_agent_call_batch_for_dp(batch, SimpleNamespace(rank_size=2, group="dp"))
    torch.testing.assert_close(_prepare(padded).advantages[:2], _prepare(batch).advantages)
    assert not _prepare(padded).advantages[2].any()


def test_segmented_ppo_fails_before_missing_critic_inputs() -> None:
    """PPO cannot reuse per-call GRPO semantics or silently bootstrap rewritten contexts."""
    algorithm = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})
    with pytest.raises(ValueError, match="episode-level GRPO"):
        ExperiencePreparer(algorithm).prepare(_batch())


@pytest.mark.parametrize("changes,error", [
    ({"action_mask": torch.zeros(4, dtype=torch.bool)}, "valid next-token actions"),
    ({"reward": float("nan")}, "reward must be finite"),
    ({"rollout_log_probs": torch.tensor([0.0, float("nan"), -0.3])}, "log probabilities must be finite"),
    ({"metadata": {"dp_padding": True}}, "DP padding must have zero reward"),
])
def test_invalid_training_rows_fail_instead_of_becoming_zero_reward(changes: dict, error: str) -> None:
    """Invalid actions and fabricated padding never enter a training batch."""
    with pytest.raises(ValueError, match=error):
        _batch((replace(_call("a", 0, 1, 1.0), **changes),))


def test_episode_targets_reject_reward_tensor_drift() -> None:
    """Tensor labels cannot disagree with the validated episode outcome."""
    batch = _batch()
    with pytest.raises(ValueError, match="reward tensor must match"):
        _prepare(replace(batch, rewards=torch.ones_like(batch.rewards)))
