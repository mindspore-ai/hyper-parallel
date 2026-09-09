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
"""CPU unit tests for successful Algorithm registry construction."""

from rl.algorithm import build_algorithm
from rl.algorithm.advantage import GAEAdvantageEstimator, GroupRelativeAdvantageEstimator, get_advantage_estimator
from rl.algorithm.loss import ClippedPolicyObjective, get_policy_loss


def test_registered_algorithms_expose_expected_requirements() -> None:
    """GRPO and PPO recipes declare the roles and data the Trainer must provide."""
    grpo = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    ppo = build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"})

    assert grpo.name == "grpo"
    assert grpo.requirements.roles.reference
    assert not grpo.requirements.roles.critic
    assert grpo.requirements.data.rollout_log_probs
    assert grpo.requirements.data.reference_log_probs
    assert grpo.requirements.data.grouped_responses
    assert ppo.name == "ppo"
    assert ppo.requirements.roles.reference
    assert ppo.requirements.roles.critic
    assert ppo.requirements.data.values
    assert ppo.requirements.data.returns


def test_registered_algorithm_components_build_successfully() -> None:
    """Built-in advantage estimators and clipped objective retain their configuration."""
    grpo = get_advantage_estimator("grpo", epsilon=1.0e-5)
    gae = get_advantage_estimator("gae", gamma=0.9, gae_lambda=0.8, normalize=False)
    objective = get_policy_loss(
        "clipped",
        clip_ratio_low=0.1,
        clip_ratio_high=0.3,
        dual_clip=2.0,
    )

    assert isinstance(grpo, GroupRelativeAdvantageEstimator)
    assert grpo.epsilon == 1.0e-5
    assert isinstance(gae, GAEAdvantageEstimator)
    assert gae.gamma == 0.9
    assert gae.gae_lambda == 0.8
    assert not gae.normalize
    assert isinstance(objective, ClippedPolicyObjective)
    assert objective.clip_ratio_low == 0.1
    assert objective.clip_ratio_high == 0.3
    assert objective.dual_clip == 2.0
