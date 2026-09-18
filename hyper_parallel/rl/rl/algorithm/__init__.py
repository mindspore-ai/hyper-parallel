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
"""Stable algorithm construction and extension API."""

from rl.algorithm.advantage import (
    TargetOutput,
    get_advantage_estimator,
    register_advantage_estimator,
)
from rl.algorithm.loss import (
    AlgorithmRequirements,
    CriticLossOutput,
    DataRequirements,
    GRPOAlgorithm,
    GRPOConfig,
    LossOutput,
    PPOAlgorithm,
    PPOConfig,
    RLAlgorithm,
    RoleRequirements,
    build_algorithm,
    get_policy_loss,
    register_algorithm,
    register_policy_loss,
)
from rl.algorithm.reward import (
    RewardFunction,
    compute_rule_reward,
    extract_answer,
    get_reward,
    register_reward,
)


__all__ = [
    "AlgorithmRequirements",
    "CriticLossOutput",
    "DataRequirements",
    "GRPOAlgorithm",
    "GRPOConfig",
    "LossOutput",
    "PPOAlgorithm",
    "PPOConfig",
    "RLAlgorithm",
    "RewardFunction",
    "RoleRequirements",
    "TargetOutput",
    "build_algorithm",
    "compute_rule_reward",
    "extract_answer",
    "get_advantage_estimator",
    "get_policy_loss",
    "get_reward",
    "register_advantage_estimator",
    "register_algorithm",
    "register_policy_loss",
    "register_reward",
]
