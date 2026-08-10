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
"""Hyper-Parallel-backed Policy role implementations."""

from rl.roles.policy.actor import ActorManager, ActorModel, UpdateMetrics
from rl.roles.policy.critic import CriticManager, CriticUpdateMetrics
from rl.roles.policy.value import CriticModel, attach_value_head

__all__ = [
    "ActorManager",
    "ActorModel",
    "CriticManager",
    "CriticModel",
    "CriticUpdateMetrics",
    "UpdateMetrics",
    "attach_value_head",
]
