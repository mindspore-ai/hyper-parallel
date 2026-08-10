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
"""Stable role API over policy, rollout, and weight-sync implementations."""

from rl.roles.model import ModelRegistration, register_configured_model
from rl.roles.policy import (
    ActorManager,
    ActorModel,
    CriticManager,
    CriticModel,
    CriticUpdateMetrics,
    UpdateMetrics,
    attach_value_head,
)

__all__ = [
    "ActorManager",
    "ActorModel",
    "CriticManager",
    "CriticModel",
    "CriticUpdateMetrics",
    "ModelRegistration",
    "UpdateMetrics",
    "attach_value_head",
    "register_configured_model",
]
