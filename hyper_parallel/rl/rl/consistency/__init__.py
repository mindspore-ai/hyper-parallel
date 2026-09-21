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
"""Versioned training-rollout numerical consistency profiles."""

from rl.consistency.gates import (
    measure_post_update_old_policy_mismatch,
    validate_consistency_forward_inputs,
    validate_pre_update_consistency,
)
from rl.consistency.qwen3_dense import (
    CONSISTENCY_PROFILE_OFF,
    QWEN3_ASCEND_CONSISTENCY_V1,
    configure_consistency_profile,
    consistency_profile,
    consistency_runtime_state,
    install_rollout_consistency_profile,
    install_trainer_consistency_profile,
    trainer_sequence_log_probs,
    validate_consistency_model_identity,
    validate_rollout_consistency_profile,
)

__all__ = [
    "CONSISTENCY_PROFILE_OFF",
    "QWEN3_ASCEND_CONSISTENCY_V1",
    "configure_consistency_profile",
    "consistency_profile",
    "consistency_runtime_state",
    "install_rollout_consistency_profile",
    "install_trainer_consistency_profile",
    "measure_post_update_old_policy_mismatch",
    "trainer_sequence_log_probs",
    "validate_consistency_forward_inputs",
    "validate_pre_update_consistency",
    "validate_consistency_model_identity",
    "validate_rollout_consistency_profile",
]
