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
"""Policy Actor-to-rollout weight synchronization."""
from rl.roles.model import (
    HYPER_MODEL_IMPLEMENTATION,
    HYPER_QWEN3_ARCHITECTURE,
    HYPER_QWEN3_5_ARCHITECTURE,
    NATIVE_MODEL_IMPLEMENTATION,
    NATIVE_QWEN3_ARCHITECTURE,
    NATIVE_QWEN3_5_ARCHITECTURE,
    SUPPORTED_MODEL_IMPLEMENTATIONS,
    architecture_for_implementation,
    normalize_model_implementation,
)
from rl.roles.weight_sync.sync import (
    ActorRolloutWeightSync,
    POLICY_FINGERPRINT_ALGORITHM,
    PolicySnapshot,
    VLLMWeightSyncClientMixin,
    aggregate_policy_fingerprint,
    canonical_policy_weight_name,
    is_policy_fingerprint_weight,
    policy_fingerprint_header,
    policy_tensor_fingerprint,
    policy_weight_fingerprint,
    synchronized_call,
    synchronize_error,
    verify_policy_fingerprints,
)
from rl.roles.weight_sync.transfer import (
    CPUStateDictRefitter,
    CPUWeightTransfer,
    HCCLWeightRefitter,
    HCCLWeightTransfer,
    NPUIPCWeightRefitter,
    NPUIPCWeightTransfer,
    VLLMWeightRefitter,
    WeightTransfer,
    build_weight_transfer,
    map_actor_state_dict,
    map_policy_state_dict,
)
__all__ = [
    "ActorRolloutWeightSync",
    "CPUStateDictRefitter",
    "CPUWeightTransfer",
    "HCCLWeightRefitter",
    "HCCLWeightTransfer",
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_ARCHITECTURE",
    "HYPER_QWEN3_5_ARCHITECTURE",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_ARCHITECTURE",
    "NATIVE_QWEN3_5_ARCHITECTURE",
    "NPUIPCWeightRefitter",
    "NPUIPCWeightTransfer",
    "PolicySnapshot",
    "SUPPORTED_MODEL_IMPLEMENTATIONS",
    "VLLMWeightRefitter",
    "VLLMWeightSyncClientMixin",
    "WeightTransfer",
    "aggregate_policy_fingerprint",
    "architecture_for_implementation",
    "build_weight_transfer",
    "canonical_policy_weight_name",
    "is_policy_fingerprint_weight",
    "map_actor_state_dict",
    "map_policy_state_dict",
    "normalize_model_implementation",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
    "policy_weight_fingerprint",
    "synchronized_call",
    "synchronize_error",
    "verify_policy_fingerprints",
]
