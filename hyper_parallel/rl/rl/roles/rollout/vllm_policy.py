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
"""Compatibility exports for policy weight naming and fingerprints."""
from rl.roles.weight_sync.transfer import (
    HYPER_MODEL_IMPLEMENTATION,
    HYPER_QWEN3_5_ARCHITECTURE,
    NATIVE_MODEL_IMPLEMENTATION,
    NATIVE_QWEN3_5_ARCHITECTURE,
    POLICY_FINGERPRINT_ALGORITHM,
    SUPPORTED_MODEL_IMPLEMENTATIONS,
    aggregate_policy_fingerprint,
    architecture_for_implementation,
    canonical_policy_weight_name,
    is_policy_fingerprint_weight,
    map_policy_state_dict,
    normalize_model_implementation,
    policy_fingerprint_header,
    policy_tensor_fingerprint,
)
__all__ = [
    "HYPER_MODEL_IMPLEMENTATION",
    "HYPER_QWEN3_5_ARCHITECTURE",
    "NATIVE_MODEL_IMPLEMENTATION",
    "NATIVE_QWEN3_5_ARCHITECTURE",
    "POLICY_FINGERPRINT_ALGORITHM",
    "SUPPORTED_MODEL_IMPLEMENTATIONS",
    "aggregate_policy_fingerprint",
    "architecture_for_implementation",
    "canonical_policy_weight_name",
    "is_policy_fingerprint_weight",
    "map_policy_state_dict",
    "normalize_model_implementation",
    "policy_fingerprint_header",
    "policy_tensor_fingerprint",
]
