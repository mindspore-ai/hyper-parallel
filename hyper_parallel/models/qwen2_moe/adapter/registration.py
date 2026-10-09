# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Register the Qwen2-MoE adapter capabilities.

The family contributes only declarative sharding rules and otherwise uses the
HF-native implementation with generic framework templates.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen2_moe.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


QWEN2_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen2MoeForCausalLM",
    model_type="qwen2_moe",
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(QWEN2_MOE_ADAPTER_SPEC)
