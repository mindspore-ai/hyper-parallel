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
"""Lazy registration of the independently constructed JT family."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.jt_deepseek_v3.adapter.policies.sharding import build_parameter_sharding_rules
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


JT_DEEPSEEK_V3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="JTDeepseekV3ForCausalLM",
    model_type="jt_deepseek_v3",
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(JT_DEEPSEEK_V3_ADAPTER_SPEC)
register_custom_model(
    "JTDeepseekV3ForCausalLM",
    "hyper_parallel.models.jt_deepseek_v3.modeling_jt_deepseek_v3",
    "JTDeepseekV3ForCausalLM",
)
