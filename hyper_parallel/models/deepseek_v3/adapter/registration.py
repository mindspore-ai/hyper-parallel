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
"""Register the DeepSeek-V2/V3 adapter capabilities.

DeepSeek-V2 shares the same MLA structure and declarative sharding policy.
Conversion providers remain lazy so registry discovery does not import model
implementations or backend dependencies.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.deepseek_v3.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the family's replacement-factory module (lazy provider)."""
    from hyper_parallel.models.deepseek_v3.adapter.conversion import (  # pylint: disable=C0415
        module_replacement,
    )

    return module_replacement


DEEPSEEK_V3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV3ForCausalLM",
    model_type="deepseek_v3",
    replacements=_load_replacements,
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(DEEPSEEK_V3_ADAPTER_SPEC)

# DeepSeek-V2 is structurally isomorphic for sharding purposes (same MLA
# naming); it gets its own spec identity but shares the rules provider.
DEEPSEEK_V2_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV2ForCausalLM",
    model_type="deepseek_v2",
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(DEEPSEEK_V2_ADAPTER_SPEC)
