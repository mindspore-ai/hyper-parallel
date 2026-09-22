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
"""Architecture and lazy provider registration for DeepSeek-V3.2."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.deepseek_v32.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the family's replacement factories without importing NPU ops."""
    from hyper_parallel.models.deepseek_v32.adapter.conversion import (  # pylint: disable=C0415
        module_replacement,
    )

    return module_replacement


def _load_context_parallel():
    """Return the DeepSeek-V3.2 DSA CP wrapper provider lazily."""
    from hyper_parallel.models.deepseek_v32.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )

    return context_parallel


DEEPSEEK_V32_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV32ForCausalLM",
    model_type="deepseek_v32",
    replacements=_load_replacements,
    context_parallel=_load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(DEEPSEEK_V32_ADAPTER_SPEC)
