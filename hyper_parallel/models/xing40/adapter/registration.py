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
"""Register the Xing4.0 model and its adapter.

Expert parallelism and the optimizer are selected by the recipe through
``plan_overrides`` and the ``optimizer`` target.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter
from hyper_parallel.models.xing40.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)


def _load_replacements():
    """Return the family's replacement-factory module (lazy provider)."""
    from hyper_parallel.models.xing40.adapter.conversion import (  # pylint: disable=import-outside-toplevel
        module_replacement,
    )

    return module_replacement


register_custom_model(
    "Xing40ForCausalLM",
    "hyper_parallel.models.xing40.modeling_xing40",
    "Xing40ForCausalLM",
)

XING40_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Xing40ForCausalLM",
    model_type="xing40",
    replacements=_load_replacements,
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(XING40_ADAPTER_SPEC)

__all__ = ["XING40_ADAPTER_SPEC"]
