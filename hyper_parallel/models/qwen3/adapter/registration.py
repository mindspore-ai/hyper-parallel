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
"""Architecture identity and adapter providers for Qwen3 dense models."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the Qwen3 structure-replacement module through a lazy provider."""
    from hyper_parallel.models.qwen3.adapter.conversion import (  # pylint: disable=C0415
        module_replacement,
    )

    return module_replacement


def _load_attention():
    """Return the Qwen3 attention-contract module through a lazy provider."""
    from hyper_parallel.models.qwen3.adapter.runtime import (  # pylint: disable=C0415
        attention,
    )

    return attention


QWEN3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    replacements=_load_replacements,
    attention=_load_attention,
    sharding_rules=build_parameter_sharding_rules,
)

register_model_adapter(QWEN3_ADAPTER_SPEC)
