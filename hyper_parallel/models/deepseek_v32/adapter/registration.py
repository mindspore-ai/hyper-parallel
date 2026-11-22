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

from hyper_parallel.components.modules.attention.deepseek_sparse_attention.indexer_share import install_indexer_share
from hyper_parallel.models.adapter_spec import ModelAdapterSpec, RecomputePolicy
from hyper_parallel.models.deepseek_v32.adapter.materialization import register_rotary_materialization
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


def _build_recompute_policy() -> RecomputePolicy:
    """Keep shared DSA inside non-reentrant decoder-layer recomputation."""
    return RecomputePolicy(
        region_patterns=("model.layers.*", "layers.*"), on_applied=install_indexer_share,
    )


DEEPSEEK_V32_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV32ForCausalLM",
    model_type="deepseek_v32",
    replacements=_load_replacements,
    materialization=register_rotary_materialization,
    recompute=_build_recompute_policy,
    sharding_rules=build_parameter_sharding_rules,
)
register_model_adapter(DEEPSEEK_V32_ADAPTER_SPEC)
