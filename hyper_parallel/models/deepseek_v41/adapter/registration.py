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
"""Register the DeepSeek-V4.1 model and adapter providers."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.deepseek_v41.adapter.conversion.checkpoint_mapping import (
    register_deepseek_v41_checkpoint_mapping,
)
from hyper_parallel.models.deepseek_v41.adapter.ops.fused_lightning_indexer import (
    register_fused_indexer,
)
from hyper_parallel.models.deepseek_v41.adapter.policies.activation_checkpointing import (
    build_recompute_policy,
)
from hyper_parallel.models.deepseek_v41.adapter.policies.sharding import (
    build_parameter_sharding_rules,
    get_fsdp_excluded_subtrees,
    get_fsdp_execution_order,
    get_fsdp_wrap_modules,
)
from hyper_parallel.models.registry import (
    register_custom_model,
    register_model_adapter,
)


def _load_expert_parallel():
    """Return the model's MoE and Engram EP factories lazily."""
    from hyper_parallel.models.deepseek_v41.adapter.distributed import (  # pylint: disable=C0415
        moe_engram_expert_parallel,
    )

    return moe_engram_expert_parallel


def _load_context_parallel():
    """Return the model's shared-attention CP wrapper lazily."""
    from hyper_parallel.models.deepseek_v41.adapter.distributed import (  # pylint: disable=C0415
        shared_attention_context_parallel,
    )

    return shared_attention_context_parallel


def _load_validation_spec():
    """Return model-owned parity and self-consistency declarations lazily."""
    from hyper_parallel.models.deepseek_v41.adapter.validation.model_validation_spec import (  # pylint: disable=C0415
        get_validation_spec,
    )

    return get_validation_spec()


register_custom_model(
    "DeepseekV41ForCausalLM",
    "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41",
    "DeepseekV41ForCausalLM",
)
register_deepseek_v41_checkpoint_mapping()

DEEPSEEK_V41_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV41ForCausalLM",
    model_type="deepseek_v41",
    context_parallel=_load_context_parallel,
    expert_parallel=_load_expert_parallel,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
    recompute=build_recompute_policy,
    validation=_load_validation_spec,
)
register_model_adapter(DEEPSEEK_V41_ADAPTER_SPEC)

# The CSA selection chains live in the shared module; this installs the V4.1
# fused-operator provider they consult when it is available.
register_fused_indexer()


__all__ = ["DEEPSEEK_V41_ADAPTER_SPEC"]
