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
from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import CompressedSparseAttention2
from hyper_parallel.models.replacement import ModuleReplacementRequest
from hyper_parallel.models.registry import register_custom_model
from hyper_parallel.models.deepseek_v41.adapter.conversion.checkpoint_mapping import (
    register_deepseek_v41_checkpoint_mapping,
)
from hyper_parallel.models.deepseek_v41.adapter.engram.host_state import (
    build_deepseek_v41_external_state,
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


def _load_expert_parallel():
    """Return the model's MoE and Engram EP factories lazily."""
    from hyper_parallel.models.deepseek_v41.adapter.distributed import (  # pylint: disable=C0415
        moe_engram_expert_parallel,
    )

    return moe_engram_expert_parallel


def _load_validation_spec():
    """Return model-owned parity and self-consistency declarations lazily."""
    from hyper_parallel.models.deepseek_v41.adapter.validation.model_validation_spec import (  # pylint: disable=C0415
        get_validation_spec,
    )

    return get_validation_spec()


def _load_encoder_parallel_spec():
    """Load DeepSeek construction rules and its model-bound bridge factory lazily."""
    from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import (  # pylint: disable=C0415
        get_deepseek_v41_encoder_parallel_spec,
    )

    return get_deepseek_v41_encoder_parallel_spec()


def _load_loss():
    """Return the V4.1 model-integrated output-loss adapter."""
    from hyper_parallel.models.deepseek_v41.adapter import chunk_loss  # pylint: disable=C0415

    return chunk_loss


def _replacements(model, context):
    """Declare the complete CSA2 family; user options override this default."""
    # Resolve the canonical class after model construction to avoid a spec/model import cycle.
    from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (  # pylint: disable=import-outside-toplevel
        DeepseekV41Attention,
    )

    del model, context
    return (ModuleReplacementRequest(
        match=("model.layers.*.self_attn",), target=CompressedSparseAttention2,
        module_type=DeepseekV41Attention,
    ),)


MODEL_SPEC = ModelAdapterSpec(
    architecture="DeepseekV41ForCausalLM",
    model_type="deepseek_v41",
    replacements=_replacements,
    expert_parallel=_load_expert_parallel,
    external_state=build_deepseek_v41_external_state,
    sharding_rules=build_parameter_sharding_rules,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
    recompute=build_recompute_policy,
    validation=_load_validation_spec,
    encoder_parallel=_load_encoder_parallel_spec,
    loss=_load_loss,
)

register_custom_model(
    "DeepseekV41ForCausalLM",
    "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41",
    "DeepseekV41ForCausalLM",
)
register_deepseek_v41_checkpoint_mapping()

__all__ = ["MODEL_SPEC"]
