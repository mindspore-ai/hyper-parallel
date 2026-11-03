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
"""Register the Kimi K3 adapter capabilities."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.kimi_k3.adapter import load_context_parallel, load_loss, load_text_loss
from hyper_parallel.models.kimi_k3.adapter.policies.activation_checkpointing import (
    build_recompute_policy,
    build_text_recompute_policy,
)
from hyper_parallel.models.kimi_k3.adapter.policies.sharding import (
    build_parameter_sharding_rules,
    get_fsdp_wrap_modules,
    get_fsdp_excluded_subtrees,
    get_fsdp_execution_order,
)
from hyper_parallel.models.registry import register_custom_model, register_model_adapter


def _load_text_validation_spec():
    """Keep optional HF diagnostic dependencies out of ordinary adapter discovery."""
    from hyper_parallel.models.kimi_k3.adapter.validation.model_validation_spec import (  # pylint: disable=C0415
        get_validation_spec,
    )
    return get_validation_spec()


def _load_multimodal_validation_spec():
    """Load the image contract only when model-integration diagnostics request it."""
    from hyper_parallel.models.kimi_k3.adapter.validation.model_validation_spec import (  # pylint: disable=C0415
        get_multimodal_validation_spec,
    )
    return get_multimodal_validation_spec()


def _load_encoder_parallel_spec():
    """Load the optional visual execution provider only when encoder parallelism is requested."""
    from hyper_parallel.models.kimi_k3.adapter.distributed.encoder_parallel import (  # pylint: disable=C0415
        get_kimi_k3_encoder_parallel_spec,
    )
    return get_kimi_k3_encoder_parallel_spec()


KIMI_K3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="KimiK3ForConditionalGeneration",
    encoder_parallel=_load_encoder_parallel_spec,
    model_type="kimi_k3",
    context_parallel=load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
    loss=load_loss,
    validation=_load_multimodal_validation_spec,
    recompute=build_recompute_policy,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
)

register_model_adapter(KIMI_K3_ADAPTER_SPEC)

register_custom_model(
    "KimiK3ForConditionalGeneration",
    "hyper_parallel.models.kimi_k3.modeling_kimi_k3",
    "KimiK3ForConditionalGeneration",
)

KIMI_LINEAR_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="KimiLinearForCausalLM",
    model_type="kimi_linear",
    context_parallel=load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
    loss=load_text_loss,
    validation=_load_text_validation_spec,
    recompute=build_text_recompute_policy,
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
)

register_model_adapter(KIMI_LINEAR_ADAPTER_SPEC)
register_custom_model(
    "KimiLinearForCausalLM",
    "hyper_parallel.models.kimi_k3.modeling_kimi_linear",
    "KimiLinearForCausalLM",
)
