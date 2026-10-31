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
"""Architecture identity and GDN sharding rules for Qwen3.5."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3_5.adapter.materialization import register_rotary_materialization
from hyper_parallel.models.qwen3_5.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.qwen3_5_moe.adapter.policies.sharding import get_fsdp_execution_order
from hyper_parallel.models.registry import register_model_adapter


def _load_context_parallel():
    """Return the Qwen3.5 CP wrapper module through a lazy provider."""
    from hyper_parallel.models.qwen3_5.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )
    return context_parallel


def _load_loss():
    """Load the optional text-only output-loss adapter lazily."""
    from hyper_parallel.models.qwen3_5.adapter import chunk_loss  # pylint: disable=C0415
    return chunk_loss


def _load_vlm_loss():
    """Load the shared Qwen VLM loss provider without eager optional imports."""
    from hyper_parallel.models.qwen3_5_moe.adapter import vlm_chunk_loss  # pylint: disable=C0415
    return vlm_chunk_loss


def _load_encoder_parallel():
    """Reuse the shared Qwen visual ownership contract through a lazy provider."""
    from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import (  # pylint: disable=C0415
        get_encoder_parallel_spec,
    )
    return get_encoder_parallel_spec()


QWEN3_5_ADAPTER_SPEC = ModelAdapterSpec(
    materialization=register_rotary_materialization,
    architecture="Qwen3_5ForConditionalGeneration",
    model_type="qwen3_5",
    loss=_load_vlm_loss,
    encoder_parallel=_load_encoder_parallel,
    fsdp_execution_order=get_fsdp_execution_order,
    context_parallel=_load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
)

QWEN3_5_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    materialization=register_rotary_materialization,
    architecture="Qwen3_5ForCausalLM",
    model_type="qwen3_5_text",
    loss=_load_loss,
    context_parallel=_load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
)

register_model_adapter(QWEN3_5_ADAPTER_SPEC)
register_model_adapter(QWEN3_5_TEXT_ADAPTER_SPEC)
