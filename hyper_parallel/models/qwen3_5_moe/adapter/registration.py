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
"""Architecture and sharding contracts for Qwen3.5-MoE and Qwen3.6."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3_5.adapter.materialization import register_rotary_materialization
from hyper_parallel.models.qwen3_5_moe.adapter.policies.sharding import get_fsdp_execution_order
from hyper_parallel.models.registry import register_model_adapter


def _load_sharding_rules():
    """Return GDN and shared-expert parameter role overrides."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        ("in_proj_qkv", ParamRole.FUSED_QKV),
        (["in_proj_z", "in_proj_b", "in_proj_a", "conv1d"], ParamRole.COLWISE),
        (["A_log", "dt_bias"], ParamRole.COLWISE),
        ("out_proj", ParamRole.ROWWISE),
        ("shared_expert_gate", ParamRole.REPLICATED),
    ]


def _load_loss():
    """Reuse the dense/MoE text objective without loading optional kernels eagerly."""
    from hyper_parallel.models.qwen3_5.adapter import chunk_loss  # pylint: disable=C0415
    return chunk_loss


def _load_vlm_loss():
    """Keep the visual training objective separate from the text loss adapter."""
    from hyper_parallel.models.qwen3_5_moe.adapter import vlm_chunk_loss  # pylint: disable=C0415
    return vlm_chunk_loss


def _load_encoder_parallel():
    """Load the opt-in visual runtime without importing optional HF vision code."""
    from hyper_parallel.models.qwen3_5_moe.adapter.encoder_parallel import (  # pylint: disable=C0415
        get_encoder_parallel_spec,
    )
    return get_encoder_parallel_spec()


QWEN3_5_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    materialization=register_rotary_materialization,
    architecture="Qwen3_5MoeForConditionalGeneration",
    model_type="qwen3_5_moe",
    loss=_load_vlm_loss,
    encoder_parallel=_load_encoder_parallel,
    sharding_rules=_load_sharding_rules,
    fsdp_execution_order=get_fsdp_execution_order,
)

QWEN3_5_MOE_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    materialization=register_rotary_materialization,
    architecture="Qwen3_5MoeForCausalLM",
    model_type="qwen3_5_moe_text",
    loss=_load_loss,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_5_MOE_ADAPTER_SPEC)
register_model_adapter(QWEN3_5_MOE_TEXT_ADAPTER_SPEC)
