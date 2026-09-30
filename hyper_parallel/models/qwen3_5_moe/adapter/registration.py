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


QWEN3_5_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5MoeForConditionalGeneration",
    model_type="qwen3_5_moe",
    sharding_rules=_load_sharding_rules,
)

QWEN3_5_MOE_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5MoeForCausalLM",
    model_type="qwen3_5_moe_text",
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_5_MOE_ADAPTER_SPEC)
register_model_adapter(QWEN3_5_MOE_TEXT_ADAPTER_SPEC)
