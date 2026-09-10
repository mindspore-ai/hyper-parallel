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
from hyper_parallel.models.registry import register_model_adapter


def _load_context_parallel():
    """Return the Qwen3.5 CP wrapper module through a lazy provider."""
    from hyper_parallel.models.qwen3_5.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )
    return context_parallel


def _load_sharding_rules():
    """Return GDN parameter roles not covered by generic naming rules."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        ("in_proj_qkv", ParamRole.FUSED_QKV),
        (["in_proj_z", "in_proj_b", "in_proj_a", "conv1d"], ParamRole.COLWISE),
        (["A_log", "dt_bias"], ParamRole.COLWISE),
        ("out_proj", ParamRole.ROWWISE),
    ]


QWEN3_5_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5ForConditionalGeneration",
    model_type="qwen3_5",
    context_parallel=_load_context_parallel,
    sharding_rules=_load_sharding_rules,
)

QWEN3_5_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5ForCausalLM",
    model_type="qwen3_5_text",
    context_parallel=_load_context_parallel,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_5_ADAPTER_SPEC)
register_model_adapter(QWEN3_5_TEXT_ADAPTER_SPEC)
