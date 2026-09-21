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
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the Qwen3 structure-replacement module through a lazy provider."""
    from hyper_parallel.models.qwen3.adapter import replacements  # pylint: disable=C0415
    return replacements


def _load_attention():
    """Return the Qwen3 attention-contract module through a lazy provider."""
    from hyper_parallel.models.qwen3.adapter import attention  # pylint: disable=C0415
    return attention


def _load_sharding_rules():
    """Return TP roles for parameters introduced by the fused SwiGLU replacement."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        ("linear_fc1.weight", ParamRole.FUSED_GATE_UP),
        ("linear_fc2.weight", ParamRole.ROWWISE),
    ]


QWEN3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    replacements=_load_replacements,
    attention=_load_attention,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_ADAPTER_SPEC)
