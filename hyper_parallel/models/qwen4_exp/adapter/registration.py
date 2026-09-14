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
"""Architecture identities and initialization adapters for Qwen4 experimental."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_init_weights():
    """Return the family's shard-aware weight initializer."""
    from hyper_parallel.models.qwen4_exp.adapter.init_weights import (  # pylint: disable=C0415
        initialize_weights,
    )
    return initialize_weights


QWEN4_EXP_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen4ExpForConditionalGeneration",
    model_type="qwen4_exp",
    min_transformers_version="5.16.0",
    init_weights=_load_init_weights,
)

QWEN4_EXP_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen4ExpForCausalLM",
    model_type="qwen4_exp_text",
    min_transformers_version="5.16.0",
    init_weights=_load_init_weights,
)

register_model_adapter(QWEN4_EXP_ADAPTER_SPEC)
register_model_adapter(QWEN4_EXP_TEXT_ADAPTER_SPEC)
