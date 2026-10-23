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
"""Qwen3-VL architecture identity and visual FSDP policies."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.qwen3_vl.adapter.fsdp import (
    get_fsdp_execution_order,
    get_fsdp_excluded_subtrees,
    get_fsdp_prefetch_groups,
    get_fsdp_wrap_modules,
)
from hyper_parallel.models.registry import register_model_adapter


QWEN3_VL_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3VLForConditionalGeneration",
    model_type="qwen3_vl",
    fsdp_wrap_modules=get_fsdp_wrap_modules,
    fsdp_excluded_subtrees=get_fsdp_excluded_subtrees,
    fsdp_execution_order=get_fsdp_execution_order,
    fsdp_prefetch_groups=get_fsdp_prefetch_groups,
)

register_model_adapter(QWEN3_VL_ADAPTER_SPEC)
