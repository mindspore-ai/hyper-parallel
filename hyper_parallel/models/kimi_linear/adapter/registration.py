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
"""Architecture identity for the Kimi Linear text-model family."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.kimi_k3.adapter.registration import (
    _load_context_parallel,
    _load_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


KIMI_LINEAR_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="KimiLinearForCausalLM",
    model_type="kimi_linear",
    context_parallel=_load_context_parallel,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(KIMI_LINEAR_ADAPTER_SPEC)
