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
from hyper_parallel.models.kimi_k3.adapter import load_context_parallel
from hyper_parallel.models.kimi_k3.adapter.policies.sharding import (
    build_parameter_sharding_rules,
)
from hyper_parallel.models.registry import register_model_adapter


def _load_codegen():
    """Return Kimi source-export compatibility without loading it in training."""
    from hyper_parallel.models.kimi_k3.adapter import codegen  # pylint: disable=C0415

    return codegen


KIMI_K3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="KimiK3ForConditionalGeneration",
    model_type="kimi_k3",
    context_parallel=load_context_parallel,
    sharding_rules=build_parameter_sharding_rules,
    codegen=_load_codegen,
)

register_model_adapter(KIMI_K3_ADAPTER_SPEC)
