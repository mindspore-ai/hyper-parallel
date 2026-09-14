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
"""Architecture identity and initialization adapter for Qwen3-Next."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_init_weights():
    """Return the family's shard-aware weight initializer."""
    from hyper_parallel.models.qwen3_next.adapter.init_weights import (  # pylint: disable=C0415
        initialize_weights,
    )
    return initialize_weights


QWEN3_NEXT_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3NextForCausalLM",
    model_type="qwen3_next",
    min_transformers_version="4.57.0",
    init_weights=_load_init_weights,
)

register_model_adapter(QWEN3_NEXT_ADAPTER_SPEC)
