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
"""Architecture identity and sharding rules for the Kimi K3 family."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_context_parallel():
    """Return the Kimi K3 CP wrapper module through a lazy provider."""
    from hyper_parallel.models.kimi_k3.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )
    return context_parallel


def _load_sharding_rules():
    """Return KDA-specific TP roles not covered by generic naming rules."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        (["q_conv1d", "k_conv1d", "v_conv1d"], ParamRole.COLWISE),
        (["A_log", "dt_bias"], ParamRole.COLWISE),
        (["f_a_proj", "g_a_proj"], ParamRole.REPLICATED),
        (["f_b_proj", "b_proj", "g_proj", "g_b_proj"], ParamRole.COLWISE),
    ]


KIMI_K3_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="KimiK3ForConditionalGeneration",
    model_type="kimi_k3",
    context_parallel=_load_context_parallel,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(KIMI_K3_ADAPTER_SPEC)
