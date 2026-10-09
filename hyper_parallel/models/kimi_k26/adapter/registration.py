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
"""Architecture identity and sharding rules for the Kimi K2.5/K2.6 family.

HyperParallel registers the family as ``kimi_k26`` (directory
``models/kimi_k26/``); the native Transformers configuration shipped with the K2.6
checkpoint reports ``kimi_k25`` / ``KimiK25ForConditionalGeneration``, and
``models/registry.py`` maps that spelling onto this registration.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_sharding_rules():
    """Return MLA and vision-tower TP roles for the K2.5/K2.6 multimodal arch.

    The family's top-level HF ``model_type`` is ``kimi_k25`` (architecture
    ``KimiK25ForConditionalGeneration``), registered here as ``kimi_k26``, so the
    DeepSeek-V3 MLA naming rules
    do not apply here even though the nested text tower is MLA-based: the
    low-rank down-projections stay replicated (their LoRA rank dim is not
    sharded) and the up-projections are colwise along the head dim, matching
    the standard attention template's o_proj rowwise contract.

    The vision tower and projector are replicated on every TP rank: their
    native attention reshapes heads on the full hidden dim, so TP sharding of
    the vision path is not composable.
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        (["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
        (["vision_tower."], ParamRole.REPLICATED),
        (["mm_projector."], ParamRole.REPLICATED),
    ]


def _load_loss():
    """Return the family's model-integrated output-loss adapter."""
    from hyper_parallel.models.kimi_k26.adapter import (  # pylint: disable=C0415
        chunk_loss,
    )
    return chunk_loss


KIMI_K26_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="KimiK25ForConditionalGeneration",
    model_type="kimi_k26",
    sharding_rules=_load_sharding_rules,
    loss=_load_loss,
)

register_model_adapter(KIMI_K26_ADAPTER_SPEC)
