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
"""Qwen3-VL-MoE parameter sharding roles."""

from typing import Any

from torch import nn

from hyper_parallel.components.modules.module_family import FamilyShardingContext
from hyper_parallel.distributed._builder.family_layout import replicated_subtree_layout
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole


def build_vision_layout(
    *, model: nn.Module, context: FamilyShardingContext,
) -> dict[str, ModuleShardingSpec]:
    """Declare the vision tower's parameters replicated without a forward wrapper."""
    vision = getattr(getattr(model, "model", None), "visual", None)
    return {} if vision is None else {"model.visual": replicated_subtree_layout(vision, context)}


def build_parameter_sharding_rules() -> list[tuple[Any, Any]]:
    """Replicate the separately owned vision tower across the TP domain."""
    return [("model.visual.", ParamRole.REPLICATED)]


__all__ = ["build_parameter_sharding_rules", "build_vision_layout"]
