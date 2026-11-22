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
"""Independent expert matrices retaining the source ModuleList and Muon optimizer boundaries."""
from __future__ import annotations

from typing import Any

from torch import nn

from hyper_parallel.components.modules.common.situ_mlp import SiTUMLPModule
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


class IndependentExpertsModule(nn.ModuleList):
    """Own source-indexed SiTU experts without flattening or concatenating their parameter matrices."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Retain the source's indexing protocol while replacing every expert's complete mathematics."""
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            if not isinstance(module, nn.ModuleList) or len(module) == 0:
                raise ValueError("Independent experts require a nonempty source ModuleList")
            experts = [SiTUMLPModule(module=expert) for expert in module]
        else:
            if not isinstance(config.num_experts, int) or config.num_experts < 1:
                raise ValueError("num_experts must be a positive integer")
            experts = [SiTUMLPModule(config=config, state_layout="w1_w3_w2") for _ in range(config.num_experts)]
        super().__init__(experts)
        if module is not None:
            self.training = module.training

    def make_transforms(self) -> list:
        """Source expert indices, projection keys and independent matrix shapes remain unchanged."""
        return []


def independent_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own container parameters without inventing an input/output call on a source ModuleList."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(
        params={name: dict(placements) for name, _ in module.named_parameters()},
        in_src={},
        in_dst={},
        out_src=None,
        out_dst=None,
        is_boundary=False,
    )


def validate_independent_experts(*, config: Any, module: nn.Module | None, context: Any, member: Any) -> None:
    """Keep source routing outside this container until independent-parameter EP transport is supported."""
    del config, module, context
    if member.axis_sizes["tp"] > 1 or member.axis_sizes["ep"] > 1:
        raise NotImplementedError("Independent experts currently support TP1/EP1 with FSDP data sharding")


@module_replacement
def IndependentExperts(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Build source-indexed experts while preserving each logical optimizer matrix."""
    return build_family(
        IndependentExperts, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh
    )


register_module_family(
    ModuleFamily(
        name="independent_experts",
        baseline=IndependentExperts,
        variants={"local": "hyper_parallel.components.modules.moe.independent_experts.IndependentExpertsModule"},
        default_variant="local",
        sharding=independent_sharding,
        validate=validate_independent_experts,
    )
)
