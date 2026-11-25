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
"""Normalized spatial patch grouping and two-layer language-width projection."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.vision.learned_axial_vision_transformer import replicated_sharding
from hyper_parallel.models.replacement import module_replacement


class SpatialPatchProjectionModule(nn.Module):
    """Own pre-normalization, grouped patch reshape, exact GELU and language projection."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Adopt source primitives or directly allocate a complete projection contract."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self._owns_initialization = module is None
        if module is not None:
            children = set(dict(module.named_children()))
            if children == {"proj", "post_norm"}:
                if not isinstance(module.proj, nn.Sequential) or len(module.proj) != 3:
                    raise ValueError("Post-normalized spatial projector requires Linear/GELU/Linear")
                first, activation, last = module.proj
                if (not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear)
                        or first.bias is not None or last.bias is not None
                        or not isinstance(activation, nn.GELU) or activation.approximate != "none"
                        or not isinstance(module.post_norm, nn.RMSNorm)):
                    raise ValueError("Unsupported post-normalized spatial patch projection")
                self.hidden_size = module.hidden_size
                self.layout = "post_normalized"
                self.proj, self.post_norm = module.proj, module.post_norm
                self.training = module.training
                return
            if set(dict(module.named_children())) != {"pre_norm", "in_proj", "act", "out_proj"}:
                raise ValueError("Unsupported normalized spatial patch projection source")
            if not isinstance(module.act, nn.GELU) or module.act.approximate != "none":
                raise ValueError("Spatial patch projection requires exact GELU")
            self.hidden_size = module.hidden_size
            self.pre_norm, self.in_proj, self.act, self.out_proj = (
                module.pre_norm,
                module.in_proj,
                module.act,
                module.out_proj,
            )
            self.training = module.training
        else:
            vision = getattr(config, "vision_config", config)
            if getattr(vision, "mm_projector_type", None) == "patchmergerv2":
                self.hidden_size = vision.mm_hidden_size * math.prod(vision.merge_kernel_size)
                self.layout = "post_normalized"
                self.proj = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size, bias=False),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, vision.hidden_size, bias=False),
                )
                self.post_norm = nn.RMSNorm(vision.hidden_size, eps=vision.projector_ln_eps)
                self.reset_parameters()
                return
            self.hidden_size = config.vision_config.hidden_size * (
                config.vision_config.merge_kernel_size[0] * config.vision_config.merge_kernel_size[1]
            )
            self.pre_norm = nn.LayerNorm(config.projection_hidden_size, eps=config.projection_layer_norm_eps)
            self.in_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.act = nn.GELU()
            self.out_proj = nn.Linear(self.hidden_size, config.text_config.hidden_size)
        self.layout = "pre_normalized"

    def reset_parameters(self) -> None:
        """Restore directly constructed state after materialization; retain source-owned initialization."""
        if not self._owns_initialization:
            return
        if self.layout == "post_normalized":
            for child in self.proj:
                if isinstance(child, nn.Linear):
                    nn.init.trunc_normal_(child.weight, std=math.sqrt(2 / child.in_features))
            nn.init.ones_(self.post_norm.weight)
        else:
            for child in (self.pre_norm, self.in_proj, self.out_proj):
                child.reset_parameters()

    def make_transforms(self) -> list:
        """Preserve each source matrix, bias and normalization key."""
        return []

    def forward(  # pylint: disable=keyword-arg-before-vararg
        self,
        hidden_states: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        *args: Any,
        x: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Normalize grouped patch features before projecting into language hidden width."""
        del args, kwargs
        if hidden_states is not None and x is not None:
            raise ValueError("Provide patch features through hidden_states or x, not both")
        hidden_states = hidden_states if x is None else x
        if hidden_states is None:
            raise ValueError("Spatial patch projection requires input features")
        if self.layout == "post_normalized":
            if isinstance(hidden_states, (list, tuple)):
                lengths = [item.shape[0] for item in hidden_states]
                hidden_states = torch.cat([item.view(item.shape[0], -1) for item in hidden_states])
                return self.post_norm(self.proj(hidden_states)).split(lengths)
            batch = hidden_states.shape[0]
            return self.post_norm(self.proj(hidden_states.view(batch, -1, self.hidden_size)))
        batch = hidden_states.shape[0]
        hidden_states = self.pre_norm(hidden_states).view(batch, -1, self.hidden_size)
        return self.out_proj(self.act(self.in_proj(hidden_states)))


@module_replacement
def SpatialPatchProjection(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Build a complete spatial patch projector through the module-family mechanism."""
    return build_family(
        SpatialPatchProjection, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh
    )


register_module_family(
    ModuleFamily(
        name="spatial_patch_projection",
        baseline=SpatialPatchProjection,
        variants={"local": (
            "hyper_parallel.components.modules.vision.spatial_patch_projection."
            "SpatialPatchProjectionModule"
        )},
        default_variant="local",
        sharding=replicated_sharding,
    )
)
