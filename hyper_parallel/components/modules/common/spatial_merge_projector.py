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
"""Pixel-unshuffle spatial merging followed by a two-layer GELU projector."""
from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.models.replacement import module_replacement


class SpatialMergeProjectorModule(nn.Module):
    """Own V4.1-compatible channel-major spatial merging and its projections."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Construct logical dimensions or reuse a compatible source's loaded projections."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            self.downsample_ratio = module.downsample_ratio
            if not all(isinstance(getattr(module, key, None), nn.Linear) for key in ("w1", "w2")):
                raise TypeError("spatial projector source requires w1/w2 linear projections")
            self.w1, self.w2 = module.w1, module.w2
            self.train(module.training)
        else:
            self.downsample_ratio = config.downsample_ratio
            for key in ("input_size", "hidden_size", "output_size"):
                value = getattr(config, key)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
            if not isinstance(config.bias, bool):
                raise TypeError("bias must be bool")
            self._validate_ratio()
            self.w1 = nn.Linear(config.input_size * self.downsample_ratio**2, config.hidden_size, bias=config.bias)
            self.w2 = nn.Linear(config.hidden_size, config.output_size, bias=config.bias)
        self._validate_ratio()
        if self.w1.in_features % self.downsample_ratio**2 or self.w1.out_features != self.w2.in_features:
            raise ValueError("spatial projector dimensions do not match the merge ratio and hidden projection")

    def _validate_ratio(self) -> None:
        """Reject fractional, boolean or nonpositive spatial ratios."""
        if (isinstance(self.downsample_ratio, bool) or not isinstance(self.downsample_ratio, int)
                or self.downsample_ratio < 1):
            raise ValueError("downsample_ratio must be a positive integer")

    def forward(self, hidden_states: Tensor, grid_height: int | None = None,
                grid_width: int | None = None, *, input_is_merged: bool = False) -> Tensor:
        """Project a full image grid or already merged channel-major output rows.

        Args:
            hidden_states: Image patch rows, or [output_rows, ratio**2 * input_size].
            grid_height: Full image patch height for ordinary spatial merging.
            grid_width: Full image patch width for ordinary spatial merging.
            input_is_merged: Use preassembled independent spatial blocks. This
                retains the module forward boundary for FSDP and checkpointing.
        """
        if input_is_merged:
            if grid_height is not None or grid_width is not None:
                raise ValueError("premerged projector inputs do not take image grid dimensions")
            if hidden_states.ndim != 2 or hidden_states.shape[1] != self.w1.in_features:
                raise ValueError("premerged projector rows must match w1.in_features")
            return self.w2(F.gelu(self.w1(hidden_states)))  # pylint: disable=not-callable

        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in (grid_height, grid_width)):
            raise ValueError("grid dimensions must be positive integers")
        channels = self.w1.in_features // self.downsample_ratio**2
        if hidden_states.ndim != 2 or hidden_states.shape != (grid_height * grid_width, channels):
            raise ValueError("features must have shape [grid_height * grid_width, input_size]")
        ratio = self.downsample_ratio
        grid = hidden_states.reshape(grid_height, grid_width, channels).permute(2, 0, 1)
        grid = F.pad(grid, (0, -grid_width % ratio, 0, -grid_height % ratio))
        merged = F.unfold(grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(merged)))  # pylint: disable=not-callable


@module_replacement
def SpatialMergeProjector(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None,
) -> nn.Module:
    """Build the spatial-merging family; window/LayerNorm mergers require a different contract."""
    return build_family(SpatialMergeProjector, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="spatial_merge_projector", baseline=SpatialMergeProjector,
    variants={"local":
              "hyper_parallel.components.modules.common.spatial_merge_projector.SpatialMergeProjectorModule"},
    default_variant="local",
))
