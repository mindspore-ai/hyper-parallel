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
"""Qwen-style gated residual streams with HF-compatible parameter ownership."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.ops import gated_residual
from hyper_parallel.models.replacement import module_replacement


class _GroupedScale(nn.Module):
    """Keep the HF hc_norm.weight checkpoint path and grouped scale semantics."""

    def __init__(self, width: int, group_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.group_size, self.eps = group_size, eps


class GatedResidualModule(nn.Module):
    """Token-local residual read/inject; no Sinkhorn or attention-output gate."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, use_combine: bool = True,
    ) -> None:
        """Construct Qwen parameters or reuse a compatible source module."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        source = module if module is not None else config
        self.hc_count, self.hidden_size = source.hc_count, source.hidden_size
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (self.hc_count, self.hidden_size)):
            raise ValueError("hc_count and hidden_size must be positive integers")
        width = self.hc_count * self.hidden_size
        if module is not None:
            self.hc_norm = module.hc_norm
            if self.hc_norm.group_size != self.hidden_size or self.hc_norm.weight.shape != (width,):
                raise ValueError("source must use per-stream RMSNorm and a full-width offset scale")
            self.input_mix_weight_down = module.input_mix_weight_down
            self.input_mix_weight_up = module.input_mix_weight_up
            self.block_inject_weight = module.block_inject_weight
            self.train(module.training)
        else:
            if isinstance(config.hc_lowrank, bool) or not isinstance(config.hc_lowrank, int) or config.hc_lowrank < 1:
                raise ValueError("hc_lowrank must be a positive integer")
            if not isinstance(use_combine, bool):
                raise TypeError("use_combine must be bool")
            self.hc_norm = _GroupedScale(width, self.hidden_size, config.rms_norm_eps)
            self.input_mix_weight_down = nn.Linear(width, config.hc_lowrank, bias=False)
            self.input_mix_weight_up = nn.Linear(config.hc_lowrank, width, bias=False)
            self.block_inject_weight = nn.Linear(width, self.hc_count, bias=False) if use_combine else None
        for projection in (self.input_mix_weight_down, self.input_mix_weight_up, self.block_inject_weight):
            if projection is not None and (not isinstance(projection, nn.Linear) or projection.bias is not None):
                raise TypeError("gated residual requires bias-free linear projections")

    def forward(self, hyper_input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match HF: mixed, or (mixed, original residual, injection weights)."""
        gate = None if self.block_inject_weight is None else self.block_inject_weight.weight
        result = gated_residual(
            hyper_input, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight,
            self.hc_norm.weight, gate, channels=self.hc_count, eps=self.hc_norm.eps,
        )
        if gate is None:
            return result
        mixed, inject = result
        return mixed, hyper_input, inject

    @staticmethod
    def combine(output: torch.Tensor, residual: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        """Inject a sublayer output back into the original residual streams."""
        return residual + (output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)


@module_replacement
def GatedResidual(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None,
    mesh: Any = None, use_combine: bool = True,
) -> nn.Module:
    """Build from config or retain an HF module's loaded parameter objects."""
    return build_family(GatedResidual, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, use_combine=use_combine)


register_module_family(ModuleFamily(
    name="gated_residual", baseline=GatedResidual,
    variants={"local": "hyper_parallel.components.modules.gated_residual.GatedResidualModule"},
    default_variant="local",
))
