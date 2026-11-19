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
"""Four-stream shifted mHC boundary with ordinary and multicore execution."""
from __future__ import annotations

from math import isfinite
from typing import Any

import torch
from torch import nn



class MegaMhcModule(nn.Module):
    """Own predictors and input normalization; advance the complete shifted boundary."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Initialize logical settings or retain compatible source Parameters."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        source = config if config is not None else module
        self._set_settings(source)
        shapes = {"phi": (24, 4 * self.hidden_size), "alpha": (3,), "bias": (24,),
                  "norm_weight": (self.hidden_size,)}
        for name, shape in shapes.items():
            dtype = torch.bfloat16 if name == "norm_weight" else torch.float32
            if module is None:
                parameter = nn.Parameter(torch.empty(shape, dtype=dtype))
            else:
                parameter = getattr(module, name, None)
                if (not isinstance(parameter, nn.Parameter) or tuple(parameter.shape) != shape
                        or parameter.dtype != dtype):
                    raise ValueError(f"source {name} must have shape {shape} and dtype {dtype}")
            setattr(self, name, parameter)
        if len({parameter.device for parameter in self.parameters()}) != 1:
            raise ValueError("MegaMhc parameters must share a device")
        if module is None:
            self.reset_parameters()
        else:
            self.train(module.training)

    def _set_settings(self, source: Any) -> None:
        """Validate and retain dimensions, normalization and tiling settings."""
        for name in ("hidden_size", "num_iters"):
            value = getattr(source, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        for name in ("hc_eps", "norm_eps"):
            value = getattr(source, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        self.token_tile = getattr(source, "token_tile", 32)
        if isinstance(self.token_tile, bool) or not isinstance(self.token_tile, int) or self.token_tile < 1:
            raise ValueError("token_tile must be a positive integer")

    def reset_parameters(self) -> None:
        """Restore predictor and normalization initialization after materialization."""
        nn.init.normal_(self.phi, std=(4 * self.hidden_size) ** -0.5)
        nn.init.ones_(self.alpha)
        nn.init.zeros_(self.bias)
        nn.init.ones_(self.norm_weight)

    def _apply(self, fn: Any, recurse: bool = True) -> MegaMhcModule:
        """Preserve declared FP32 parameters and gradients during conversion."""
        # The predictor uses FP32 even when the enclosing model uses BF16.
        protected = {id(getattr(self, name)) for name in ("phi", "alpha", "bias")}
        protected.update(id(parameter.grad) for parameter in self.parameters() if parameter.grad is not None
                         and id(parameter) in protected)

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            converted = fn(tensor)
            if id(tensor) not in protected or converted.dtype == torch.float32:
                return converted
            return converted.float() if tensor.is_meta else tensor.to(device=converted.device, dtype=torch.float32)

        return super()._apply(apply, recurse=recurse)

    def forward(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return residual, three next mappings and normalized next-block input."""
        if residual.shape[-2:] != (4, self.hidden_size):
            raise ValueError("residual must end in [4, hidden_size]")
        prefix = residual.shape[:-2]
        expected = ((previous_output, (*prefix, self.hidden_size)), (previous_pre_mix, (*prefix, 4)),
                    (previous_post_mix, (*prefix, 4)), (previous_residual_mix, (*prefix, 4, 4)))
        if any(tuple(value.shape) != shape for value, shape in expected):
            raise ValueError("shifted boundary tensors must share the same token dimensions")
        updated = (torch.einsum("...ji,...jh->...ih", previous_residual_mix.float(), residual.float())
                   + previous_post_mix.float().unsqueeze(-1) * previous_output.float().unsqueeze(-2)).to(residual.dtype)
        updated_fp32 = updated.float()
        variance = updated_fp32.square().mean((-2, -1), keepdim=True)
        normalized = updated_fp32 * torch.rsqrt(variance + self.norm_eps)
        logits = normalized.flatten(-2) @ self.phi.float().T
        pre, post, mixing = logits.split((4, 4, 16), dim=-1)
        pre_bias, post_bias, mixing_bias = self.bias.float().split((4, 4, 16))
        pre = torch.sigmoid(pre * self.alpha[0] + pre_bias) + self.hc_eps
        post = 2 * torch.sigmoid(post * self.alpha[1] + post_bias)
        mixing = (mixing * self.alpha[2] + mixing_bias).reshape(*prefix, 4, 4).softmax(-1) + self.hc_eps
        mixing = mixing / (mixing.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.num_iters - 1):
            mixing = mixing / (mixing.sum(-1, keepdim=True) + self.hc_eps)
            mixing = mixing / (mixing.sum(-2, keepdim=True) + self.hc_eps)
        block_input = (updated.float() * previous_pre_mix.float().unsqueeze(-1)).sum(-2)
        block_input = block_input * torch.rsqrt(block_input.square().mean(-1, keepdim=True) + self.norm_eps)
        block_input = (block_input * self.norm_weight.float()).to(residual.dtype)
        return updated, pre, post, mixing, block_input


class FusedMegaMhcModule(MegaMhcModule):
    """Execute the existing multicore kernel with its unchanged native contract."""

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Validate native training dimensions and fixed Sinkhorn iterations before execution."""
        source = config if config is not None else module
        if source is not None:
            # The optional native package owns its supported training dimensions.
            from hyper_multicore.modules.mega_mhc_common import (  # pylint: disable=import-outside-toplevel
                validate_mega_mhc_hidden_size,
            )
            validate_mega_mhc_hidden_size(source.hidden_size, need_backward=True)
            if source.num_iters != 20:
                raise ValueError("fused MegaMhc requires num_iters=20")
        super().__init__(config=config, module=module)

    def forward(self, previous_output: torch.Tensor, residual: torch.Tensor,
                previous_pre_mix: torch.Tensor, previous_post_mix: torch.Tensor,
                previous_residual_mix: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Delegate native shape/dtype validation and execution to multicore."""
        # Native operators are optional and loaded only when fused execution is requested.
        from hyper_multicore.modules.mega_mhc.function import hyper_mega_mhc  # pylint: disable=C0415
        return hyper_mega_mhc(previous_output, residual, previous_pre_mix, previous_post_mix,
                              previous_residual_mix, self.phi, self.alpha, self.bias, self.norm_weight,
                              hc_eps=self.hc_eps, norm_eps=self.norm_eps, num_iters=self.num_iters,
                              token_tile=self.token_tile)
