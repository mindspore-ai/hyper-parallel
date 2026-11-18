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
"""Explicit rotary-position mathematics and rebuildable frequency state."""
from __future__ import annotations

from math import isfinite
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


class RotaryEmbeddingModule(nn.Module):
    """Return split-half cos/sin; standard, linear and dynamic NTK have distinct validated semantics.

    Dynamic NTK takes an explicit logical sequence extent. It does not depend on
    a mutable cache left by a preceding request or checkpoint recomputation.
    Multi-axis sections select axis frequencies before the split-half duplication.
    """

    def __init__(self, *, config: Any = None, module: nn.Module | None = None) -> None:
        """Build from logical config or adopt a supported static source frequency buffer."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        if module is not None:
            source_config = getattr(module, "config", None)
            params = getattr(source_config, "rope_parameters", None) or getattr(source_config, "rope_scaling", {}) or {}
            self.rope_type = getattr(module, "rope_type", params.get("rope_type", "default"))
            if self.rope_type not in ("default", "linear"):
                raise ValueError("source conversion supports static default/linear RoPE; "
                                 "dynamic cache needs explicit config")
            frequency = getattr(module, "inv_freq", None)
            if not isinstance(frequency, torch.Tensor) or frequency.ndim != 1 or frequency.is_meta:
                raise ValueError("source requires a materialized one-dimensional inv_freq buffer")
            if any(True for _ in module.parameters()):
                raise ValueError("parameterized rotary sources require a separate state contract")
            self.dim = 2 * frequency.numel()
            self.base = float(params.get("rope_theta", getattr(source_config, "rope_theta", 10000.0)))
            self.factor = float(params.get("factor", 1.0))
            self.max_position_embeddings = int(getattr(source_config, "max_position_embeddings", 2048))
            self.attention_factor = float(getattr(module, "attention_scaling", 1.0))
            self.mrope_section = tuple(getattr(module, "mrope_section", params.get("mrope_section", ())))
            if self.mrope_section:
                raise ValueError("multi-axis source requires an explicit position layout config")
            register_rebuildable_buffer(self, "inv_freq", value=frequency.detach().cpu(), persistent=False)
            self.inv_freq = frequency
            self.train(module.training)
        else:
            self.dim = config.dim
            self.base = config.base
            self.factor = config.factor
            self.rope_type = config.rope_type
            self.max_position_embeddings = config.max_position_embeddings
            self.attention_factor = config.attention_factor
            self.mrope_section = tuple(config.mrope_section)
            self._validate()
            target = torch.empty(0).device
            frequency = self.base ** (-torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu") / self.dim)
            if self.rope_type == "linear":
                frequency = frequency / self.factor
            register_rebuildable_buffer(self, "inv_freq", value=frequency, persistent=False)
            self.inv_freq = self.inv_freq.to(target)
        self._validate()

    def _validate(self) -> None:
        """Validate supported frequency and position layouts before execution."""
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 2 or self.dim % 2:
            raise ValueError("rotary dim must be a positive even integer")
        if self.rope_type not in ("default", "linear", "dynamic"):
            raise ValueError("supported rope_type values are default, linear and dynamic")
        for name in ("base", "factor", "attention_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.max_position_embeddings, bool)
                or not isinstance(self.max_position_embeddings, int) or self.max_position_embeddings < 1):
            raise ValueError("max_position_embeddings must be positive")
        self._validate_position_layout()

    def _validate_position_layout(self) -> None:
        """Validate dynamic NTK extent constraints and multi-axis section coverage."""
        if self.rope_type == "dynamic" and (self.dim <= 2 or self.factor < 1):
            raise ValueError("dynamic NTK requires dim > 2 and factor >= 1")
        if self.mrope_section and (
                any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in self.mrope_section)
                or sum(self.mrope_section) != self.dim // 2):
            raise ValueError("mrope_section must partition the half-dimension into positive axis sections")

    def _apply(self, fn: Any, recurse: bool = True) -> RotaryEmbeddingModule:
        # Frequencies must not be rounded by model-wide BF16 conversion.
        original = self.inv_freq
        super()._apply(fn, recurse=recurse)
        converted = self.inv_freq
        if converted.dtype != torch.float32:
            self.inv_freq = (converted.float() if original.is_meta
                             else original.to(device=converted.device, dtype=torch.float32))
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor,
                *, sequence_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin in x.dtype for [B,S] or explicitly sectioned [axes,B,S] positions.

        Args:
            x: Activation supplying output device and dtype.
            position_ids: Actual logical positions, including caller-owned packed offsets.
            sequence_length: Required global context extent for dynamic NTK; no host read of device positions.

        Returns:
            Cosine and sine tensors shaped [B,S,dim].
        """
        if position_ids.device != x.device or self.inv_freq.device != x.device:
            raise ValueError("rotary buffers, positions and activations must share a device")
        frequency = self.inv_freq.float()
        if self.rope_type == "dynamic":
            if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
                raise ValueError("dynamic NTK requires explicit positive sequence_length")
            extent = max(sequence_length, self.max_position_embeddings)
            adjusted = self.base * (
                self.factor * extent / self.max_position_embeddings - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            frequency = adjusted ** (-torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
        # Explicit elementwise FP32 multiplication is unaffected by GEMM autocast.
        angles = position_ids.float().unsqueeze(-1) * frequency
        if position_ids.ndim == 3 and self.mrope_section:
            if position_ids.shape[0] != len(self.mrope_section):
                raise ValueError("position axes must match mrope_section")
            sections = angles.split(self.mrope_section, dim=-1)
            angles = torch.cat([section[axis] for axis, section in enumerate(sections)], dim=-1)
        elif position_ids.ndim != 2 or self.mrope_section:
            raise ValueError("positions must be [B,S], or sectioned [axes,B,S] with mrope_section")
        embedding = torch.cat((angles, angles), dim=-1)
        return ((embedding.cos() * self.attention_factor).to(x.dtype),
                (embedding.sin() * self.attention_factor).to(x.dtype))


@module_replacement
def RotaryEmbedding(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None,
) -> nn.Module:
    """Construct declared rotary mathematics; unsupported scaling schemes fail explicitly."""
    return build_family(RotaryEmbedding, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="rotary_embedding", baseline=RotaryEmbedding,
    variants={"local": "hyper_parallel.components.modules.rotary_embedding.RotaryEmbeddingModule"},
    default_variant="local",
))
