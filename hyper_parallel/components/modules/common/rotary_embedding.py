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

from math import ceil, floor, isfinite, log, pi
from typing import Any

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


def _yarn_frequencies(parameters: dict, dim: int, base: float, factor: float) -> torch.Tensor:
    """Build the static interpolation/extrapolation blend from its declared rotation range."""
    original = parameters.get("original_max_position_embeddings")
    if isinstance(original, bool) or not isinstance(original, int) or original < 1:
        raise ValueError("YaRN requires positive original_max_position_embeddings")
    fast, slow = parameters.get("beta_fast") or 32, parameters.get("beta_slow") or 1
    if fast <= 0 or slow <= 0 or base <= 1:
        raise ValueError("YaRN requires positive beta values and rope base greater than one")
    low = dim * log(original / (fast * 2 * pi)) / (2 * log(base))
    high = dim * log(original / (slow * 2 * pi)) / (2 * log(base))
    if parameters.get("truncate", True):
        low, high = floor(low), ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32, device="cpu") - low) / (high - low)).clamp(0, 1)
    extrapolation = 1 - ramp
    periods = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
    return (1.0 / (factor * periods)) * (1 - extrapolation) + (1.0 / periods) * extrapolation


class RotaryEmbeddingModule(nn.Module):
    """Return split-half cos/sin; standard, linear and dynamic NTK have distinct validated semantics.

    Dynamic NTK takes an explicit logical sequence extent. It does not depend on
    a mutable cache left by a preceding request or checkpoint recomputation.
    Multi-axis sections select axis frequencies before the split-half duplication.
    """

    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 output_format: str | None = None, mrope_interleaved: bool | None = None) -> None:
        """Build from logical config or adopt a supported static source frequency buffer."""
        super().__init__()
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self.config = config if module is None else getattr(module, "config", None)
        source = config if module is None else module
        self.output_format = (getattr(source, "output_format", "cos_sin")
                              if output_format is None else output_format)
        if self.output_format not in ("cos_sin", "complex"):
            raise ValueError("Rotary output_format must be cos_sin or complex")
        if mrope_interleaved is not None and not isinstance(mrope_interleaved, bool):
            raise TypeError("mrope_interleaved must be bool or None")
        if module is not None:
            self._initialize_source(module, mrope_interleaved=mrope_interleaved)
        else:
            self.dim = config.dim
            self.base = config.base
            self.factor = config.factor
            self.rope_type = config.rope_type
            self.max_position_embeddings = config.max_position_embeddings
            self.attention_factor = config.attention_factor
            self.mrope_section = tuple(config.mrope_section)
            self.mrope_interleaved = (getattr(config, "mrope_interleaved", False)
                                      if mrope_interleaved is None else mrope_interleaved)
            self._validate()
            target = torch.empty(0).device
            frequency = self.base ** (-torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu") / self.dim)
            if self.rope_type == "linear":
                frequency = frequency / self.factor
            elif self.rope_type == "yarn":
                parameters = getattr(config, "rope_parameters", None)
                if not parameters:
                    raise ValueError("YaRN direct construction requires explicit rope_parameters")
                frequency = _yarn_frequencies(parameters, self.dim, self.base, self.factor)
            register_rebuildable_buffer(self, "inv_freq", value=frequency, persistent=False)
            self.inv_freq = self.inv_freq.to(target)
        self._validate()

    def _initialize_source(self, module: nn.Module, *, mrope_interleaved: bool | None) -> None:
        """Adopt static frequency buffers and retain their materialization rebuild recipes."""
        source_config = getattr(module, "config", None)
        params = getattr(source_config, "rope_parameters", None) or getattr(source_config, "rope_scaling", {}) or {}
        self.rope_type = getattr(module, "rope_type", params.get("rope_type", "default"))
        if self.rope_type not in ("default", "linear", "yarn"):
            raise ValueError("source conversion supports static default/linear/YaRN RoPE; "
                             "dynamic cache needs explicit config")
        frequency = getattr(module, "inv_freq", None)
        if not isinstance(frequency, torch.Tensor) or frequency.ndim != 1:
            raise ValueError("source requires a one-dimensional inv_freq buffer")
        if any(True for _ in module.parameters()):
            raise ValueError("parameterized rotary sources require a separate state contract")
        self.dim = 2 * frequency.numel()
        self.base = float(params.get("rope_theta", getattr(source_config, "rope_theta", 10000.0)))
        self.factor = float(params.get("factor", 1.0))
        self.max_position_embeddings = int(getattr(source_config, "max_position_embeddings", 2048))
        self.attention_factor = float(getattr(module, "attention_scaling", 1.0))
        self.mrope_section = tuple(getattr(module, "mrope_section", params.get("mrope_section", ())))
        source_interleaved = getattr(module, "mrope_interleaved", params.get("mrope_interleaved"))
        if mrope_interleaved is None:
            mrope_interleaved = source_interleaved
        if self.mrope_section and mrope_interleaved is None:
            raise ValueError("multi-axis source requires an explicit mrope_interleaved contract")
        if mrope_interleaved is not None and not isinstance(mrope_interleaved, bool):
            raise TypeError("mrope_interleaved must be bool or None")
        self.mrope_interleaved = False if mrope_interleaved is None else mrope_interleaved
        rebuilt = self._rebuild_source_frequency(module, source_config, params, frequency)
        register_rebuildable_buffer(self, "inv_freq", value=rebuilt, persistent=False)
        self.inv_freq = frequency
        original = getattr(module, "original_inv_freq", None)
        if isinstance(original, torch.Tensor):
            if original.shape != frequency.shape:
                raise ValueError("original rotary frequency shape must match inv_freq")
            value = rebuilt.clone() if original.is_meta else original.detach().cpu()
            register_rebuildable_buffer(self, "original_inv_freq", value=value, persistent=False)
            self.original_inv_freq = original
        self.train(module.training)

    def _rebuild_source_frequency(self, module: nn.Module, source_config: Any, params: dict,
                                  frequency: torch.Tensor) -> torch.Tensor:
        """Reconstruct only meta frequencies with the audited source arithmetic order."""
        if frequency.is_meta:
            if isinstance(module, RotaryEmbeddingModule):
                rebuilt = self.base ** (-torch.arange(0, self.dim, 2, device="cpu").float() / self.dim)
            elif callable(getattr(module, "compute_default_rope_parameters", None)):
                with torch.device("cpu"):
                    rebuilt, _ = module.compute_default_rope_parameters(source_config, device="cpu")
                if rebuilt.is_meta or rebuilt.shape != frequency.shape:
                    raise ValueError("source static rotary builder must produce the declared CPU frequency shape")
                rebuilt = rebuilt.detach().cpu()
            else:
                raise ValueError("meta rotary source must declare a supported frequency construction contract")
            if self.rope_type == "linear":
                rebuilt = rebuilt / self.factor
            elif self.rope_type == "yarn":
                rebuilt = _yarn_frequencies(params, self.dim, self.base, self.factor)
        else:
            rebuilt = frequency.detach().cpu()
        return rebuilt


    @staticmethod

    def compute_default_rope_parameters(config: Any, **kwargs: Any) -> tuple[torch.Tensor, float]:
        """Expose the static HF initialization protocol using component-owned math."""
        del kwargs
        params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(dim * params.get("partial_rotary_factor", 1.0))
        base = params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        frequency = 1.0 / (base ** (torch.arange(0, dim, 2, device="cpu").float() / dim))
        return frequency, 1.0

    def _validate(self) -> None:
        """Validate supported frequency and position layouts before execution."""
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim < 2 or self.dim % 2:
            raise ValueError("rotary dim must be a positive even integer")
        if self.rope_type not in ("default", "linear", "dynamic", "yarn"):
            raise ValueError("supported rope_type values are default, linear, dynamic and yarn")
        for name in ("base", "factor", "attention_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.max_position_embeddings, bool)
                or not isinstance(self.max_position_embeddings, int) or self.max_position_embeddings < 1):
            raise ValueError("max_position_embeddings must be positive")
        if self.output_format not in ("cos_sin", "complex"):
            raise ValueError("Rotary output_format must be cos_sin or complex")
        self._validate_position_layout()

    def _validate_position_layout(self) -> None:
        """Validate dynamic NTK extent constraints and multi-axis section coverage."""
        if self.rope_type == "dynamic" and (self.dim <= 2 or self.factor < 1):
            raise ValueError("dynamic NTK requires dim > 2 and factor >= 1")
        if not isinstance(self.mrope_interleaved, bool):
            raise ValueError("mrope_interleaved must be boolean")
        if self.mrope_interleaved and len(self.mrope_section) != 3:
            raise ValueError("interleaved multimodal RoPE requires three axis sections")
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
        if self.mrope_interleaved and position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        angles = position_ids.float().unsqueeze(-1) * frequency
        if position_ids.ndim == 3 and self.mrope_section:
            if position_ids.shape[0] != len(self.mrope_section):
                raise ValueError("position axes must match mrope_section")
            if self.mrope_interleaved:
                axes = angles
                angles = axes[0].clone()
                for axis in (1, 2):
                    indices = slice(axis, self.mrope_section[axis] * 3, 3)
                    angles[..., indices] = axes[axis, ..., indices]
            else:
                sections = angles.split(self.mrope_section, dim=-1)
                angles = torch.cat([section[axis] for axis, section in enumerate(sections)], dim=-1)
        elif position_ids.ndim != 2 or self.mrope_section:
            raise ValueError("positions must be [B,S], or sectioned [axes,B,S] with mrope_section")
        if self.output_format == "complex":
            return torch.polar(torch.ones_like(angles), angles) * self.attention_factor
        embedding = torch.cat((angles, angles), dim=-1)
        return ((embedding.cos() * self.attention_factor).to(x.dtype),
                (embedding.sin() * self.attention_factor).to(x.dtype))


@module_replacement
def RotaryEmbedding(  # pylint: disable=invalid-name
    *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "",
    context: Any = None, mesh: Any = None, output_format: str | None = None,
    mrope_interleaved: bool | None = None,
) -> nn.Module:
    """Construct rotary math with explicit output and multi-axis layout contracts.

    Args:
        config: Logical settings for direct construction, exclusive with module.
        module: Source frequencies and state to adopt.
        module_fqn: Replacement scope supplied by the framework.
        context: Family construction context.
        mesh: Existing job mesh.
        output_format: Complex factors or cos/sin; None retains declared source metadata.
        mrope_interleaved: Interleaved multi-axis channels; sectioned sources must declare their layout.
    """
    return build_family(RotaryEmbedding, config=config, module=module, module_fqn=module_fqn,
                        context=context, mesh=mesh, output_format=output_format,
                        mrope_interleaved=mrope_interleaved)


register_module_family(ModuleFamily(
    name="rotary_embedding", baseline=RotaryEmbedding,
    variants={"local": "hyper_parallel.components.modules.common.rotary_embedding.RotaryEmbeddingModule"},
    default_variant="local",
))
