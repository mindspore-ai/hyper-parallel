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
"""Logical configuration owned by the module-family decoder example."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

from hyper_parallel.components.modules.engram.ngram_hash_mapping import NgramHashMapping


@dataclass(frozen=True)
class DecoderConfig:
    """Global algorithm dimensions independent of a parallel job."""

    hidden_size: int = 8
    num_heads: int = 2
    num_experts: int = 4
    intermediate_size: int = 16
    top_k: int = 2
    model_type: str = "module_family_decoder"
    architectures: tuple[str, ...] = ("FamilyDecoder",)

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_heads", "num_experts", "intermediate_size", "top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads or self.top_k > self.num_experts:
            raise ValueError("heads must divide hidden size and top_k cannot exceed num_experts")


@dataclass(frozen=True)
class MegaGateConfig:
    """Public MegaGate dimensions and scoring semantics, independent of a model adapter."""

    hidden_size: int
    num_experts: int
    top_k: int
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 1.0
    vision_enabled: bool = False
    initializer_range: float = 0.02


@dataclass(frozen=True)
class MHCPostConfig:
    """Residual stream count for the parameter-free post contract."""

    num_stream: int

    def __post_init__(self) -> None:
        if isinstance(self.num_stream, bool) or not isinstance(self.num_stream, int) or self.num_stream <= 0:
            raise ValueError("num_stream must be a positive integer")


@dataclass(frozen=True)
class PipelinedMhcConfig(MHCPostConfig):
    """Logical PipelinedMhc projection and normalization settings."""

    hidden_size: int
    sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("hidden_size", "sinkhorn_iters"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("hc_eps", "norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MHCPreConfig(PipelinedMhcConfig):
    """Pre projection with optional flattened RMS scale."""

    use_gamma: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.use_gamma, bool):
            raise TypeError("use_gamma must be bool")


@dataclass(frozen=True)
class EngramConfig:
    """Shared host/device-table Engram semantics with reproducible hash assets."""

    assets: Mapping[str, Any]
    layer_id: int
    hidden_size: int
    hc_mult: int
    embedding_dim: int
    padded_num_embeddings: int | None = None
    eps: float = 1e-6
    clamp_value: float = 1e-6
    engram_max_pending_entries: int = 1000000
    engram_max_sparse_rows_per_step: int = 1000000

    def __post_init__(self) -> None:
        for name in ("hidden_size", "hc_mult", "embedding_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.layer_id, bool) or not isinstance(self.layer_id, int):
            raise ValueError("layer_id must be an integer")
        for name in ("eps", "clamp_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._initialize_hash_assets()

    def _initialize_hash_assets(self) -> None:
        """Copy hash assets and resolve the table capacity after scalar validation."""
        object.__setattr__(self, "assets", deepcopy(dict(self.assets)))
        mapping = NgramHashMapping(self.assets, self.layer_id)
        rows = self.padded_num_embeddings
        if rows is None:
            rows = mapping.logical_num_embeddings
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < mapping.logical_num_embeddings:
            raise ValueError("padded_num_embeddings must cover all logical hash rows")
        object.__setattr__(self, "padded_num_embeddings", rows)


@dataclass(frozen=True)
class RMSNormConfig:
    """Complete direct-scale RMSNorm construction configuration."""

    hidden_size: int
    eps: float = 1e-5
    rms_norm_cast_order: str = "after_weight"
    fp32_weight: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError("RMSNorm hidden_size must be a positive integer")
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)):
            raise TypeError("RMSNorm eps must be a real scalar")
        if not isfinite(self.eps) or self.eps <= 0:
            raise ValueError("RMSNorm eps must be finite and positive")
        if self.rms_norm_cast_order not in ("before_weight", "after_weight"):
            raise ValueError("rms_norm_cast_order must be before_weight or after_weight")
        if not isinstance(self.fp32_weight, bool):
            raise TypeError("fp32_weight must be bool")


@dataclass(frozen=True)
class SwiGLUMLPConfig:
    """Logical projection sizes and bias policy for a SwiGLU MLP."""

    hidden_size: int
    intermediate_size: int
    bias: bool = False
    swiglu_limit: float = 0.0
    activation_dtype: str | None = None

    def __post_init__(self) -> None:
        for name in ("hidden_size", "intermediate_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.bias, bool):
            raise TypeError("bias must be bool")
        if self.activation_dtype not in (None, "input", "float32"):
            raise ValueError("activation_dtype must be input or float32")


@dataclass(frozen=True, init=False)
class SharedExpertConfig(SwiGLUMLPConfig):
    """Shared SwiGLU projections with an explicit output-bias protocol."""

    return_bias_tuple: bool = False
    num_layers: int = 1
    init_method_std: float = 0.02

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False,
                 return_bias_tuple: bool = False, *, swiglu_limit: float = 0.0,
                 activation_dtype: str | None = None, num_layers: int = 1, init_method_std: float = 0.02) -> None:
        """Keep the shared output protocol positional and activation policy explicit."""
        object.__setattr__(self, "return_bias_tuple", return_bias_tuple)
        object.__setattr__(self, "num_layers", num_layers)
        object.__setattr__(self, "init_method_std", init_method_std)
        super().__init__(hidden_size, intermediate_size, bias, swiglu_limit, activation_dtype)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.return_bias_tuple, bool):
            raise TypeError("return_bias_tuple must be bool")


@dataclass(frozen=True)
class MegaMhcConfig:
    """Four-stream shifted boundary settings; native limits apply to fused execution."""

    hidden_size: int
    hc_eps: float = 1e-6
    norm_eps: float = 1e-6
    num_iters: int = 20
    token_tile: int = 32


@dataclass(frozen=True)
class SpatialMergeProjectorConfig:
    """Channel-major spatial merge with an exact GELU two-layer projector."""

    input_size: int
    hidden_size: int
    output_size: int
    downsample_ratio: int = 2
    bias: bool = True


@dataclass(frozen=True)
class RotaryEmbeddingConfig:
    """Explicit split-half position/scaling contract, without a mutable request cache."""

    dim: int
    base: float = 10000.0
    factor: float = 1.0
    rope_type: str = "default"
    max_position_embeddings: int = 2048
    attention_factor: float = 1.0
    mrope_section: tuple[int, ...] = ()


@dataclass(frozen=True)
class GateConfig:
    """Shared gate contract with optional MegaGate modality correction state."""

    hidden_size: int
    num_experts: int
    top_k: int
    scoring_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    normalize_topk: bool = True
    sorted_indices: bool = False
    initializer_range: float = 0.02

    vision_enabled: bool = False
