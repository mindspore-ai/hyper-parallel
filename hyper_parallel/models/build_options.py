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
"""Model-construction options that do not depend on the Trainer.

This module owns the single authoritative definitions of ``CompileConfig``
and the FSDP2 strategy configuration; ``trainer.config`` only re-exports
them for the YAML layer. Nothing here may import from
``hyper_parallel.trainer`` or ``hyper_parallel.models.trainer``.

It also hosts the accelerator-discovery primitives (``get_device_type`` /
``get_torch_device`` / ``get_device_id``) that model construction needs when
no explicit ``ModelBuildOptions.device`` is given — split out of the former
``components/utils/device.py`` in stage 7 (05 §10.4). The full runtime
device API (synchronize, cache/RNG management, ...) lives in
``hyper_parallel.trainer.runtime.device`` and re-uses these primitives.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch  # pylint: disable=forbidden-backend-import

# Importing torch_npu registers the ``torch.npu`` namespace and HCCL backend.
# The dependency is optional on CUDA-only and CPU-only installations.
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


logger = logging.getLogger(__name__)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = torch_npu is not None and torch.npu.is_available()

if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning("Device namespace '%s' not found in torch, try to load 'torch.cuda'.", device_name)
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


# NOTE: this import must stay *below* the device primitives above.
# ``quantization.config`` triggers the quantization package ``__init__``,
# whose module chain (via ``checkpoint``) reaches modules that import the
# device primitives from here; keeping the primitives defined first makes
# that circular edge resolve against already-defined names.
from hyper_parallel.components.quantization.config import (  # noqa: E402
    LowPrecisionConfig,
)


@dataclass
class CompileConfig:
    """Decoder-layer ``torch.compile`` options exposed by the Trainer."""

    enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    backend: Optional[str] = None
    options: Optional[dict[str, Any]] = None
    dynamo_cache_size_limit: int = 256

    def __post_init__(self) -> None:
        """Validate values that the YAML resolver cannot express precisely."""
        for name in ("enabled", "fullgraph", "dynamic"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"compile.{name} must be a bool")
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("compile.mode must be a non-empty string")
        if self.backend is not None and (
            not isinstance(self.backend, str) or not self.backend.strip()
        ):
            raise ValueError("compile.backend must be None or a non-empty string")
        if self.options is not None and not isinstance(self.options, dict):
            raise TypeError("compile.options must be a mapping or None")
        if self.options and self.mode != "default":
            raise ValueError(
                "compile.options cannot be combined with a non-default compile.mode"
            )
        if (
            isinstance(self.dynamo_cache_size_limit, bool)
            or not isinstance(self.dynamo_cache_size_limit, int)
            or self.dynamo_cache_size_limit <= 0
        ):
            raise ValueError("compile.dynamo_cache_size_limit must be a positive integer")


@dataclass
class FSDP2MixedPrecisionConfig:
    """FSDP2 mixed-precision policy expressed in YAML-friendly strings.

    All dtypes default to None, which means no mixed precision at all;
    the core ``MixedPrecisionPolicy`` then falls back to framework
    defaults. Dtype strings are resolved to platform dtypes when the
    FSDP2 manager builds the core policy.
    """

    param_dtype: Optional[Literal["bfloat16", "float16", "float32"]] = None
    reduce_dtype: Optional[Literal["bfloat16", "float16", "float32"]] = None
    output_dtype: Optional[Literal["bfloat16", "float16", "float32"]] = None
    cast_forward_inputs: bool = True


@dataclass
class FSDP2Config:
    """FSDP2 strategy configuration (06 §4.1)."""
    dp_shard_size: int = 1
    edp_shard_size: int = 1
    replicate_params: list[str] = field(default_factory=list)
    mix_precision: FSDP2MixedPrecisionConfig = field(
        default_factory=FSDP2MixedPrecisionConfig
    )
    enable_offload: bool = False
    reshard_after_forward: bool = True
    reshard_after_backward: bool = True
    requires_grad_sync: bool = True
    backward_prefetch_depth: int = 1
    forward_prefetch_depth: int = 1
    comm_fusion: bool = False
    comm_fusion_zero_copy: Optional[bool] = None

    def __post_init__(self) -> None:
        """Validate topology sizes and prefetch depths."""
        if self.dp_shard_size < 1:
            raise ValueError("dp_shard_size must be greater than or equal to 1")
        if self.edp_shard_size < 1:
            raise ValueError("edp_shard_size must be greater than or equal to 1")
        if self.backward_prefetch_depth < 0:
            raise ValueError("backward_prefetch_depth must be greater than or equal to 0")
        if self.forward_prefetch_depth < 0:
            raise ValueError("forward_prefetch_depth must be greater than or equal to 0")


_MODEL_INIT_DTYPES = ("float16", "bfloat16", "float32")
_ACTIVATION_CHECKPOINT_MODES = ("off", "full", "selective")
_ACTIVATION_SWAP_MODES = ("none", "attention")


@dataclass
class ModelBuildOptions:
    """Trainer-independent options for one model build.

    Mirrors the option surface consumed by the model-construction pipeline
    (``_transformers.model_builder.apply_model_infrastructure``) so a
    programmatic caller can drive the same entry point without
    constructing Trainer DTOs.
    """

    device: Optional[torch.device] = None
    # Final floating-point dtype after weights are loaded or initialized
    # from scratch; None preserves the initialization-path dtype.
    model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None
    activation_checkpoint: Optional[Literal["off", "full", "selective"]] = None
    activation_swap: Literal["none", "attention"] = "none"
    swap_inputs: bool = False
    compile: CompileConfig = field(default_factory=CompileConfig)
    # Validate DTensor placements (skip compile) instead of a production build.
    validate_placement: bool = False
    low_precision: Optional[LowPrecisionConfig] = None

    def __post_init__(self) -> None:
        """Normalize dict/string inputs and validate enum-like fields."""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif self.device is not None and not isinstance(self.device, torch.device):
            raise TypeError(
                "ModelBuildOptions.device must be a torch.device, a device "
                f"string, or None; got {type(self.device).__name__}"
            )
        if (
            self.model_init_dtype is not None
            and self.model_init_dtype not in _MODEL_INIT_DTYPES
        ):
            raise ValueError(
                "model_init_dtype must be one of float16, bfloat16, float32, "
                f"or null; got {self.model_init_dtype!r}"
            )
        if self.activation_checkpoint is not None and (
            self.activation_checkpoint not in _ACTIVATION_CHECKPOINT_MODES
        ):
            raise ValueError(
                "activation_checkpoint must be one of off, full, selective, "
                f"or null; got {self.activation_checkpoint!r}"
            )
        if self.activation_swap not in _ACTIVATION_SWAP_MODES:
            raise ValueError(
                "activation_swap must be one of none, attention; "
                f"got {self.activation_swap!r}"
            )
        for name in ("swap_inputs", "validate_placement"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"ModelBuildOptions.{name} must be a bool")
        if isinstance(self.compile, Mapping):
            self.compile = CompileConfig(**dict(self.compile))
        elif not isinstance(self.compile, CompileConfig):
            raise TypeError(
                "ModelBuildOptions.compile must be a CompileConfig or a "
                f"mapping; got {type(self.compile).__name__}"
            )
        if isinstance(self.low_precision, Mapping):
            self.low_precision = LowPrecisionConfig(**dict(self.low_precision))
        elif self.low_precision is not None and not isinstance(
            self.low_precision, LowPrecisionConfig
        ):
            raise TypeError(
                "ModelBuildOptions.low_precision must be a LowPrecisionConfig, "
                f"a mapping, or None; got {type(self.low_precision).__name__}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelBuildOptions":
        """Normalize a plain mapping (e.g. programmatic kwargs) into options."""
        if not isinstance(data, Mapping):
            raise TypeError(
                "ModelBuildOptions.from_dict requires a mapping; "
                f"got {type(data).__name__}"
            )
        return cls(**dict(data))


def normalize_build_options(
    value: Optional[Any],
) -> ModelBuildOptions:
    """Normalize None / mapping / ModelBuildOptions into ModelBuildOptions.

    This is the AutoModels-side normalization boundary: callers may pass a
    plain dict and receive this package's own options object, never a
    Trainer DTO.
    """
    if value is None:
        return ModelBuildOptions()
    if isinstance(value, ModelBuildOptions):
        return value
    if isinstance(value, Mapping):
        return ModelBuildOptions.from_dict(value)
    raise TypeError(
        "build options must be a ModelBuildOptions, a mapping, or None; "
        f"got {type(value).__name__}"
    )


__all__ = [
    "CompileConfig",
    "FSDP2Config",
    "FSDP2MixedPrecisionConfig",
    "ModelBuildOptions",
    "normalize_build_options",
    "IS_CUDA_AVAILABLE",
    "IS_NPU_AVAILABLE",
    "get_device_type",
    "get_torch_device",
    "get_device_id",
]
