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
"""RMSNorm contracts and implementations."""

from typing import Protocol

import torch

from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class RMSNormFn(Protocol):
    """Normalize the last dimension and apply the caller-owned scale."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Return the normalized and scaled tensor."""


RMS_NORM = OpSpec[RMSNormFn](config_key="rms_norm")
RMS_NORM_FP32_AFFINE = OpSpec[RMSNormFn](config_key="rms_norm", variant="fp32_affine")


def _native(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    accumulation_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    values = x.to(accumulation_dtype)
    normalized = values * (values.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (normalized * weight.to(accumulation_dtype)).to(x.dtype)


def _native_fp32_affine(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _native(x.float(), weight.float(), eps).to(x.dtype)


def _load_native() -> RMSNormFn:
    return _native


def _load_native_fp32_affine() -> RMSNormFn:
    return _native_fp32_affine


def _load_npu() -> RMSNormFn:
    require_symbol(require_package("torch_npu"), "npu_rms_norm")
    # The existing implementation imports the optional NPU package.
    from hyper_parallel.components.functional.rms_norm import rms_norm  # pylint: disable=import-outside-toplevel
    return rms_norm


def _load_npu_fp32_affine() -> RMSNormFn:
    kernel = _load_npu()

    def normalize(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return kernel(x.float(), weight.float(), eps).to(x.dtype)

    return normalize


def register_rms_norm(registry: OpRegistry) -> None:
    """Register the standard and FP32-affine RMSNorm contracts."""
    registry.register_family("rms_norm", specs=(RMS_NORM, RMS_NORM_FP32_AFFINE))
    registry.register_impl(RMS_NORM, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(RMS_NORM, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        RMS_NORM_FP32_AFFINE, name="native", devices=("cuda", "cpu", "npu"), load=_load_native_fp32_affine,
    )
    registry.register_impl(RMS_NORM_FP32_AFFINE, name="torch_npu", devices=("npu",), load=_load_npu_fp32_affine)
