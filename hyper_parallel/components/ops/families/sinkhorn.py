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
"""Sinkhorn normalization of mHC residual mixing logits."""

from typing import Protocol

import torch

from hyper_parallel.components.functional.sinkhorn import sinkhorn, sinkhorn_knopps

from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class SinkhornFn(Protocol):
    """Normalize square mixing matrices without changing their shape."""

    def __call__(self, x: torch.Tensor, sinkhorn_iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
        """Return the normalized matrix."""


SINKHORN = OpSpec[SinkhornFn](config_key="sinkhorn")


def _load_omni() -> SinkhornFn:
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_sinkhorn")
    require_symbol(torch.ops.custom, "npu_sinkhorn_grad")
    return sinkhorn


def register_sinkhorn(registry: OpRegistry) -> None:
    """Register basic Torch normalization and Omni's fused implementation."""
    registry.register_family("sinkhorn", specs=(SINKHORN,))
    registry.register_impl(SINKHORN, name="native", devices=("cpu", "cuda", "npu"), load=lambda: sinkhorn_knopps)
    registry.register_impl(SINKHORN, name="omni", devices=("npu",), load=_load_omni)
