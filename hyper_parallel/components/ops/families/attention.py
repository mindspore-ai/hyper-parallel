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
"""Vision attention over one image using canonical BNSD input."""

from typing import Protocol

import torch
from torch.nn import functional as F

from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward

from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class VisionAttentionFn(Protocol):
    """Compute dropout-free bidirectional attention and return BSND output."""

    def __call__(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Return attention over all supplied valid keys."""


VISION_ATTENTION = OpSpec[VisionAttentionFn](config_key="attention", variant="vision")


def _native(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    accumulator = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        scores = (x.to(accumulator) @ key.to(accumulator).transpose(-1, -2)) * (x.shape[-1] ** -0.5)
        return (scores.softmax(dim=-1) @ value.to(accumulator)).to(x.dtype).transpose(1, 2)


def _sdpa(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(x, key, value, is_causal=False).transpose(1, 2)  # pylint: disable=not-callable


def _npu(x: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output, _ = npu_fusion_attention_forward(
        None, x, key, value, attention_mask=None,
        dropout=0.0, scaling=x.shape[-1] ** -0.5, is_causal=False, sparse_mode=0,
        pre_tokens=2147483647, next_tokens=2147483647,
    )
    return output


def _load_npu() -> VisionAttentionFn:
    require_symbol(require_package("torch_npu"), "npu_fusion_attention")
    return _npu


def register_attention(registry: OpRegistry) -> None:
    """Register reference, Torch SDPA and Ascend vision attention."""
    registry.register_family("attention", specs=(VISION_ATTENTION,))
    registry.register_impl(VISION_ATTENTION, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _native)
    registry.register_impl(VISION_ATTENTION, name="sdpa", devices=("cpu", "cuda"), load=lambda: _sdpa)
    registry.register_impl(VISION_ATTENTION, name="torch_npu", devices=("npu",), load=_load_npu)
