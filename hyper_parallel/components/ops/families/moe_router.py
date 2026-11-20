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
"""Compute DeepSeek V4.1 learned top-k routing before training statistics."""

from __future__ import annotations

from typing import Protocol

import torch
from torch.nn import functional as F

from .._dependency import require_package
from ..registry import ImplementationUnavailable, OpRegistry
from ..spec import OpSpec


class MoeRouterFn(Protocol):
    """Return logits, normalized routing weights and selected expert indices."""

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
        image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project tokens and perform expert selection with modality corrections."""


MOE_ROUTER = OpSpec[MoeRouterFn](config_key="moe_router", variant="deepseek_v41")


def router_scores(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Convert router logits to the released nonnegative expert scores."""
    if scoring_func == "sqrtsoftplus":
        return F.softplus(logits).sqrt()  # pylint: disable=not-callable
    if scoring_func == "softmax":
        return logits.softmax(dim=-1)
    if scoring_func == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"Unsupported V4.1 router scoring function: {scoring_func!r}")


def _torch_router(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
    image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.linear(x.reshape(-1, x.shape[-1]).float(), weight.float())  # pylint: disable=not-callable
    scores = router_scores(logits, scoring_func)
    if image_mask is not None and vision_bias is not None:
        bias = torch.where(image_mask.reshape(-1, 1), vision_bias.unsqueeze(0), bias.unsqueeze(0))
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    return logits, weights * scale, indices


def _load_megagate() -> MoeRouterFn:
    require_package("hyper_multicore")
    # Only the selected MegaGate backend needs the standalone native package.
    # pylint: disable=import-outside-toplevel
    from hyper_multicore._loader import NativeComponentUnavailableError
    from hyper_multicore.modules.mega_gate.function import mega_gate
    from hyper_multicore.modules.mega_gate.module import _plan_for_device
    from hyper_multicore.torch.ops import _load_native
    # pylint: enable=import-outside-toplevel
    try:
        _load_native()
    except NativeComponentUnavailableError as error:
        raise ImplementationUnavailable(str(error)) from error

    def route(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, vision_bias: torch.Tensor | None,
        image_mask: torch.Tensor | None, top_k: int, scoring_func: str, scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del scoring_func
        return mega_gate(
            x, weight, bias, _plan_for_device(x.device), vision_bias=vision_bias,
            image_mask=None if image_mask is None else image_mask.reshape(-1).contiguous(),
            top_k=top_k, routed_scaling_factor=scale,
        )

    return route


def register_moe_router(registry: OpRegistry) -> None:
    """Register DeepSeek's Torch router and MegaGate route kernel."""
    registry.register_family("moe_router", specs=(MOE_ROUTER,))
    registry.register_impl(MOE_ROUTER, name="native", devices=("cpu", "cuda", "npu"), load=lambda: _torch_router)
    registry.register_impl(MOE_ROUTER, name="megagate", devices=("npu",), load=_load_megagate)
