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
"""Bias-free expert-major grouped SwiGLU."""

from __future__ import annotations

from typing import Protocol

import torch
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..registry import OpRegistry
from ..spec import OpSpec
from .grouped_matmul import check_torch_grouped_mm


class GroupedSwiGLUFn(Protocol):
    """Apply gate, up and down projections to the token rows assigned to each expert."""

    def __call__(
        self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
        counts: torch.Tensor, scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run bias-free SwiGLU, weighting intermediates before the down projection if requested."""


GROUPED_SWIGLU = OpSpec[GroupedSwiGLUFn](config_key="grouped_swiglu")


def _weighted(x: torch.Tensor, scores: torch.Tensor | None) -> torch.Tensor:
    return x if scores is None else x * scores.unsqueeze(-1)


def _empty_output(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, scores: torch.Tensor | None,
) -> torch.Tensor:
    zeros = w1[:0].sum() + w2[:0].sum() + w3[:0].sum()
    if scores is not None:
        zeros = zeros + scores.sum()
    return x + zeros.to(x.dtype)


def native_grouped_swiglu(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
    counts: torch.Tensor, scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SwiGLU one expert at a time, keeping empty experts in the gradient graph."""
    if x.shape[0] == 0:
        return _empty_output(x, w1, w2, w3, scores)
    tokens = x.split(counts.tolist())
    routing = [None] * len(tokens) if scores is None else scores.split([part.shape[0] for part in tokens])
    parts = [
        _weighted(F.silu(part @ gate.T) * (part @ up.T), score) @ down.T
        for part, gate, down, up, score in zip(tokens, w1.unbind(), w2.unbind(), w3.unbind(), routing)
    ]
    return torch.cat(parts)


def _cuda(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
    counts: torch.Tensor, scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply grouped expert SwiGLU with optional intermediate routing weights."""
    if x.shape[0] == 0:
        return _empty_output(x, w1, w2, w3, scores)
    offs = counts.cumsum(0, dtype=torch.int32)
    # Transpose views preserve the parameter storage across pending forwards.
    grouped_mm = torch._grouped_mm  # pylint: disable=protected-access
    gate = grouped_mm(x, w1.transpose(1, 2), offs=offs)
    up = grouped_mm(x, w3.transpose(1, 2), offs=offs)
    return grouped_mm(_weighted(F.silu(gate) * up, scores), w2.transpose(1, 2), offs=offs)


def _load_cuda() -> GroupedSwiGLUFn:
    require_symbol(torch, "_grouped_mm")
    return _cuda


def _load_npu() -> GroupedSwiGLUFn:
    """Load the NPU expert projections with native activation rounding."""
    extension = require_package("torch_npu")
    require_symbol(extension, "npu_grouped_matmul")
    # This autograd implementation imports the optional NPU extension.
    from hyper_parallel.components.functional.grouped_matmul import (  # pylint: disable=import-outside-toplevel
        grouped_matmul,
    )

    def run(
        x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
        counts: torch.Tensor, scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply NPU gate, up and down projections with the reference routing semantics."""
        if x.shape[0] == 0:
            return _empty_output(x, w1, w2, w3, scores)
        offsets = counts.cumsum(0, dtype=torch.int64)
        # Separate projections preserve native's two rounded input-gradient
        # contributions. SiLU must also round before multiplying by up.
        gate = grouped_matmul(x, w1.transpose(1, 2), group_list=offsets)
        up = grouped_matmul(x, w3.transpose(1, 2), group_list=offsets)
        hidden = _weighted(F.silu(gate) * up, scores)
        return grouped_matmul(hidden, w2.transpose(1, 2), group_list=offsets)

    return run


def register_grouped_swiglu(registry: OpRegistry) -> None:
    """Register equivalent expert SwiGLU implementations on CPU, CUDA and NPU."""
    registry.register_family("grouped_swiglu", specs=(GROUPED_SWIGLU,))
    registry.register_impl(
        GROUPED_SWIGLU, name="native", devices=("cpu", "cuda", "npu"), load=lambda: native_grouped_swiglu,
    )
    registry.register_impl(GROUPED_SWIGLU, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        GROUPED_SWIGLU, name="torch_grouped_mm", devices=("cuda",), load=_load_cuda,
        check_compatibility=check_torch_grouped_mm,
    )
