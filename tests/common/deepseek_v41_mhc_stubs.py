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
"""Shape-preserving mHC stubs for CPU tests of unrelated V4.1 behavior."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch


def _coefficients(_module: object, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the first stream; these placeholder coefficients do not model mHC."""
    pre = torch.zeros_like(hidden[..., 0])
    pre[..., 0] = 1
    return pre, torch.zeros_like(pre), pre.new_zeros((*pre.shape, pre.shape[-1]))


def _collapse(hidden: torch.Tensor, _pre: torch.Tensor) -> torch.Tensor:
    """Forward the first stream for non-mHC model tests."""
    return hidden[..., 0, :]


def _expand(output: torch.Tensor, residual: torch.Tensor, _post: torch.Tensor,
            _mix: torch.Tensor) -> torch.Tensor:
    """Preserve shape and a gradient path without computing mHC."""
    return residual + output.unsqueeze(-2)


def _advance(module: Any, previous_output: torch.Tensor, residual: torch.Tensor,
             previous_pre: torch.Tensor, previous_post: torch.Tensor,
             previous_mix: torch.Tensor, norm_weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Preserve the shifted layer protocol without executing an mHC kernel."""
    updated = _expand(previous_output, residual, previous_post, previous_mix)
    pre, post, mix = _coefficients(module, updated)
    block_input = _collapse(updated, previous_pre).float()
    block_input = block_input * torch.rsqrt(block_input.square().mean(dim=-1, keepdim=True) + module.norm_eps)
    return updated, pre, post, mix, block_input.to(updated.dtype) * norm_weight


@contextmanager
def stub_mhc_for_cpu() -> Iterator[None]:
    """Isolate mHC while testing V4.1 routing, DSpark, or model wiring on CPU."""
    model = "hyper_parallel.models.deepseek_v41.modeling_deepseek_v41"
    dspark = "hyper_parallel.models.deepseek_v41.dspark"
    module = "hyper_parallel.components.modules.mhc"
    adapter = "hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc"
    with (patch(f"{model}.DeepseekV41PipelinedHyperConnection.forward", new=_coefficients),
          patch(f"{model}._hc_pre", new=_collapse),
          patch(f"{model}._hc_post", new=_expand),
          patch(f"{dspark}._DSparkHcCoefficients.forward", new=_coefficients),
          patch(f"{module}.pipelined_mhc_post", new=_expand),
          patch(f"{adapter}.DeepseekV41HyperMegaMhc.advance", new=_advance)):
        yield
