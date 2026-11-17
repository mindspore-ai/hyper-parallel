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
"""Expert-major grouped matrix multiplication adapters."""

from __future__ import annotations

from typing import Protocol

import torch

from ..registry import OpBuildContext, OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class GroupedMatmulFn(Protocol):
    """Multiply expert-major rows by weights using cumulative group boundaries."""

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Multiply expert-major rows without a bias.

        Args:
            x: Token rows of shape ``[T, K]``, already grouped by expert.
            weight: Expert matrices of shape ``[E, K, N]``.
            group_list: One cumulative end per expert. Ends are nondecreasing,
                repeated ends represent empty experts, and the final end is T.

        Returns:
            Token rows of shape ``[T, N]`` in their input order.
        """


GROUPED_MATMUL = OpSpec[GroupedMatmulFn](config_key="grouped_matmul")


def _native(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
    # One small matmul per expert keeps the reference path independent of
    # grouped kernels without gathering an entire expert weight per token.
    ends = group_list.tolist()
    starts = [0, *ends[:-1]]
    if len(ends) != weight.shape[0] or (ends[-1] if ends else 0) != x.shape[0]:
        raise ValueError("Grouped matmul boundaries must cover every token and contain one entry per expert")
    if any(start > end for start, end in zip(starts, ends)):
        raise ValueError("Grouped matmul boundaries must be nondecreasing")
    if not ends:
        return x @ weight.sum(dim=0)
    return torch.cat([x[start:end] @ expert for start, end, expert in zip(starts, ends, weight.unbind())])


def _load_native() -> GroupedMatmulFn:
    return _native


def _load_npu() -> GroupedMatmulFn:
    require_symbol(require_package("torch_npu"), "npu_grouped_matmul")
    # The shared autograd adapter imports the optional NPU package.
    from hyper_parallel.components.functional.grouped_matmul import (  # pylint: disable=import-outside-toplevel
        grouped_matmul,
    )

    def run(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        return grouped_matmul(
            x, weight, bias=None, group_list=group_list.to(torch.int64), group_type=0, group_list_type=0,
        )

    return run


def _load_cuda() -> GroupedMatmulFn:
    require_symbol(torch, "_grouped_mm")

    def run(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        return torch._grouped_mm(x, weight, offs=group_list.to(torch.int32))  # pylint: disable=protected-access

    return run


def check_torch_grouped_mm(context: OpBuildContext) -> str | None:
    """Check CUDA requirements shared by grouped matrix multiplication users."""
    if context.input_dtype != torch.bfloat16:
        return "torch_grouped_mm requires BF16 inputs"
    if torch.cuda.get_device_capability(context.target_device)[0] < 8:
        return "torch_grouped_mm requires compute capability 8.0 or later"
    return None


def register_grouped_matmul(registry: OpRegistry) -> None:
    """Register one grouped GEMM over local expert-major token rows."""
    registry.register_family("grouped_matmul", specs=(GROUPED_MATMUL,))
    registry.register_impl(GROUPED_MATMUL, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(GROUPED_MATMUL, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        GROUPED_MATMUL, name="torch_grouped_mm", devices=("cuda",), load=_load_cuda,
        check_compatibility=check_torch_grouped_mm,
    )
