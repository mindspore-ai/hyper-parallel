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
"""Exact integer row placement for returned Indexer selections."""

from typing import Protocol

import torch

from hyper_parallel.components.functional.indexer_topk import restore_indexer_topk, validate_indexer_topk
from .._dependency import require_package
from ..registry import OpRegistry
from ..spec import OpSpec


class IndexerTopKRestoreFn(Protocol):
    """Place disjoint local and received selections into an existing output."""

    def __call__(
        self, selected: torch.Tensor, received: torch.Tensor,
        rows: tuple[torch.Tensor, ...], output: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve all integer bits, slot order and negative sentinel values."""


INDEXER_TOPK_RESTORE = OpSpec[IndexerTopKRestoreFn](config_key="indexer_topk_restore")


@torch.no_grad()
def native_indexer_topk_restore(
    selected: torch.Tensor, received: torch.Tensor, rows: tuple[torch.Tensor, ...], output: torch.Tensor,
) -> torch.Tensor:
    """Place unique owner destinations using the original Torch operations."""
    validate_indexer_topk(selected, received, rows, output)
    output.index_copy_(0, rows[1], selected.index_select(0, rows[0]))
    output.index_copy_(0, rows[2], received)
    return output


def _load_triton() -> IndexerTopKRestoreFn:
    require_package("triton.backends.ascend")
    require_package("hyper_parallel.components.functional._triton.indexer_topk")
    return restore_indexer_topk


def _integer_input(selected: torch.Tensor) -> str | None:
    if selected.dtype not in (torch.int32, torch.int64):
        return "TopK restoration requires int32 or int64 selections"
    return None


def register_indexer_topk_restore(registry: OpRegistry) -> None:
    """Register exact Torch and Triton-Ascend integer restoration."""
    registry.register_family("indexer_topk_restore", specs=(INDEXER_TOPK_RESTORE,))
    registry.register_impl(
        INDEXER_TOPK_RESTORE, name="native", devices=("cpu", "cuda", "npu"),
        load=lambda: native_indexer_topk_restore, check_compatibility=_integer_input,
    )
    registry.register_impl(
        INDEXER_TOPK_RESTORE, name="triton_ascend", devices=("npu",), load=_load_triton,
        check_compatibility=_integer_input,
    )
