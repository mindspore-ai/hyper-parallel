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
"""Optional direct construction of large NPU receive permutations."""
from functools import lru_cache
import importlib
import importlib.util
from typing import Callable, Optional

import torch


# Smaller maps do not amortize the optional kernel's launch cost. This is a
# receive-size guard, independent of the expert-parallel rank numbering.
_MIN_DIRECT_ROWS = 8192


@lru_cache(maxsize=1)
def _get_direct_builder() -> Optional[Callable]:
    """Discover the optional Ascend backend without importing it on CPU paths."""
    try:
        if importlib.util.find_spec("triton.backends.ascend") is None:
            return None
        # Triton-Ascend is optional; importing HyperParallel must not require it.
        backend = importlib.import_module(
            "hyper_parallel.core.expert_parallel.receive_permutation_triton"
        )
    except ImportError:
        return None
    return backend.generate_receive_indices


def try_generate_receive_indices(
    source_offsets: torch.Tensor,
    counts: torch.Tensor,
    total_received: Optional[int],
) -> Optional[torch.Tensor]:
    """Construct a receive permutation when the optional backend is applicable.

    Args:
        source_offsets: Contiguous integer source starts, in expert-major order.
        counts: Contiguous integer lengths for those same segments.
        total_received: Known receive-buffer row count, or None. The caller
            remains responsible for ensuring that this equals the count sum.

    Returns:
        Long permutation on the input device, or None to use the reference path.
        Missing optional dependencies fall back; kernel failures are surfaced.
    """
    if total_received is None or total_received < _MIN_DIRECT_ROWS or counts.device.type != "npu":
        return None
    if (counts.dtype not in (torch.int32, torch.int64)
            or source_offsets.dtype not in (torch.int32, torch.int64)
            or not counts.is_contiguous() or not source_offsets.is_contiguous()):
        return None
    builder = _get_direct_builder()
    if builder is None:
        return None
    return builder(source_offsets, counts, total_received)
