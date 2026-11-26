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
"""Scope deterministic prefill exceptions to fully written CSA2 cache buffers."""

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

import torch

_fp8_allocation_lock = RLock()


@contextmanager
def allow_fully_written_cache_allocation() -> Iterator[None]:
    """Suspend prefill only while cache operators fully overwrite their outputs.

    The installed torch_npu cannot prefill FP8 outputs. With deterministic
    prefill enabled, packed UINT8 index_select/contiguous intermediates also
    incur redundant AI CPU Fill before the cache read overwrites every byte.
    The caller must restrict this scope to complete cache writes or reads;
    restore the process-wide setting even on error. Other training allocations
    retain their original deterministic behavior.
    """
    with _fp8_allocation_lock:
        previous = torch.utils.deterministic.fill_uninitialized_memory
        try:
            torch.utils.deterministic.fill_uninitialized_memory = False
            yield
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = previous
