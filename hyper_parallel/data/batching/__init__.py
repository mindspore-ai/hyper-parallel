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
"""Batch collation, dynamic packing, and DataLoader construction."""

from hyper_parallel.data.batching.build_collate_fn import (
    build_default_collate_fn,
    build_indexed_collate_fn,
    build_omni_collate_fn,
    build_online_text_collate_fn,
)
from hyper_parallel.data.batching.build_dataloader import (
    FixedBatchDataLoader,
    OmniPackingLoader,
    TokenBatchLoader,
    build_dataloader,
    calculate_num_micro_batches,
)
from hyper_parallel.data.batching.get_batch import OmniParallelBatch, TextParallelBatch
from hyper_parallel.data.batching.packing import (
    FirstFitPackingSelector,
    PackingSelector,
    SamplePacker,
)
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)

__all__ = [
    "FirstFitPackingSelector",
    "FixedBatchDataLoader",
    "OmniParallelBatch",
    "OmniPackingLoader",
    "PackingSelector",
    "TextParallelBatch",
    "RuntimeInputAdapter",
    "RuntimeInputContext",
    "SamplePacker",
    "TokenBatchLoader",
    "build_default_collate_fn",
    "build_dataloader",
    "build_indexed_collate_fn",
    "build_omni_collate_fn",
    "build_online_text_collate_fn",
    "calculate_num_micro_batches",
]
