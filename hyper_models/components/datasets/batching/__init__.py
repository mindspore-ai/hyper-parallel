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
"""Public Dataset batching and DataLoader interfaces."""

from hyper_models.components.datasets.batching.attention_runtime import AttentionRuntimeAdapter
from hyper_models.components.datasets.batching.build_collate_fn import (
    DataCollator,
    MainCollator,
    TextPackingCollator,
    build_indexed_collate_fn,
    build_online_text_collate_fn,
)
from hyper_models.components.datasets.batching.build_dataloader import (
    DynamicBatchDataLoader,
    FixedBatchDataLoader,
    TextTokenBatcher,
    build_dataloader,
    calculate_num_micro_batches,
)
from hyper_models.components.datasets.batching.get_batch import ParallelBatch
from hyper_models.components.datasets.batching.sequence_boundaries import (
    IndexedBoundaryResolver,
    OnlineBoundaryResolver,
)
