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
"""data.parallel: batch-level parallel sharding primitives.

Stage 4e contributed ``shard_batch_for_cp``; stage 6 (05 §11.2) adds the
DataLoader-side CP/TP batch distribution, DP samplers and parallel
dataloader construction from ``components/datasets/parallel``, plus the
``OnlineDatasetBarrier`` build synchronization primitive (05 §15.10
step 3).
"""

from hyper_parallel.data.parallel.batch_parallel import (
    CPBatchSharder,
    TPBatchBroadcaster,
    shard_batch_for_cp,
)
from hyper_parallel.data.parallel.batch_sampler import build_dataset_batch_sampler
from hyper_parallel.data.parallel.build_barrier import OnlineDatasetBarrier
from hyper_parallel.data.parallel.dataloader_parallel import (
    DataLoaderParallelContext,
    build_dataset_for_dataloader,
    create_dataloader_parallel_context,
    split_iterable_dataset_by_dp,
)

__all__ = [
    "CPBatchSharder",
    "DataLoaderParallelContext",
    "OnlineDatasetBarrier",
    "TPBatchBroadcaster",
    "build_dataset_batch_sampler",
    "build_dataset_for_dataloader",
    "create_dataloader_parallel_context",
    "shard_batch_for_cp",
    "split_iterable_dataset_by_dp",
]
