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
"""Dataset Reader, Planner, and Data Constructor distributed data pipeline."""

from hyper_parallel.distributed_data.api import DistributedDatasetConfig, build_distributed_dataloader
from hyper_parallel.distributed_data.data_constructor import default_collate_fn, default_pack_fn
from hyper_parallel.distributed_data.indexed_text import collate_indexed_text_sequences, pack_indexed_text_samples
from hyper_parallel.distributed_data.device_prefetch import DeviceBatchPrefetcher
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DataConstructorPlan,
    DistributedPackingPlan,
    PackingBinPlan,
    PlannedSample,
    SampleKey,
    SampleMetadata,
    WorkloadCost,
)

__all__ = [
    "BufferedSampleMetadata",
    "DataConstructorPlan",
    "DeviceBatchPrefetcher",
    "DistributedDataLoader",
    "DistributedDatasetConfig",
    "DistributedPackingPlan",
    "PackingBinPlan",
    "PlannedSample",
    "SampleKey",
    "SampleMetadata",
    "WorkloadCost",
    "build_distributed_dataloader",
    "collate_indexed_text_sequences",
    "default_collate_fn",
    "default_pack_fn",
    "pack_indexed_text_samples",
]
