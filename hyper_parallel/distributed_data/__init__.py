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
from hyper_parallel.distributed_data.balance_logging import format_balance_stats, log_balance_stats
from hyper_parallel.distributed_data.balancing_algorithm import BalancingAlgorithm, LPTBalancingAlgorithm
from hyper_parallel.distributed_data.cost_model import BackboneFlopsConfig, CostModel, DefaultCostModel
from hyper_parallel.distributed_data.data_constructor import default_collate_fn, default_pack_fn
from hyper_parallel.distributed_data.dataset import DistributedDataset, build_distributed_dataset
from hyper_parallel.distributed_data.dataset_dataloader import DatasetDataLoader
from hyper_parallel.distributed_data.device_prefetch import DeviceBatchPrefetcher
from hyper_parallel.distributed_data.distributed_dataloader import DistributedDataLoader
from hyper_parallel.distributed_data.external_step import ExternalStepAdapter, ExternalStepSource
from hyper_parallel.distributed_data.indexed_text import collate_indexed_text_sequences, pack_indexed_text_samples
from hyper_parallel.distributed_data.packed_balancing import LocalBalancingDataLoader, build_local_balancing_dataloader
from hyper_parallel.distributed_data.schema import (
    BufferedSampleMetadata,
    DistributedPackingPlan,
    PackingBinPlan,
    PackingConstraints,
    SampleKey,
    SampleMetadata,
    WorkloadCost,
)

__all__ = [
    "BackboneFlopsConfig",
    "BalancingAlgorithm",
    "LPTBalancingAlgorithm",
    "BufferedSampleMetadata",
    "CostModel",
    "DefaultCostModel",
    "DatasetDataLoader",
    "DeviceBatchPrefetcher",
    "DistributedDataLoader",
    "DistributedDataset",
    "DistributedDatasetConfig",
    "DistributedPackingPlan",
    "ExternalStepAdapter",
    "ExternalStepSource",
    "LocalBalancingDataLoader",
    "PackingBinPlan",
    "PackingConstraints",
    "SampleKey",
    "SampleMetadata",
    "WorkloadCost",
    "build_distributed_dataloader",
    "build_distributed_dataset",
    "build_local_balancing_dataloader",
    "collate_indexed_text_sequences",
    "default_collate_fn",
    "default_pack_fn",
    "format_balance_stats",
    "log_balance_stats",
    "pack_indexed_text_samples",
]
