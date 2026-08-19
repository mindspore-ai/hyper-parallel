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
"""Shared LLM and Omni dataset pipeline stages."""

from hyper_models.components.datasets.batch import PreparedBatch
from hyper_models.components.datasets.build_collate_fn import (
    MakeMicroBatchCollator,
    build_collate_fn,
    calculate_num_micro_batches,
)
from hyper_models.components.datasets.build_dataloader import DataLoader, build_dataloader
from hyper_models.components.datasets.build_dataset import DummyDataset, build_dataset
from hyper_models.components.datasets.collator import FieldCollateSpec, ModelSampleCollator
from hyper_models.components.datasets.contracts import (
    BatchCollator,
    MicroBatch,
    ModelSample,
    RawSample,
    SampleTransform,
    TransformedSample,
    is_iterable_dataset,
)
from hyper_models.components.datasets.parallel import (
    DatasetParallelContext,
    build_dataset_batch_sampler,
    build_distributed_dataset,
    create_dataset_parallel_context,
)


__all__ = [
    "BatchCollator",
    "DataLoader",
    "DatasetParallelContext",
    "DummyDataset",
    "FieldCollateSpec",
    "MakeMicroBatchCollator",
    "MicroBatch",
    "ModelSample",
    "ModelSampleCollator",
    "PreparedBatch",
    "RawSample",
    "SampleTransform",
    "TransformedSample",
    "build_collate_fn",
    "build_dataloader",
    "build_dataset_batch_sampler",
    "build_dataset",
    "build_distributed_dataset",
    "calculate_num_micro_batches",
    "create_dataset_parallel_context",
    "is_iterable_dataset",
]
