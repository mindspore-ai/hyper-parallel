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
"""HyperParallel dataset module for the unified ``data.type`` registry.

Public API:

- :func:`build_dataset`: registry-driven dispatch on ``data.type``.
- :data:`DATASET_REGISTRY`: registry instance; register custom formats
  with ``@DATASET_REGISTRY.register("<name>")``.
- :class:`DummyDataset`, :class:`DummyVLDataset`, :class:`TokenizedDataset`,
  :class:`PresetPtDataset`: first-party dataset classes.
- :class:`IndexedDataset`, :class:`IndexedDatasetBuilder`,
  :class:`GPTDataset`, :class:`BlendableDataset`: Megatron support.

Importing this package registers every first-party builder
(dummy, vl_dummy, hf_datasets, json_file, preset_pt, megatron).
External plugins can register additional builders by importing their
modules before the trainer is constructed.
"""
from hyper_parallel.data.dummy import DummyDataset
from hyper_parallel.data.hf import TokenizedDataset
from hyper_parallel.data.megatron import (
    BlendableDataset,
    GPTDataset,
    IndexedDataset,
    IndexedDatasetBuilder,
)
from hyper_parallel.data.preset_pt import PresetPtDataset
from hyper_parallel.data.registry import DATASET_REGISTRY, build_dataset
from hyper_parallel.data.vl_dummy import DummyVLDataset

# Each module registers a builder when imported.
from hyper_parallel.data import (  # noqa: F401  # registers builders
    dummy,
    hf,
    preset_pt,
    vl_dummy,
)


__all__ = [
    "BlendableDataset",
    "DATASET_REGISTRY",
    "DummyDataset",
    "DummyVLDataset",
    "GPTDataset",
    "IndexedDataset",
    "IndexedDatasetBuilder",
    "PresetPtDataset",
    "TokenizedDataset",
    "build_dataset",
]
