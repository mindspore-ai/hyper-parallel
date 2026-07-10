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
"""Megatron ``.bin`` / ``.idx`` dataset support.

Public API:

- :class:`IndexedDataset` + :class:`IndexedDatasetBuilder` — wire-format
  compatible reader/writer for Megatron-LM's ``.bin`` / ``.idx``.
- :class:`GPTDataset` — fixed-length sampling on top of an IndexedDataset.
- :class:`BlendableDataset` — weighted multi-source blend.

Importing this package registers ``data.type='megatron'`` with
:data:`hyper_parallel.data.DATASET_REGISTRY` via ``builder``.
"""
from hyper_parallel.data.megatron.blendable_dataset import BlendableDataset
from hyper_parallel.data.megatron.gpt_dataset import GPTDataset
from hyper_parallel.data.megatron.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
    strip_suffix,
)

from hyper_parallel.data.megatron import builder  # noqa: F401  # registers builder


__all__ = [
    "BlendableDataset",
    "GPTDataset",
    "IndexedDataset",
    "IndexedDatasetBuilder",
    "get_bin_path",
    "get_idx_path",
    "strip_suffix",
]
