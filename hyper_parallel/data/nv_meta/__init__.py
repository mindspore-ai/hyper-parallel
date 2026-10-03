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
"""Prepared WebDataset sources backed by NVIDIA ``.nv-meta`` metadata.

The reader owns metadata and bytes; Online views own ordering, sharding and
replay state. The built-in or a custom sample adapter converts records into the canonical
HP RawSample consumed by the existing Text/Omni lifecycle.
"""

from hyper_parallel.data.nv_meta.build_dataset import build_nv_meta_dataset
from hyper_parallel.data.nv_meta.provider import NvMetaSource
from hyper_parallel.data.nv_meta.sample_adapter import NvMetaSampleAdapter
from hyper_parallel.data.nv_meta.reader import (
    NvMetaDataset,
    NvMetaPartLocation,
    NvMetaSampleIndex,
)

__all__ = [
    "NvMetaDataset",
    "NvMetaPartLocation",
    "NvMetaSource",
    "NvMetaSampleAdapter",
    "NvMetaSampleIndex",
    "build_nv_meta_dataset",
]
