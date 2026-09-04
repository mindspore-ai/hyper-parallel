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
"""VLM (vision-language model) dataset building facade.

Moved from ``components/data/vlm`` in stage 6 (05 §15.10 step 4); this
facade is the only supported import surface for VLM dataset building and
its symbol names and signatures are unchanged by the move. The modality
token-index constants live in ``hyper_parallel.data.vlm.constants``
(split in step 3); the top-level ``hyper_parallel.data`` package does not
re-export VLM symbols flat.
"""

from hyper_parallel.data.vlm.build_data_transform import build_vlm_data_transform
from hyper_parallel.data.vlm.build_processor import build_processor
from hyper_parallel.data.vlm.collator import build_vlm_collator
from hyper_parallel.data.vlm.dataset import build_vlm_dataset
from hyper_parallel.data.vlm.get_batch import (
    VLMBatchProcessor,
    VLMGetBatch,
    build_vlm_get_batch,
)

__all__ = [
    "VLMBatchProcessor",
    "VLMGetBatch",
    "build_processor",
    "build_vlm_collator",
    "build_vlm_data_transform",
    "build_vlm_dataset",
    "build_vlm_get_batch",
]
