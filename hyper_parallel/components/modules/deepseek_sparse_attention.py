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
"""Compatibility exports for FlashMLA sparse-attention modules."""

from hyper_parallel.components.modules.flash_mla import (
    FlashMLASparseDecode as DeepseekV41SparseDecodeAttention,
    FlashMLASparsePrefill as DeepseekV41SparsePrefillAttention,
)

__all__ = ["DeepseekV41SparseDecodeAttention", "DeepseekV41SparsePrefillAttention"]
