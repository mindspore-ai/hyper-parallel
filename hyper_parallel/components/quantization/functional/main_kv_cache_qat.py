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
"""Producer-side Main KV packed-cache strategy."""

from torch import Tensor

from hyper_parallel.components.quantization.config import CSA2MainKVFormat
from hyper_parallel.components.quantization.tensor.csa2_cache_tensor import pack_csa2_cache


class MainKVCacheQATStrategy:
    """Quantize the Full owner's compressed Main KV once; consumers reuse it."""

    def __init__(self, format_config: CSA2MainKVFormat) -> None:
        """Retain the configured Main KV format."""
        if not isinstance(format_config, CSA2MainKVFormat):
            raise TypeError("MainKVCacheQATStrategy requires a CSA2MainKVFormat.")
        self.format_config = format_config

    def pack(self, compressed_kv: Tensor) -> Tensor:
        """Keep a physical FP4 cache while preserving STE gradients to the producer."""
        return pack_csa2_cache(compressed_kv, "main_kv", self.format_config.scale_format)

__all__ = ["MainKVCacheQATStrategy"]
