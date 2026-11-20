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
"""Sliding-window KV packed-cache strategy."""

from torch import Tensor

from hyper_parallel.components.quantization.config import CSA2SWAKVFormat
from hyper_parallel.components.quantization.tensor.csa2_cache_tensor import pack_csa2_cache


class SWAKVQATStrategy:
    """Apply E4M3/E8M0 group32 storage before local SWA KV consumption."""

    def __init__(self, format_config: CSA2SWAKVFormat) -> None:
        """Retain the configured SWA KV format."""
        if not isinstance(format_config, CSA2SWAKVFormat):
            raise TypeError("SWAKVQATStrategy requires a CSA2SWAKVFormat.")
        self.format_config = format_config

    def pack(self, kv: Tensor) -> Tensor:
        """Hold local SWA KV in physical FP8 form until attention reads it."""
        return pack_csa2_cache(kv, "swa_kv", self.format_config.scale_format)

__all__ = ["SWAKVQATStrategy"]
