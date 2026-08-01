# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""fully_shard implementation on PyTorch."""

from hyper_parallel.platform.torch.fully_shard.extension import (
    FSDPGatherContext,
    FSDPLocalTensorExtension,
    fsdp_post_all_gather,
    fsdp_pre_all_gather,
    fsdp_shard_tensor,
    fsdp_to_dtensor,
)

__all__ = [
    "FSDPGatherContext",
    "FSDPLocalTensorExtension",
    "fsdp_post_all_gather",
    "fsdp_pre_all_gather",
    "fsdp_shard_tensor",
    "fsdp_to_dtensor",
]
