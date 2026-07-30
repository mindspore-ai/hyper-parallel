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
"""Distributed helpers — stubs matching 06_distributed_infrastructure.md §7.

get_world_size_safe / get_rank_safe: safe wrappers around torch.distributed.
"""

import torch
import torch.distributed as dist


def get_world_size_safe() -> int:
    """Return dist.get_world_size() if initialized, else 1."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_global_rank_safe() -> int:
    """Return dist.get_rank() if initialized, else 0."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_local_rank_safe() -> int:
    """Return dist.get_node_local_rank() if initialized, else 0."""
    if dist.is_initialized():
        return dist.get_node_local_rank()
    return 0
