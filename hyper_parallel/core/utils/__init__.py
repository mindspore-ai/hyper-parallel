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
"""Utility subpackage for distributed tensor operations."""

__all__ = [
    "compute_local_shape_and_global_offset",
    "compute_local_shape_and_global_offset_by_ceil_chunk",
    "clip_grad_norm_",
    "EXISTING_COMM_GROUPS",
    "differentiable_all_gather_concat",
    "differentiable_all_to_all",
    "differentiable_all_to_all_single",
    "differentiable_all_to_all_single_async",
    "differentiable_all_reduce",
    "differentiable_reduce_scatter",
    "differentiable_variable_all_gather",
    "exchange_splits_via_all_to_all",
    "gather_counts_via_all_gather",
    "p2p_exchange",
    "wait_async_tensor",
    "get_device_handle",
    "get_group_local_rank",
]

from hyper_parallel.core.utils.shape_utils import (
    compute_local_shape_and_global_offset,
    compute_local_shape_and_global_offset_by_ceil_chunk,
)
from hyper_parallel.core.utils.clip_grad import clip_grad_norm_
from hyper_parallel.core.utils.communication import (
    EXISTING_COMM_GROUPS,
    differentiable_all_gather_concat,
    differentiable_all_to_all,
    differentiable_all_to_all_single,
    differentiable_all_to_all_single_async,
    differentiable_all_reduce,
    differentiable_reduce_scatter,
    differentiable_variable_all_gather,
    exchange_splits_via_all_to_all,
    gather_counts_via_all_gather,
    get_device_handle,
    get_group_local_rank,
    p2p_exchange,
    wait_async_tensor,
)
