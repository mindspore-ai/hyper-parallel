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
"""Context-parallel helpers for the LlamaFactory integration."""
from .inputs import (
    _get_cp_dp_ranks,
    get_cp_group,
    get_cp_group_ranks,
    get_cp_rank,
    get_dp_rank,
    shard_inputs_for_cp,
)
from .loss import _build_cp_shift_labels, _enable_context_parallel_loss_patch
from .context_parallel_prepare import cp_prepare_model

__all__ = [
    "_enable_context_parallel_loss_patch",
    "_get_cp_dp_ranks",
    "_build_cp_shift_labels",
    "cp_prepare_model",
    "get_cp_group",
    "get_cp_group_ranks",
    "get_cp_rank",
    "get_dp_rank",
    "shard_inputs_for_cp",
]
