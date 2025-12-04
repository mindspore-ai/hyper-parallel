# Copyright 2025 Huawei Technologies Co., Ltd
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
"""_group_manager"""

from mindspore.communication import get_rank, create_group

_REDISTRIBUTION_GROUP_CACHE = {}

def _get_comm_group(rank_list):
    """_get_comm_group"""
    map_key = hash(tuple(rank_list))
    if map_key not in _REDISTRIBUTION_GROUP_CACHE:
        hash_str_rank_list = '-'.join([str(rank) for rank in rank_list])
        group = f"{len(rank_list)}-{hash_str_rank_list}"
        create_group(group, list(rank_list))
        _REDISTRIBUTION_GROUP_CACHE[map_key] = group
    return _REDISTRIBUTION_GROUP_CACHE[map_key]

