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
"""test_process_group.py"""

from hyper_parallel import (get_platform, init_process_group, destroy_process_group, get_process_group_ranks,
                            get_backend, split_group)

platform = get_platform()


def test_process_group():
    """
    Init process group with backend is ``hccl``, then try to get rank list and backend in this
    process group. After that, create a sub process group and get rank list and backend in sub process group.
    Finally, destroy the sub process group and process group.
    """
    # init process group
    init_process_group()
    world_size = platform.get_world_size()
    # pylint: disable=C0415
    rank_list = get_process_group_ranks()
    assert rank_list == list(range(world_size))
    backend = get_backend()
    assert backend == "hccl"

    # create process group
    group = platform.create_group(rank_list=list(range(world_size)))
    assert group is not None
    rank_list = get_process_group_ranks(group)
    assert rank_list == list(range(world_size))
    backend = get_backend(group)
    assert backend == "hccl"

    # split group
    split_ranks = []
    for rank in range(world_size):
        if rank % 2 == 0:
            split_ranks.append([rank, rank + 1])
    split_ranks.pop()
    split_group_from_group = split_group(group, split_ranks)
    rank_id = platform.get_rank()
    split_rank = [rank_id]
    if rank_id % 2 == 0:
        split_rank.append(rank_id + 1)
    else:
        split_rank.insert(0, rank_id - 1)

    if world_size - 1 in split_rank:
        assert not split_group_from_group
    else:
        assert split_group_from_group
        rank_list = get_process_group_ranks(split_group_from_group)
        assert rank_list == split_rank
        backend = get_backend(split_group_from_group)
        assert backend == "hccl"

        destroy_process_group(split_group_from_group)

    destroy_process_group(group)
    destroy_process_group()
