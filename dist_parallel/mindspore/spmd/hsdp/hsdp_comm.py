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
"""HSDP comm"""
import mindspore.communication.comm_func as comm_func
from mindspore.communication import create_group
from mindspore import ops
use_ops_interface = False
EXISTING_GROUPS = set()

def all_gather_into_tensor(data, group, async_op=False):
    """all_gather_into_tensor"""
    if use_ops_interface:
        output = ops.AllGather(group=group)(data)
        return output, output
    return comm_func.all_gather_into_tensor(data, group=group, async_op=async_op)

def all_reduce(data, group, async_op=False):
    """all_reduce"""
    if use_ops_interface:
        output = ops.AllReduce(group=group)(data)
        return output, output
    return comm_func.all_reduce(data, group=group, async_op=async_op)

def reduce_scatter_tensor(data, group, async_op=False):
    """reduce_scatter_tensor"""
    if use_ops_interface:
        output = ops.ReduceScatter(group=group)(data)
        return output, output
    return comm_func.reduce_scatter_tensor(data, group=group, async_op=async_op)

def create_group_if_needed(group, rank_ids, option=None):
    """create_group"""
    if group in EXISTING_GROUPS:
        return

    create_group(group, rank_ids, option)
    EXISTING_GROUPS.add(group)
