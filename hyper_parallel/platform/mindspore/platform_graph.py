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
"""MindSpore platform api for grap mode"""
from mindspore import ops

from hyper_parallel.platform.mindspore.platform import MindSporePlatform


class MindSporeGraphPlatform(MindSporePlatform):
    """MindSpore platform api"""

    @staticmethod
    def all_gather_into_tensor(data, group_info, async_op=False):
        output = ops.AllGather(group=group_info.group_name)(data)
        return output, None

    @staticmethod
    def all_reduce(data, group_info, async_op=False):
        output = ops.AllReduce(group=group_info.group_name)(data)
        return output, None

    @staticmethod
    def reduce_scatter_tensor(data, group_info, async_op=False):
        output = ops.ReduceScatter(group=group_info.group_name)(data)
        return output, None

    @staticmethod
    def swap_wrapper(module, policy_fn=None, group_swap=False):
        raise NotImplementedError("swap_wrapper is not supported on MindSpore Graph platform")

    @staticmethod
    def checkpoint_wrapper(module, **checkpoint_kwargs):
        raise NotImplementedError("checkpoint_wrapper is not supported on MindSpore Graph platform")

    @property
    def noop_context_fn(self):
        raise NotImplementedError("noop_context_fn is not supported on MindSpore Graph platform")

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False, group_swap=False):
        raise NotImplementedError("create_selective_checkpoint_contexts is not supported on MindSpore Graph platform")

    @staticmethod
    def ignore_sac_ops(ops: list[object | None]) -> None:
        raise NotImplementedError("ignore_sac_ops is not supported on MindSpore Graph platform")

    @staticmethod
    def async_save_on_cpu(policy_fn=None, group_swap: bool = False):
        raise NotImplementedError("async_save_on_cpu is not supported on MindSpore Graph platform")

    @staticmethod
    def get_class_activation_wrapper():
        raise NotImplementedError("get_class_activation_wrapper is not supported on MindSpore Graph platform")
