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
from hyper_parallel.platform.mindspore.platform import MindSporePlatform
from mindspore import ops

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
    def ckpt_wrapper(module, checkpoint_fn=None, **checkpoint_fn_kwargs):
        raise NotImplementedError("ckpt_wrapper is not supported on MindSpore Graph platform")

    @property
    def noop_context_fn(self):
        raise NotImplementedError("noop_context_fn is not supported on MindSpore Graph platform")

    @staticmethod
    def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
        raise NotImplementedError("create_selective_checkpoint_contexts is not supported on MindSpore Graph platform")

    @staticmethod
    def async_save_on_cpu(policy_fn=None):
        raise NotImplementedError("async_save_on_cpu is not supported on MindSpore Graph platform")
