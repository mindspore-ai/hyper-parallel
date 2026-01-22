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
"""HSDP optimizer shared level"""
from enum import auto, Enum


class OptimizerLevel(Enum):
    """
        Optimizer level:
                - SHARD_OPT:
                  Splitting is performed on optimizer state.
                - SHARD_OPT_GRAD:
                  Splitting is performed on optimizer state, and gradients.
                - SHARD_OPT_GRAD_PARAM:
                  Splitting is performed on optimizer state, gradients and weights.
    """
    SHARD_OPT = auto()
    SHARD_OPT_GRAD = auto()
    SHARD_OPT_GRAD_PARAM = auto()

class GroupInfo:
    """
    GroupInfo
    """
    def __init__(self, group_name, group, rank_size):
        self.group_name = group_name
        self.group = group
        self.rank_size = rank_size


class HSDPConfigV2:
    """HSDPConfigV2 inspect by torch fully_shard"""

    def __init__(self, mesh, reshard_after_forward, shard_placement_fn, mp_policy, offload_policy, ignored_param,
                 reduce_dtype=None, comm_async=False, comm_fusion=False, bucket_size=-1):
        """
            HSDP config init method
            Args:
                shard_size: optimizer weight sharded size.
                threshold: minimum weight size to shard.
                requires_acc_grad: requires gradient accumulation.
                grad_scale: use grad_scale to scale grad.
                shard_level: optimizer shard level.
                use_eager_hook: use eager hook or graph hook to implement hsdp.
                reduce_dtype: set gradient reduce dtype.
                comm_async: use async communication op for grad reduction.
                comm_fusion: use communication op fusion to reduce the number of communication op.
                bucket_size: the size of comm fusion buffer.
        """
        self.mesh = mesh
        self.reshard_after_forward = reshard_after_forward
        self.shard_placement_fn = shard_placement_fn
        self.mp_policy = mp_policy
        self.offload_policy = offload_policy
        self.reduce_dtype = self.offload_policy.reduce_dtype
        # TODO: 下方属性待删除
        self.comm_async = False
        self.comm_fusion = False
        self.bucket_size = 9999
        self.grad_fusion = False

class ShardedState(Enum):
    """
    Parameter shard state
    """
    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()

class FSDPSchedulerState(Enum):
    """
        Scheduler state:
                - PRE_FORWARD:
                  already run hook before forward.
                - FORWARD:
                  already run hook after forward.
                - PRE_BACKWARD:
                  already run hook before backward.
                - PRE_BACKWARD:
                  already run hook after backward.
    """
    PRE_FORWARD = auto()
    FORWARD = auto()
    PRE_BACKWARD = auto()
    BACKWARD = auto()