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
"""HSDP parameter"""


class HSDPParamV2:
    """
    HSDP parameter.
    """

    def __init__(
        self,
        param,
        module_info,
        mesh_info,
        post_forward_mesh_info,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        threshold,
    ):
        raise NotImplementedError("HSDP param subclasses must implement __init__")

    def _init_sharded_param(self, param, shard_placement_fn):
        """add and init sharded param"""
        raise NotImplementedError("HSDP param subclasses must implement _init_sharded_param")

    def _init_sharded_post_forward_param_metadata(self, param):
        raise NotImplementedError("HSDP param subclasses must implement _init_sharded_post_forward_param_metadata")

    def init_dtype_attrs(self, mp_policy):
        raise NotImplementedError("HSDP param subclasses must implement init_dtype_attrs")

    def _init_extensions(self):
        raise NotImplementedError("HSDP param subclasses must implement _init_extensions")

    def init_all_gather_outputs(self, all_gather_input_numels, all_gather_input_dtypes, world_size, device, force_recreate=False):
        raise NotImplementedError("HSDP param subclasses must implement init_all_gather_outputs")

    def init_unsharded_param(self):
        raise NotImplementedError("HSDP param subclasses must implement init_unsharded_param")

    def to_sharded(self):
        raise NotImplementedError("HSDP param subclasses must implement to_sharded")

    def to_sharded_post_forward(self):
        raise NotImplementedError("HSDP param subclasses must implement to_sharded_post_forward")

    def to_unsharded(self):
        raise NotImplementedError("HSDP param subclasses must implement to_unsharded")

    def to_sharded_dtensor(self, tensor):
        raise NotImplementedError("HSDP param subclasses must implement to_sharded_dtensor")

    def to_sharded_post_forward_dtensor(self, tensor):
        raise NotImplementedError("HSDP param subclasses must implement to_sharded_post_forward_dtensor")

    def to_accumulated_grad_if_needed(self):
        raise NotImplementedError("HSDP param subclasses must implement to_accumulated_grad_if_needed")

    def accumulate_unsharded_grad_if_needed(self):
        raise NotImplementedError("HSDP param subclasses must implement accumulate_unsharded_grad_if_needed")

    def alloc_all_gather_outputs(self):
        raise NotImplementedError("HSDP param subclasses must implement alloc_all_gather_outputs")

    def free_unsharded_param(self):
        raise NotImplementedError("HSDP param subclasses must implement free_unsharded_param")

    @property
    def all_gather_inputs(self):
        raise NotImplementedError("HSDP param subclasses must implement all_gather_inputs")

    @property
    def unsharded_param(self):
        raise NotImplementedError("HSDP param subclasses must implement unsharded_param")

    @property
    def unsharded_grad_data(self):
        raise NotImplementedError("HSDP param subclasses must implement unsharded_grad_data")

    @property
    def unsharded_accumulated_grad_data(self):
        raise NotImplementedError("HSDP param subclasses must implement unsharded_accumulated_grad_data")

    @property
    def _sharded_local_tensor(self):
        raise NotImplementedError("HSDP param subclasses must implement _sharded_local_tensor")

    def _get_unsharded_param_data(self, async_op=False):
        raise NotImplementedError("HSDP param subclasses must implement _get_unsharded_param_data")
    
    def unshard(self, async_op=False):
        raise NotImplementedError("HSDP param subclasses must implement unshard")

    def wait_for_unshard(self):
        raise NotImplementedError("HSDP param subclasses must implement wait_for_unshard")
    
    def shard(self):
        raise NotImplementedError("HSDP param subclasses must implement shard")
    
    def reduce_scatter_grad(self):
        raise NotImplementedError("HSDP param subclasses must implement reduce_scatter_grad")
    
    def all_reduce_grad(self):
        raise NotImplementedError("HSDP param subclasses must implement all_reduce_grad")
