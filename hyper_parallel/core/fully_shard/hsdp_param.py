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
        """
        Initialize HSDPParamV2.

        Args:
            param (nn.Parameter): The original parameter to shard.
            module_info (ParamModuleInfo): Ownership and shared-weight metadata for the parameter.
            mesh_info (FSDPMeshInfo): Mesh topology describing shard/replicate dimensions.
            post_forward_mesh_info: Mesh info used after forward (reserved for subclass use).
            shard_placement_fn (Callable, optional): Returns a Shard placement for the parameter,
                or None to use default (Shard(0)).
            mp_policy (MixedPrecisionPolicy, optional): Mixed precision dtype policy.
            offload_policy (OffloadPolicy, optional): CPU offload policy.
            threshold: Minimum parameter size to enable sharding (reserved for subclass use).
        """
        raise NotImplementedError("HSDP param subclasses must implement __init__")

    def _init_sharded_param(self, param, shard_placement_fn):
        """add and init sharded param"""
        raise NotImplementedError("HSDP param subclasses must implement _init_sharded_param")

    def init_dtype_attrs(self, mp_policy):
        """Initialize dtype attributes from mixed precision policy."""
        raise NotImplementedError("HSDP param subclasses must implement init_dtype_attrs")

    def init_all_gather_outputs(
        self, all_gather_input_numels, all_gather_input_dtypes, world_size, device, force_recreate=False
    ):
        """Allocate or reuse output buffers for all-gather communication."""
        raise NotImplementedError("HSDP param subclasses must implement init_all_gather_outputs")

    def init_unsharded_param(self):
        """Reconstruct the full unsharded parameter from all-gather outputs."""
        raise NotImplementedError("HSDP param subclasses must implement init_unsharded_param")

    def to_sharded(self):
        """Transition parameter from unsharded back to sharded state and free unsharded storage."""
        raise NotImplementedError("HSDP param subclasses must implement to_sharded")

    def to_unsharded(self):
        """Transition parameter to unsharded state after all-gather completes."""
        raise NotImplementedError("HSDP param subclasses must implement to_unsharded")

    def to_sharded_dtensor(self, tensor):
        """Wrap a local sharded tensor as a DTensor with the correct mesh and placements."""
        raise NotImplementedError("HSDP param subclasses must implement to_sharded_dtensor")

    def to_accumulated_grad_if_needed(self):
        """Move unsharded grad to accumulated grad buffer if dtype conversion is required."""
        raise NotImplementedError("HSDP param subclasses must implement to_accumulated_grad_if_needed")

    def accumulate_unsharded_grad_if_needed(self):
        """Accumulate unsharded param grad into accumulated grad buffer if both exist."""
        raise NotImplementedError("HSDP param subclasses must implement accumulate_unsharded_grad_if_needed")

    def alloc_all_gather_outputs(self):
        """Resize all-gather output buffers to their full capacity for communication."""
        raise NotImplementedError("HSDP param subclasses must implement alloc_all_gather_outputs")

    def free_unsharded_param(self):
        """Release storage of all-gather outputs and inner tensors to free device memory."""
        raise NotImplementedError("HSDP param subclasses must implement free_unsharded_param")

    @property
    def all_gather_inputs(self):
        """Return the local sharded tensor(s) to use as input for all-gather communication."""
        raise NotImplementedError("HSDP param subclasses must implement all_gather_inputs")

    @property
    def unsharded_param(self):
        """Return the full unsharded parameter after all-gather."""
        raise NotImplementedError("HSDP param subclasses must implement unsharded_param")

    @property
    def unsharded_grad_data(self):
        """Return the unsharded_param.grad."""
        raise NotImplementedError("HSDP param subclasses must implement unsharded_grad_data")

    @property
    def unsharded_accumulated_grad_data(self):
        """Return the unsharded accumulated gradient buffer."""
        raise NotImplementedError("HSDP param subclasses must implement unsharded_accumulated_grad_data")

    @property
    def _sharded_local_tensor(self):
        """Return the underlying local tensor of the sharded DTensor parameter."""
        raise NotImplementedError("HSDP param subclasses must implement _sharded_local_tensor")

    def _get_unsharded_param_data(self, async_op=False):
        """Perform all-gather to obtain unsharded parameter data, returning (tensor, handle)."""
        raise NotImplementedError("HSDP param subclasses must implement _get_unsharded_param_data")

    def unshard(self, async_op=False):
        """Trigger all-gather to unshard the parameter, optionally asynchronously."""
        raise NotImplementedError("HSDP param subclasses must implement unshard")

    def wait_for_unshard(self):
        """Wait for all-gather to complete and transition parameter to unsharded state."""
        raise NotImplementedError("HSDP param subclasses must implement wait_for_unshard")

    def shard(self):
        """Transition parameter from unsharded back to sharded state."""
        raise NotImplementedError("HSDP param subclasses must implement shard")

    def reduce_scatter_grad(self):
        """Perform reduce-scatter on the unsharded gradient to produce a sharded gradient."""
        raise NotImplementedError("HSDP param subclasses must implement reduce_scatter_grad")

    def all_reduce_grad(self):
        """Perform all-reduce on gradient across the replicate dimension (HSDP mode only)."""
        raise NotImplementedError("HSDP param subclasses must implement all_reduce_grad")
