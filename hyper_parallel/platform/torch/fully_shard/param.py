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

# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/fsdp/_fully_shard/_fsdp_param.py
# enhanced with fully_shard parameter management
# ============================================================================
"""HSDP parameter"""
from typing import List, Callable, Optional, cast, Sequence, Tuple, Any
from dataclasses import dataclass, field
import itertools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch._prims_common import make_contiguous_strides_for
from hyper_parallel.platform.torch.fully_shard.utils import (
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
    FSDPMeshInfo,
    DDPMeshInfo,
    HSDPMeshInfo,
)
from hyper_parallel.core.dtensor import DTensor
from hyper_parallel.core.layout import Layout
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import ShardedState
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo, ExtensionsData


class TorchHSDPParamV2(HSDPParamV2):
    """
    Torch HSDP parameter.
    """

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo] = None,
        shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        threshold: int = 0,
        device: Optional[torch.device] = None,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.mp_policy = mp_policy
        self.threshold = threshold
        self.device = device
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self.grad_offload_event: Optional[torch.Event] = None
        self._init_sharded_param(param, shard_placement_fn)
        if self.post_forward_mesh_info:
            self._init_sharded_post_forward_param_metadata(param)
        self._init_extensions()
        self.all_gather_outputs: List[torch.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._param_fqn: Optional[str] = None
        # Communication attributes for prefetch pattern
        self.prefetch_handle: Optional[dist.Work] = None
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )

    @torch.no_grad()
    def _init_sharded_param(
        self,
        param: nn.Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> None:
        if param.device != self.device and param.device.type != "meta":
            raise AssertionError(
                f"Expects the parameter to already be moved to device {self.device} but got {param.device}"
            )

        hsdp_placement = shard_placement_fn(param) if shard_placement_fn else None
        if hsdp_placement is None:
            hsdp_placement = Shard(0)
        elif hsdp_placement.dim < 0:
            # if dim is negative, add the number of dimensions of the parameter
            hsdp_placement = Shard(hsdp_placement.dim + param.ndim)

        if not isinstance(hsdp_placement, Shard):
            raise AssertionError(
                f"Expected Shard, got {type(hsdp_placement)}: {hsdp_placement}"
            )

        self.hsdp_placement = hsdp_placement
        shard_dim = hsdp_placement.dim

        # Non-DTensor parameters have no pre-defined SPMD semantics.
        # FSDP/DDP solely determines the mesh and placements.
        self._spmd_mesh = self.mesh_info.mesh
        if isinstance(self.mesh_info, HSDPMeshInfo):  # HSDP
            self._spmd_placements = (Replicate(), hsdp_placement)
        elif isinstance(self.mesh_info, FSDPMeshInfo):  # FSDP
            self._spmd_placements = (hsdp_placement,)
        elif isinstance(self.mesh_info, DDPMeshInfo):  # DDP
            self._spmd_placements = (Replicate(),)
        param_data = param

        shard_dim = hsdp_placement.dim
        self._orig_size = param_data.size()
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)

        if isinstance(self.mesh_info, FSDPMeshInfo):  # FSDP or HSDP
            shard_rank = self.mesh_info.shard_mesh_rank
            shard_world_size = self.mesh_info.shard_mesh_size
        else:  # DDP
            shard_rank = 0
            shard_world_size = 1

        # Check if parameter size is below threshold, if so skip sharding
        param_size = param_data.numel() * param_data.element_size()
        if self.threshold > 0 and param_size < self.threshold:
            # Parameter too small, do not shard
            self.is_sharded = False
            self.sharded_size = param_data.size()
            self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
            self._sharded_param_data = param_data.view(-1)

            self._sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
            # For unsharded params, use Replicate placement
            if isinstance(self.mesh_info, HSDPMeshInfo):
                self._spmd_placements = (Replicate(), Replicate())
            else:
                self._spmd_placements = (Replicate(),)
            self._sharding_spec.set_placements(self._spmd_placements)
            self._sharding_spec.placement_to_tensor_map(param.ndim)

            self.sharded_param = nn.Parameter(DTensor.from_local(param_data, self._spmd_mesh, self._spmd_placements))
            self.sharded_param.requires_grad_(param.requires_grad)
            self._setattr_on_modules(self.sharded_param)
            self.sharded_state = ShardedState.SHARDED
            return

        self.is_sharded = True

        if param_data.size(shard_dim) % shard_world_size != 0:
            raise NotImplementedError(
                f"Uneven sharding on dim {shard_dim} not supported: "
                f"shape={param_data.shape}, world_size={shard_world_size}"
            )
        chunks = torch.chunk(param_data, shard_world_size, dim=shard_dim)
        sharded_param = chunks[shard_rank].clone().contiguous()
        self.sharded_size = sharded_param.size()
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        if self.offload_to_cpu and not sharded_param.is_meta:
            sharded_param = sharded_param.cpu()
            if self.pin_memory:
                sharded_param = sharded_param.pin_memory()
        self._sharded_param_data = sharded_param.view(-1)

        self._sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
        self._sharding_spec.set_placements(self._spmd_placements)
        self._sharding_spec.placement_to_tensor_map(param.ndim)

        self.sharded_param = nn.Parameter(DTensor.from_local(sharded_param, self._spmd_mesh, self._spmd_placements))
        self.sharded_param.requires_grad_(param.requires_grad)
        self._setattr_on_modules(self.sharded_param)
        # 初始化后，self.sharded_param替换掉原先的param，后续梯度也需要注意要累加到这个Parameter的grad上
        self.sharded_param._hsdp_param_initialized = True
        self.sharded_state = ShardedState.SHARDED
        self.param_dtype = None

    def _init_sharded_post_forward_param_metadata(self, param: torch.Tensor) -> None:
        mesh_info = self.post_forward_mesh_info
        param_data = param._local_tensor if isinstance(param, DTensor) else param
        if isinstance(mesh_info, FSDPMeshInfo):
            chunks = torch.chunk(param_data, mesh_info.shard_mesh_size, dim=0)
            self.sharded_post_forward_size = chunks[mesh_info.shard_mesh_rank].size()
        else:  # DDP
            chunks = torch.chunk(param_data, 1, dim=0)
            self.sharded_post_forward_size = chunks[0].size()

        self.contiguous_sharded_post_forward_stride = make_contiguous_strides_for(
            self.sharded_post_forward_size
        )

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype

    def _init_extensions(self) -> None:
        inner_tensor = self._sharded_local_tensor
        has_fsdp_pre_all_gather = hasattr(inner_tensor, "fsdp_pre_all_gather")
        has_fsdp_post_all_gather = hasattr(inner_tensor, "fsdp_post_all_gather")
        if has_fsdp_pre_all_gather != has_fsdp_post_all_gather:
            raise AssertionError(
                "Both fsdp_pre_all_gather and fsdp_post_all_gather should be defined "
                f"if using all-gather extensions: {inner_tensor}"
            )
        if has_fsdp_pre_all_gather:
            self._extensions_data = ExtensionsData()
        self._unsharded_inner_tensors: list[torch.Tensor] = []

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[torch.dtype],
        world_size: int,
        device: torch.device,
        force_recreate: bool = False,
    ):
        if not force_recreate and len(self.all_gather_outputs) > 0:
            return  # already initialized
        self.all_gather_outputs = [
            torch.empty(torch.Size([numel * world_size]), dtype=dtype, device=device)
            for numel, dtype in zip(all_gather_input_numels, all_gather_input_dtypes)
        ]

    def init_unsharded_param(self):
        """
        Initialize unsharded parameter from all-gather outputs.

        This reconstructs the full parameter after all-gather by using
        the gathered data and reshaping it to the original size.
        """
        if hasattr(self, "_unsharded_param"):
            return

        # Get unsharded data from all-gather outputs
        if len(self.all_gather_outputs) != 1:
            raise AssertionError(
                f"Expected 1 all_gather_output, got {len(self.all_gather_outputs)}"
            )
        unsharded_tensor = self.all_gather_outputs[0]
        # Use reshape to safely handle both contiguous and non-contiguous memory layouts.
        # It acts as a zero-copy view if possible, otherwise it performs a copy.
        # unsharded_param = unsharded_tensor.reshape(self._orig_size)
        unsharded_param = torch.as_strided(
            unsharded_tensor,
            self._orig_size,
            self._contiguous_orig_stride,
            storage_offset=0,
        )

        self._unsharded_param = nn.Parameter(
            unsharded_param, requires_grad=self.sharded_param.requires_grad
        )

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED

    def to_sharded_post_forward(self) -> None:
        if self.sharded_state != ShardedState.UNSHARDED:
            raise AssertionError(f"Expected sharded_state to be UNSHARDED, got {self.sharded_state}")
        shard_world_size = self.post_forward_mesh_info.shard_mesh_size
        numel = self.all_gather_outputs[0].numel()
        if numel % shard_world_size != 0:
            raise AssertionError(
                f"All-gather output size ({numel}) must be divisible by the shard "
                f"world size ({shard_world_size}). Check padding/mesh alignment."
            )
        shard_rank = self.post_forward_mesh_info.shard_mesh_rank
        sharded_numel = numel // shard_world_size
        # clone to be able to free all-gather output
        self._sharded_post_forward_param_data = (
            self.all_gather_outputs[0].narrow(
                0, sharded_numel * shard_rank, sharded_numel
            )
        ).clone()
        # sharded_post_forward_tensor = self._sharded_post_forward_param_data.view(
        #     self.sharded_post_forward_size
        # )
        sharded_post_forward_tensor = torch.as_strided(
            self._sharded_post_forward_param_data,
            size=self.sharded_post_forward_size,
            stride=self.contiguous_sharded_post_forward_stride,
            storage_offset=0,
        )
        self._sharded_post_forward_param = nn.Parameter(
            self.to_sharded_post_forward_dtensor(sharded_post_forward_tensor)
        )
        self._setattr_on_modules(self._sharded_post_forward_param)
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self) -> None:
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
        if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            self._sharded_post_forward_param = None
            self._sharded_post_forward_param_data = None
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        if getattr(self._module_info.module.__setattr__, "__func__", None) is nn.Module.__setattr__:
            # fast path
            self._module_info.module._parameters[self._module_info.param_name] = param
        else:
            # slow path
            setattr(self._module_info.module, self._module_info.param_name, param)

        # Iterate through all modules that share this parameter to prevent pointer desync.
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            if getattr(shared_module.__setattr__, "__func__", None) is nn.Module.__setattr__:
                shared_module._parameters[shared_param_name] = param
            else:
                setattr(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        return DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements
        )

    def to_sharded_post_forward_dtensor(self, tensor: torch.Tensor) -> DTensor:
        """
        Converts a local tensor to DTensor with post-forward sharding layout.
        """
        post_forward_layout = Layout.from_device_mesh(self.post_forward_mesh_info.mesh)
        post_forward_layout.set_placements((Replicate(), Shard(0)))
        post_forward_layout.placement_to_tensor_map(tensor.ndim)
        return DTensor.from_local(tensor, post_forward_layout.mesh, post_forward_layout.placements)

    def to_accumulated_grad_if_needed(self) -> None:
        if (
            self._unsharded_param.grad is not None
            and self.reduce_dtype is not None
            and self._unsharded_param.grad.dtype != self.reduce_dtype
        ):
            # need to handle the gradient even after the parameter is resharded
            unsharded_grad = self._unsharded_param.grad
            self._unsharded_param.grad = None
            self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            # need to handle the gradient
            self.unsharded_accumulated_grad += self.unsharded_param.grad
            self.unsharded_param.grad = None

    def alloc_all_gather_outputs(self) -> None:
        for tensor in self.all_gather_outputs:
            expected_size = tensor.numel() * tensor.itemsize
            storage = tensor.untyped_storage()
            if storage.size() != expected_size:
                storage.resize_(expected_size)

    def free_unsharded_param(self) -> None:
        for tensor in itertools.chain(
            self.all_gather_outputs, self._unsharded_inner_tensors
        ):
            storage = tensor.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)

    @property
    def all_gather_inputs(self) -> list[torch.Tensor]:
        self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
        if self.sharded_state == ShardedState.SHARDED:
            sharded_param_data = self._sharded_param_data
            if self.offload_to_cpu:
                sharded_param_data = sharded_param_data.to(
                    self.device, non_blocking=True
                )
            if self.param_dtype is not None and self.param_dtype != sharded_param_data.dtype:
                return [sharded_param_data.to(self.param_dtype)]
            else:
                return [sharded_param_data]
        elif self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
            if self.param_dtype is not None and self.param_dtype != self._sharded_post_forward_param_data.dtype:
                return [self._sharded_post_forward_param_data.to(self.param_dtype)]
            else:
                return [self._sharded_post_forward_param_data]
        return [torch.empty(0)]

    @property
    def unsharded_param(self) -> nn.Parameter:  # ND
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        """
        Get the unsharded gradient data as a local tensor.
        """
        grad = self.unsharded_param.grad
        if grad is None:
            raise AssertionError("Expects unsharded_param.grad to not be None")
        if isinstance(grad, DTensor):
            raise AssertionError("Expected torch.Tensor, got DTensor")
        return grad

    @property
    def unsharded_accumulated_grad_data(self) -> torch.Tensor:
        """
        Get the unsharded accumulated gradient data as a local tensor.
        """
        grad = self.unsharded_accumulated_grad
        # if grad is None:
        #     raise AssertionError("Expects unsharded_accumulated_grad to not be None")
        # if isinstance(grad, DTensor):
        #     raise AssertionError("Expected torch.Tensor, got DTensor")
        return grad

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    @property
    def shard_world_size(self) -> int:
        """Get the world size for shard dimension."""
        if isinstance(self.mesh_info, FSDPMeshInfo):
            return self.mesh_info.shard_mesh_size
        return 1

    @property
    def replicate_world_size(self) -> int:
        """Get the world size for replicate dimension (HSDP only)."""
        if isinstance(self.mesh_info, HSDPMeshInfo):
            return self.mesh_info.replicate_mesh_size
        return 1

    def _assert_in_states(self, *states: ShardedState) -> None:
        """Assert current state is one of expected states."""
        if self.sharded_state not in states:
            raise AssertionError(
                f"Expected sharded_state in {states}, got {self.sharded_state}"
            )

    def reset_sharded_param(self) -> None:
        """Reset sharded param after load_state_dict."""
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        if new_param is not self.sharded_param:
            # Ensure object identity is preserved after parameter conversion.
            if torch.__future__.get_swap_module_params_on_conversion():
                raise AssertionError(
                    f"Expects swap_tensors to preserve object but got {new_param} "
                    f"instead of {self.sharded_param}"
                )
            self.sharded_param = new_param

        local_tensor = new_param._local_tensor
        if local_tensor.is_meta:
            return
        updated_local_tensor = False
        # local_tensor can be padded twice
        # 1st time in fully_shard(model)
        # 2nd time in model(input) lazy_init
        # 2nd time should be no-op if parameters remain unchanged
        # 2nd time shouldn't be no-op if people call model.load_state_dict(...) before lazy_init
        # this makes it possible for trainer to call `sd = model.state_dict()` before the training loop
        # and use `sd` without calling .state_dict() per iteration
        same_local_tensor = False
        # TODO: need to support tensor subclass
        if type(self._sharded_param_data) is torch.Tensor:
            same_local_tensor = (
                # when sharding param with shape (1, ...) over 2 ranks
                # local_tensor on rank 1 can be size 0, data_ptr() can be 0
                self._sharded_param_data.untyped_storage().data_ptr() > 0
                and self._sharded_param_data.untyped_storage().data_ptr()
                == local_tensor.untyped_storage().data_ptr()
            )
        sharded_size = self.sharded_size
        shard_dim = self.hsdp_placement.dim
        length = local_tensor.size(shard_dim) if local_tensor.numel() > 0 else 0
        if local_tensor.size() != sharded_size and not same_local_tensor:
            raise AssertionError(
                f"Expected sharded_size to be {sharded_size}, got {local_tensor.size()}"
            )
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.cpu().pin_memory()
            updated_local_tensor = True
        if not same_local_tensor:
            self._sharded_param_data = local_tensor.view(-1)
        if not isinstance(self.sharded_param, DTensor):
            raise AssertionError(f"Expected DTensor, got {type(self.sharded_param)}")
        if updated_local_tensor:
            # Only change the local tensor object if needed
            self.sharded_param._local_tensor = local_tensor.narrow(
                dim=shard_dim, start=0, length=length
            )
            if not self.sharded_param._local_tensor.is_contiguous():
                raise AssertionError(
                    "Expected sharded_param._local_tensor to be contiguous"
                )
        self._sharding_spec = cast(DTensor, self.sharded_param).layout

    def _get_unsharded_param_data(self, async_op: bool = False) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Perform all-gather to get unsharded parameter data.

        Args:
            async_op: Whether to execute asynchronously.

        Returns:
            (unsharded_param, handle): Unsharded parameter data and communication handle.
        """
        # If parameter is not sharded (below threshold), no communication needed
        if not self.is_sharded:
            self.init_all_gather_outputs(
                all_gather_input_numels=[self._sharded_param_data.numel()],
                all_gather_input_dtypes=[self._sharded_param_data.dtype],
                world_size=1,
                device=self.device,
            )
            self.alloc_all_gather_outputs()
            self.all_gather_outputs[0].copy_(self._sharded_param_data)
            return self.all_gather_outputs[0], None

        # Get input data
        all_gather_input = self.all_gather_inputs[0]

        # Initialize output buffer
        self.init_all_gather_outputs(
            all_gather_input_numels=[all_gather_input.numel()],
            all_gather_input_dtypes=[all_gather_input.dtype],
            world_size=self.shard_world_size,
            device=self.device,
        )
        self.alloc_all_gather_outputs()

        # Get communication group
        shard_group = self.mesh_info.shard_process_group if isinstance(self.mesh_info, FSDPMeshInfo) else None

        if shard_group is None or self.shard_world_size <= 1:
            # No communication needed, just copy
            self.all_gather_outputs[0].copy_(all_gather_input)
            return self.all_gather_outputs[0], None

        # Execute all_gather_into_tensor
        handle = dist.all_gather_into_tensor(
            self.all_gather_outputs[0],
            all_gather_input,
            group=shard_group,
            async_op=async_op,
        )

        return self.all_gather_outputs[0], handle

    def unshard(self, async_op: bool = False) -> None:
        if self.prefetch_handle is not None:
            # 已经被prefetch 触发过了，直接return
            return  # no-op

        _, handle = self._get_unsharded_param_data(async_op=async_op)
        self.prefetch_handle = handle

    def wait_for_unshard(self) -> None:
        self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)

        if self.prefetch_handle is not None:
            self.prefetch_handle.wait()
            self.prefetch_handle = None

        self.init_unsharded_param()
        self.to_unsharded()

    def shard(self) -> None:
        """
        Transition parameter from unsharded back to sharded state.
        """
        self._assert_in_states(ShardedState.UNSHARDED)
        self.to_sharded()

    def reduce_scatter_grad(
        self,
        async_op: bool = False,
        dtype: Optional[torch.dtype] = None,
        reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Perform reduce-scatter on gradient to reduce and shard the full gradient.

        Args:
            async_op: Whether to execute asynchronously.
            dtype: reduce dtype.
            reduce_op: do reduce-scatter avg or sum.

        Returns:
            (sharded_grad, handle): Sharded gradient and communication handle.
        """
        self._assert_in_states(ShardedState.UNSHARDED)

        # Choose gradient source based on use_accumulated_grad flag
        if self.unsharded_accumulated_grad is not None:
            grad = self.unsharded_accumulated_grad_data
        else:
            grad = self.unsharded_grad_data
        reduce_dtype = dtype or grad.dtype
        grad = grad.to(reduce_dtype)
        grad_flat = grad.view(-1)

        # If parameter is not sharded (below threshold), no reduce-scatter needed
        if not self.is_sharded:
            return grad_flat, None

        # Get communication group
        shard_group = self.mesh_info.shard_process_group if isinstance(self.mesh_info, FSDPMeshInfo) else None

        if shard_group is None or self.shard_world_size <= 1:
            # No communication needed
            return grad_flat, None

        # Calculate output size
        output_numel = grad_flat.numel() // self.shard_world_size
        output = torch.empty(output_numel, dtype=reduce_dtype, device=grad.device)

        # Execute reduce_scatter_tensor
        handle = dist.reduce_scatter_tensor(
            output,
            grad_flat,
            op=reduce_op,
            group=shard_group,
            async_op=async_op,
        )

        return output, handle

    def all_reduce_grad(
        self,
        grad: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        async_op: bool = False,
        reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG
    ) -> Tuple[torch.Tensor, Optional[dist.Work]]:
        """
        Perform all-reduce on gradient (across replicate dimension in HSDP mode).

        Args:
            grad: Gradient tensor to reduce. If None, will use unsharded_param.grad
                or unsharded_accumulated_grad based on use_accumulated_grad flag.
            async_op: Whether to execute asynchronously.
            reduce_op: Optional[dist.ReduceOp] = dist.ReduceOp.AVG.

        Returns:
            (reduced_grad, handle): Reduced gradient and communication handle.
        """
        # If grad is not provided, get from parameter
        if grad is None:
            if self.unsharded_accumulated_grad is not None:
                grad = self.unsharded_accumulated_grad_data
            else:
                grad = self.unsharded_grad_data

        if dtype is not None and dtype != grad.dtype:
            grad = grad.to(dtype)

        if not isinstance(self.mesh_info, HSDPMeshInfo):
            # Not HSDP mode, no all-reduce needed
            return grad, None

        replicate_group = self.mesh_info.replicate_process_group
        if replicate_group is None or self.replicate_world_size <= 1:
            return grad, None

        handle = dist.all_reduce(
            grad,
            op=reduce_op,
            group=replicate_group,
            async_op=async_op
        )
        return grad, handle


def set_requires_grad_if_needed(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor
) -> None:
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
