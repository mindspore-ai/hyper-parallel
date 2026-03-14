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
"""HSDP parameter"""
from typing import List, Callable, Optional, cast, Tuple
import itertools
# import torch
import mindspore as ms
from mindspore import nn
from mindspore.common.api import _no_grad
from mindspore import ops, Parameter, mint
import mindspore.mint.distributed as dist
from mindspore.communication.comm_func import CommHandle
from mindspore.ops.auto_generate.gen_ops_def import as_strided
from hyper_parallel.core.fully_shard.utils import (
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
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo


def make_contiguous_strides_for(shape, row_major=True):
    """
    Compute strides for a contiguous tensor of the given shape.

    Args:
        shape (tuple of int): The shape of the tensor. Each dimension must be a non-negative integer.
        row_major (bool): 
            - If True (default), returns C-style (row-major) strides: last dimension changes fastest.
            - If False, returns strides where the last two dimensions are Fortran-style 
              (i.e., for batched matrix operations in BLAS/LAPACK): second-to-last dim changes fastest.

    Returns:
        tuple of int: The computed strides.

    Examples:
        >>> make_contiguous_strides_for((2, 3, 4))
        (12, 4, 1)
        >>> make_contiguous_strides_for((2, 3, 4), row_major=False)
        (12, 1, 3)
        >>> make_contiguous_strides_for((5,))
        (1,)
        >>> make_contiguous_strides_for((5,), row_major=False)
        (1,)
        >>> make_contiguous_strides_for(())
        ()
    """
    if not isinstance(shape, (tuple, list)):
        raise TypeError("shape must be a tuple or list of non-negative integers")

    # Validate shape elements
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("All dimensions in shape must be non-negative integers")

    if not shape:
        return ()

    # Compute C-style (row-major) strides: stride[i] = product(shape[i+1:])
    strides = []
    multiplier = 1
    # Traverse shape in reverse order
    for size in reversed(shape):
        strides.append(multiplier)
        multiplier *= max(size, 1)  # handle size=0 gracefully (treat as 1 for stride calc)

    # Reverse to get correct order
    c_strides = tuple(reversed(strides))

    if row_major:
        return c_strides
    # For column-major: only affect last two dimensions
    if len(shape) < 2:
        return c_strides
    # In Fortran-style for matrices:
    #   stride of last dim = 1
    #   stride of second-to-last dim = shape[-1]
    # But note: in batched case (..., M, N), we want strides (..., N, 1) → wait!
    # However, the original PyTorch logic returns: result[:-2] + (1, max(shape[-2], 1))
    # Let's follow that exactly:
    # Example: shape=(B, M, N) → c_strides=(M*N, N, 1)
    #           col-major → (M*N, 1, M)
    # So: keep all but last two, then (1, shape[-2])
    return c_strides[:-2] + (1, max(shape[-2], 1))


class MindSporeHSDPParamV2(HSDPParamV2):
    """
    MindSpore HSDP parameter.
    """

    def __init__(
        self,
        param: Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        shard_placement_fn: Optional[Callable[[Parameter], Optional[Shard]]] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
        offload_policy: Optional[OffloadPolicy] = None,
        device: Optional[str] = None,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.mp_policy = mp_policy
        self.device = device
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self.grad_offload_event: Optional[ms.runtime.Event] = None
        self._init_sharded_param(param, shard_placement_fn)
        self.all_gather_outputs: List[ms.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._param_fqn: Optional[str] = None
        # Communication attributes for prefetch pattern
        self.prefetch_handle: Optional[CommHandle] = None
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )

    @_no_grad()
    def _init_sharded_param(
        self,
        param: Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> None:
        if not (param.device.startswith("Ascend") and self.device == "npu"):
            # if param.device != self.device and param.device != "meta":
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
        self._orig_size = param_data.shape
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)

        if isinstance(self.mesh_info, FSDPMeshInfo):  # FSDP or HSDP
            shard_rank = self.mesh_info.shard_mesh_rank
            shard_world_size = self.mesh_info.shard_mesh_size
        else:  # DDP
            shard_rank = 0
            shard_world_size = 1

        self.is_sharded = True

        if param_data.shape[shard_dim] % shard_world_size != 0:
            raise NotImplementedError(
                f"Uneven sharding on dim {shard_dim} not supported: "
                f"shape={param_data.shape}, world_size={shard_world_size}"
            )
        chunks = ms.mint.chunk(param_data, shard_world_size, dim=shard_dim)
        sharded_param = chunks[shard_rank].clone().contiguous()
        self.sharded_size = sharded_param.shape
        self.contiguous_sharded_stride = make_contiguous_strides_for(self.sharded_size)
        self._sharded_param_data = sharded_param.view(-1)

        self._sharding_spec = Layout.from_device_mesh(self._spmd_mesh)
        self._sharding_spec.set_placements(self._spmd_placements)
        self._sharding_spec.placement_to_tensor_map(param.ndim)

        shard_dtensor = DTensor.from_local(sharded_param, self._spmd_mesh, self._spmd_placements)
        self.sharded_param = Parameter(shard_dtensor, name=param.name)
        self.sharded_param.requires_grad_(param.requires_grad)
        self.sharded_param.grad = None

        # Directly creating a Parameter from a DTensor shares the underlying tensor; changes to the local_tensor affect the Parameter.
        # _param should share storage with sharded_param, while sharded_param should remain resident in memory.
        # We do not want changes to _param's storage reference to affect sharded_param.
        # Therefore we create a temporary dtensor (tmp_dtensor) and set its data instead of directly assigning Parameter(shard_tensor, name=param.name).
        tmp_dtensor = DTensor.from_local(mint.empty_like(sharded_param), self._spmd_mesh, self._spmd_placements)
        self._param = Parameter(tmp_dtensor, name=param.name)
        self._param.set_data(self.sharded_param.to_local())
        self._param.requires_grad_(param.requires_grad)
        self._dtensorparam_class = self._param.__class__
        self._setattr_on_modules(self._param)

        # register hook
        self._add_grad_to_unsharded_param(self._param)
        self.sharded_param._hsdp_param_initialized = True
        self._param._hsdp_param_initialized = True
        self.sharded_state = ShardedState.SHARDED
        self.param_dtype = None

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.sharded_param.dtype
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype

    def init_all_gather_outputs(
        self,
        all_gather_input_numels: list[int],
        all_gather_input_dtypes: list[ms.Type],
        world_size: int,
        device: str,
        force_recreate: bool = False,
    ):
        if not force_recreate and len(self.all_gather_outputs) > 0:
            return  # already initialized
        self.all_gather_outputs = [
            ms.mint.empty([numel * world_size], dtype=dtype, device=device.split(':')[0])
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
        unsharded_param = as_strided(
            unsharded_tensor,
            self._orig_size,
            self._contiguous_orig_stride,
            storage_offset=0,
        )

        # Create a placeholder parameter to record gradients and enable Torch backend code reuse.
        # This parameter does not participate in actual computations.
        # The actual computational parameters are handled separately via '_param'.
        self._unsharded_param = Parameter([])
        self._unsharded_param.data = unsharded_param
        self._unsharded_param.grad = None

    def _add_grad_to_unsharded_param(self, param):
        def hook(grad):
            self._unsharded_param.grad = grad
            return self._unsharded_param.grad

        param.register_hook(hook)

    def _update_param_data(self, tensor, param_class, has_init=None):
        """
        Update parameter data with encapsulated operations.

        Args:
            tensor: The tensor data to update.
            param_class: The parameter class to switch to.
            has_init: Optional has_init value to set.
        """
        self._param.__class__ = param_class
        self._param._update_data(tensor)
        self._param._local_tensor._update_data(tensor)
        if has_init is not None:
            self._param.has_init = has_init

    def to_sharded(self) -> None:
        # Switch _param type to DTensor and update data to sharded data
        self._update_param_data(
            self.sharded_param.to_local(),
            self._dtensorparam_class
        )
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        # Switch _param type to Parameter, update data to unsharded data
        self._update_param_data(
            self._unsharded_param,
            Parameter,
            has_init=self._unsharded_param.has_init
        )
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: Parameter) -> None:
        if getattr(self._module_info.module.__setattr__, "__func__", None) is nn.Cell.__setattr__:
            # fast path
            self._module_info.module._params[self._module_info.param_name] = param
        else:
            # slow path
            setattr(self._module_info.module, self._module_info.param_name, param)

        # Iterate through all modules that share this parameter to prevent pointer desync.
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            if getattr(shared_module.__setattr__, "__func__", None) is nn.Cell.__setattr__:
                shared_module._params[shared_param_name] = param
            else:
                setattr(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: ms.Tensor) -> DTensor:
        """
        Converts a local tensor representing either the sharded parameter or
        sharded gradient to DTensor.
        """
        return DTensor.from_local(
            tensor,
            self._sharding_spec.mesh,
            self._sharding_spec.placements
        )

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
            self.all_gather_outputs
        ):
            storage = tensor.untyped_storage()
            if storage.size() != 0:
                storage.resize_(0)

    @property
    def all_gather_inputs(self) -> list[ms.Tensor]:
        self._assert_in_states(ShardedState.SHARDED)
        sharded_param_data = self._sharded_param_data
        if self.offload_to_cpu:
            sharded_param_data = sharded_param_data.to(
                self.device, non_blocking=True
            )
        if self.param_dtype is not None and self.param_dtype != sharded_param_data.dtype:
            return [sharded_param_data.to(self.param_dtype)]
        return [sharded_param_data]

    @property
    def unsharded_param(self) -> Parameter:
        """Return the full unsharded parameter after all-gather."""
        return self._unsharded_param

    @property
    def unsharded_grad_data(self) -> ms.Tensor:
        """
        Get the unsharded gradient data as a local tensor.
        """
        grad = self.unsharded_param.grad
        if grad is None:
            raise AssertionError("Expects unsharded_param.grad to not be None")
        if isinstance(grad, DTensor):
            raise AssertionError("Expected ms.Tensor, got DTensor")
        return grad

    @property
    def unsharded_accumulated_grad_data(self) -> ms.Tensor:
        """
        Get the unsharded accumulated gradient data as a local tensor.
        """
        grad = self.unsharded_accumulated_grad
        return grad

    @property
    def _sharded_local_tensor(self) -> ms.Tensor:
        """Return the underlying local tensor of the sharded DTensor parameter."""
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
        if isinstance(self._sharded_param_data, ms.Tensor):
            same_local_tensor = (
                # when sharding param with shape (1, ...) over 2 ranks
                # local_tensor on rank 1 can be size 0, data_ptr() can be 0
                self._sharded_param_data.untyped_storage().data_ptr() > 0
                and self._sharded_param_data.untyped_storage().data_ptr()
                == local_tensor.untyped_storage().data_ptr()
            )
        sharded_size = self.sharded_size
        shard_dim = self.hsdp_placement.dim
        length = local_tensor.shape[shard_dim] if local_tensor.numel() > 0 else 0
        if local_tensor.shape != sharded_size and not same_local_tensor:
            raise AssertionError(
                f"Expected sharded_size to be {sharded_size}, got {local_tensor.shape}"
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

    def _get_unsharded_param_data(self, async_op: bool = False) -> Tuple[ms.Tensor, Optional[CommHandle]]:
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
                device=self._sharded_param_data.device.split(':')[0],
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
            device=self._sharded_param_data.device.split(':')[0],
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
        self._assert_in_states(ShardedState.SHARDED)

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
        dtype: Optional[ms.Type] = None,
        reduce_op: Optional[ops.ReduceOp] = ops.ReduceOp.SUM
    ) -> Tuple[ms.Tensor, Optional[CommHandle]]:
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
        output = ms.mint.empty(output_numel, dtype=reduce_dtype, device=grad.device.split(':')[0])

        # Execute reduce_scatter_tensor
        handle = dist.reduce_scatter_tensor(
            output,
            grad_flat,
            op=reduce_op,
            group=shard_group,
            async_op=async_op,
        )

        return output, handle

    def zero_grad(self):
        self.sharded_param.grad = None

    def all_reduce_grad(
        self,
        grad: Optional[ms.Tensor] = None,
        async_op: bool = False,
        reduce_op: Optional[ops.ReduceOp] = ops.ReduceOp.SUM
    ) -> Tuple[ms.Tensor, Optional[CommHandle]]:
        """
        Perform all-reduce on gradient (across replicate dimension in HSDP mode).

        Args:
            grad: Gradient tensor to reduce. If None, will use unsharded_param.grad
                or unsharded_accumulated_grad based on use_accumulated_grad flag.
            async_op: Whether to execute asynchronously.
            reduce_op: Optional[ops.ReduceOp] = ops.ReduceOp.SUM.

        Returns:
            (reduced_grad, handle): Reduced gradient and communication handle.
        """
        # If grad is not provided, get from parameter
        if grad is None:
            if self.unsharded_accumulated_grad is not None:
                grad = self.unsharded_accumulated_grad_data
            else:
                grad = self.unsharded_grad_data

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
    src_tensor: ms.Tensor, dst_tensor: ms.Tensor
) -> None:
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
