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
import mindspore as ms
from mindspore import nn
from mindspore.common.api import _no_grad
from mindspore import ops, Parameter
import mindspore.mint.distributed as dist
from mindspore.ops.function.comm_func import CommHandle
from hyper_parallel.core.fully_shard.utils import (
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
    FSDPMeshInfo,
    HSDPMeshInfo,
)
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2
from hyper_parallel.core.fully_shard.hsdp_utils import (
    ShardedState,
    FullyShardParamMode,
    unwrap_dtensor_param,
)
from hyper_parallel.core.dtensor.placement_types import Shard, StridedShard
from hyper_parallel.core.fully_shard.hsdp_utils import ParamModuleInfo
from hyper_parallel.platform.mindspore.fully_shard.pack_utils import (
    build_rs_plan,
    pack_for_reduce_scatter,
    unpack_from_all_gather,
)


def _pack_for_reduce_scatter(local_tensor: ms.Tensor, shard_dim: int, world_size: int) -> ms.Tensor:
    """Pack one local gradient into the row-major reduce-scatter layout.

    MindSpore currently aligns with the torch non-comm-fusion V1 path:

    - shard on dim 0: identity flatten
    - shard on non-dim0: chunk on shard dim, then concatenate on dim 0
    """
    if world_size <= 1 or shard_dim == 0:
        return local_tensor
    chunks = ms.mint.chunk(local_tensor, world_size, dim=shard_dim)
    return ms.mint.cat(chunks, dim=0).contiguous()


def _to_dtype_if_needed(
    tensor: ms.Tensor, dtype: Optional[ms.Type]
) -> ms.Tensor:
    """Cast tensor to the given dtype if it differs from current dtype."""
    if dtype is not None and tensor.dtype != dtype:
        return tensor.to(dtype)
    return tensor


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
        param_mode: Optional[FullyShardParamMode] = None,
        enable_fsdp_shard: bool = True,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.mp_policy = mp_policy
        self.device = device
        if param_mode is None:
            raise AssertionError("param_mode must be resolved before MindSporeHSDPParamV2 initialization.")
        self.param_mode = param_mode
        self.enable_fsdp_shard = enable_fsdp_shard
        self.offload_to_cpu: bool = isinstance(offload_policy, CPUOffloadPolicy)
        self.pin_memory = (
            self.offload_to_cpu and cast(CPUOffloadPolicy, offload_policy).pin_memory
        )
        self.grad_offload_event: Optional[ms.runtime.Event] = None
        dtensor_payload = unwrap_dtensor_param(param)
        self._orig_param_is_dtensor = dtensor_payload is not None
        self._orig_dtensor_mesh = dtensor_payload.device_mesh if dtensor_payload is not None else None
        self._orig_dtensor_placements = (
            tuple(dtensor_payload.placements) if dtensor_payload is not None else None
        )
        self._spmd_shard_mesh_dim = getattr(self.mesh_info, "shard_mesh_dim", None)
        self._spmd_replicate_mesh_dim = getattr(self.mesh_info, "replicate_mesh_dim", None)
        self._init_sharded_param(param, shard_placement_fn)
        self._init_group_infos()
        self.all_gather_outputs: List[ms.Tensor] = []
        self.unsharded_accumulated_grad = None
        self._unsharded_param: Optional[Parameter] = None
        self._param_fqn: Optional[str] = None
        # Communication attributes for prefetch pattern
        self.prefetch_handle: Optional[CommHandle] = None
        self._reduce_scatter_output = None
        self.reduce_scatter_handle: Optional[CommHandle] = None
        self._all_reduce_output = None
        self.all_reduce_handle: Optional[CommHandle] = None
        self._post_load_hook_handle = (
            module_info.module.register_load_state_dict_post_hook(
                lambda *args, **kwargs: self.reset_sharded_param()
            )
        )

    @property
    def uses_param_shard(self) -> bool:
        return self.enable_fsdp_shard

    @property
    def is_dtensor_compat_mode(self) -> bool:
        return self.param_mode == FullyShardParamMode.DTENSOR_COMPAT

    def _get_data_parallel_shard_placement(self, placements: list, shard_placement: Shard):
        """Return the explicit fully_shard placement on the unified SPMD mesh."""
        split_factor = 1
        shard_mesh_dim = getattr(self, "_spmd_shard_mesh_dim", None)
        for mesh_idx, placement in enumerate(placements):
            if mesh_idx == shard_mesh_dim:
                continue
            if placement.is_shard(shard_placement.dim):
                split_factor *= self._spmd_mesh.mesh_shape[mesh_idx]
        if split_factor > 1:
            return StridedShard(shard_placement.dim, split_factor=split_factor)
        return shard_placement

    @_no_grad()
    def _init_sharded_param(
        self,
        param: Parameter,
        shard_placement_fn: Optional[Callable],
    ) -> None:
        if param.device != "meta" and not (param.device.startswith("Ascend") and self.device == "npu"):
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
        base_placements = list(self._get_base_spmd_placements())
        self._spmd_placements = self._apply_data_parallel_placements(base_placements, hsdp_placement)
        param_data = unwrap_dtensor_param(param).to_local() if self._orig_param_is_dtensor else param

        shard_dim = hsdp_placement.dim
        self._orig_size = param_data.shape
        self._contiguous_orig_stride = make_contiguous_strides_for(self._orig_size)

        if self.uses_param_shard and isinstance(self.mesh_info, FSDPMeshInfo):  # FSDP or HSDP
            shard_rank = self.mesh_info.shard_mesh_rank
            shard_world_size = self.mesh_info.shard_mesh_size
        else:  # DDP
            shard_rank = 0
            shard_world_size = 1

        self.is_sharded = bool(self.uses_param_shard and shard_world_size > 1)

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
        set_requires_grad_if_needed(param, self.sharded_param)
        self.sharded_param.grad = None

        self._setattr_on_modules(self.sharded_param)
        self.sharded_param._hsdp_param_initialized = True
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

        This reconstructs the full parameter after all-gather by unpacking the
        gathered flat buffer back to the original tensor layout.
        """
        unsharded_param = self._get_unsharded_param_from_all_gather_output()
        if self._unsharded_param is not None:
            # Keep the Parameter identity stable across forward-reshard-backward
            # cycles so backward hooks continue to read gradients from the same
            # object that participated in the forward graph.
            if self._orig_param_is_dtensor:
                self._unsharded_param.set_data(unsharded_param)
            else:
                self._unsharded_param.data = unsharded_param
            set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
            self._unsharded_param.grad = None
            return
        if self._orig_param_is_dtensor:
            self._unsharded_param = Parameter(
                unsharded_param,
                name=self.sharded_param.name,
                requires_grad=self.sharded_param.requires_grad,
            )
            return
        # For MindSpore, if use `Parameter(tensor)`, Parameter will create a new Tensor instead of a view.
        # Here we need to share storage, so we use the `.data = tensor` approach to create shared storage.
        self._unsharded_param = Parameter(
            [],
            name=self.sharded_param.name,
            requires_grad=self.sharded_param.requires_grad,
        )
        self._unsharded_param.data = unsharded_param

    def _get_unsharded_param_from_all_gather_output(self):
        """Reconstruct the full local parameter view from the packed all-gather output."""
        if len(self.all_gather_outputs) != 1:
            raise AssertionError(
                f"Expected 1 all_gather_output, got {len(self.all_gather_outputs)}"
            )
        unsharded_tensor = self.all_gather_outputs[0]
        plan = build_rs_plan(
            self,
            self._sharded_local_tensor,
            self.shard_world_size if self.is_sharded else 1,
        )
        unsharded_param = unpack_from_all_gather(unsharded_tensor, plan)
        if getattr(self, "_orig_param_is_dtensor", False):
            unsharded_param = DTensor.from_local(
                unsharded_param,
                self._orig_dtensor_mesh,
                self._orig_dtensor_placements,
            )
        return unsharded_param

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.free_unsharded_param()
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        set_requires_grad_if_needed(self.sharded_param, self._unsharded_param)
        self._setattr_on_modules(self._unsharded_param)
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

    def _to_local_unsharded_grad(self, grad):
        """Normalize a pending gradient to the local tensor expected by fully_shard collectives."""
        return self._normalize_unsharded_grad_to_local(grad, reduce_partial_dtensor=False)

    def to_accumulated_grad_if_needed(self) -> None:
        if self._unsharded_param.grad is None:
            return
        unsharded_grad = self._unsharded_param.grad
        self._unsharded_param.grad = None
        if self.reduce_dtype is not None and unsharded_grad.dtype != self.reduce_dtype:
            unsharded_grad = unsharded_grad.to(self.reduce_dtype)
        if self.unsharded_accumulated_grad is None:
            self.unsharded_accumulated_grad = unsharded_grad
        else:
            self.unsharded_accumulated_grad += unsharded_grad

    def accumulate_unsharded_grad_if_needed(self) -> None:
        if (
            self.unsharded_accumulated_grad is not None
            and self.unsharded_param.grad is not None
        ):
            # need to handle the gradient
            self.unsharded_accumulated_grad += self._to_local_unsharded_grad(self.unsharded_param.grad)
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
        return self._to_local_unsharded_grad(grad)

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
            if isinstance(new_param, DTensor):
                self.sharded_param = new_param
                if not getattr(self.sharded_param, "_hsdp_param_initialized", None):
                    # reset _hsdp_param_initialized flag.
                    self.sharded_param._hsdp_param_initialized = True
            elif isinstance(new_param, ms.Tensor):
                # if new_param is Tensor, don't re-ref 'self.sharded_param'
                # just update self.sharded_param._local_tensor and self.sharded_param_data.
                pass

        local_tensor = new_param._local_tensor if isinstance(new_param, DTensor) else new_param
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
        if not same_local_tensor:
            if local_tensor.shape != sharded_size :
                raise AssertionError(
                    f"Expected sharded_size to be {sharded_size}, got {local_tensor.size()}"
                )
            updated_local_tensor = True
        if self.pin_memory and not local_tensor.is_pinned():
            local_tensor = local_tensor.to("cpu").pin_memory()
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
        # Optimizer steps may refresh the underlying local tensor storage. Re-sync
        # the cached flat shard view before reading all_gather_inputs for the next
        # unshard cycle.
        self.reset_sharded_param()
        all_gather_input = self.all_gather_inputs[0]

        # If parameter is not sharded (below threshold), no communication needed
        if not self.is_sharded:
            self.init_all_gather_outputs(
                all_gather_input_numels=[all_gather_input.numel()],
                all_gather_input_dtypes=[all_gather_input.dtype],
                world_size=1,
                device=all_gather_input.device.split(':')[0],
            )
            self.alloc_all_gather_outputs()
            self.all_gather_outputs[0].copy_(all_gather_input)
            return self.all_gather_outputs[0], None

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

    def reduce_scatter_output(self):
        """Return cached reduce-scatter output after waiting pending async work."""
        if self.reduce_scatter_handle is not None:
            self.reduce_scatter_handle.wait()
            self.reduce_scatter_handle = None
        return self._reduce_scatter_output

    def clear_reduce_scatter_output(self):
        """Clear cached reduce-scatter output."""
        self._reduce_scatter_output = None

    def reduce_scatter_grad(
        self,
        async_op: bool = True,
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
        shard_group_info = getattr(self, "sharded_group_info", None)
        shard_group = shard_group_info.group if shard_group_info is not None else None
        shard_group_size = shard_group_info.rank_size if shard_group_info is not None else 1
        if shard_group is None and isinstance(self.mesh_info, FSDPMeshInfo):
            shard_group = self.mesh_info.shard_process_group
            shard_group_size = self.shard_world_size
        plan_world_size = (
            shard_group_size
            if self.is_sharded and shard_group is not None and shard_group_size > 1
            else 1
        )
        plan = build_rs_plan(self, grad, plan_world_size)
        grad_flat = pack_for_reduce_scatter(grad, plan).reshape(-1)

        # If parameter is not sharded (below threshold), no reduce-scatter needed
        if not self.is_sharded:
            return grad_flat, None

        if shard_group is None or shard_group_size <= 1:
            # No communication needed
            return grad_flat, None

        # Calculate output size
        output_numel = grad_flat.numel() // shard_group_size
        self._reduce_scatter_output = ms.mint.empty(
            output_numel, dtype=reduce_dtype, device=grad.device.split(':')[0]
        )

        # Execute reduce_scatter_tensor
        self.reduce_scatter_handle = dist.reduce_scatter_tensor(
            self._reduce_scatter_output,
            grad_flat,
            op=reduce_op,
            group=shard_group,
            async_op=async_op,
        )

        return self._reduce_scatter_output, self.reduce_scatter_handle

    def zero_grad(self):
        self.sharded_param.grad = None

    def all_reduce_grad(
        self,
        grad: Optional[ms.Tensor] = None,
        dtype: Optional[ms.Type] = None,
        async_op: bool = True,
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
        else:
            grad = self._to_local_unsharded_grad(grad)

        if dtype is not None and dtype != grad.dtype:
            grad = grad.to(dtype)
        reduce_group_info = self.unsharded_group_info
        if reduce_group_info.rank_size <= 1:
            return grad, None
        reduce_group = reduce_group_info.group
        if reduce_group is None:
            raise RuntimeError("Expected a valid unsharded all-reduce group when rank_size > 1")

        self._all_reduce_output = grad
        self.all_reduce_handle = dist.all_reduce(
            grad,
            op=reduce_op,
            group=reduce_group,
            async_op=async_op
        )
        return self._all_reduce_output, self.all_reduce_handle

    def all_reduce_output(self):
        """Return cached all-reduce output after waiting pending async work."""
        if self.all_reduce_handle is not None:
            self.all_reduce_handle.wait()
            self.all_reduce_handle = None
        return self._all_reduce_output

    def clear_all_reduce_output(self):
        """Clear cached all-reduce output."""
        self._all_reduce_output = None

    def apply_reduced_grad(self, reduced_grad, param_type):
        """
        Apply reduced gradient to the sharded parameter.

        Reshapes ``reduced_grad`` to match the local shard, optionally
        offloads to CPU, then accumulates or assigns onto
        ``self.sharded_param.grad``.

        Args:
            reduced_grad (ms.Tensor): Gradient after reduce-scatter
                and/or all-reduce.
            param_type (Optional[ms.Type]): Target dtype for the gradient.
        """
        sharded_grad = self.sharded_param.grad
        reduced_grad = reduced_grad.view(self.sharded_size)
        reduced_grad = _to_dtype_if_needed(reduced_grad, param_type)
        to_accumulate_grad = sharded_grad is not None
        need_synchronize = False
        if self.offload_to_cpu:
            non_blocking = self.pin_memory and not to_accumulate_grad
            reduced_grad = reduced_grad.to(
                "cpu", non_blocking=non_blocking
            )
            need_synchronize = True
        if sharded_grad is None:
            self.sharded_param.grad = self.to_sharded_dtensor(reduced_grad)
        else:
            self.sharded_param.grad._local_tensor += reduced_grad

        if self.unsharded_accumulated_grad_data is not None:
            self.unsharded_accumulated_grad = None
        elif self.unsharded_param.grad is not None:
            self.unsharded_param.grad = None
        return need_synchronize


def set_requires_grad_if_needed(
    src_tensor: ms.Tensor, dst_tensor: ms.Tensor
) -> None:
    if src_tensor.requires_grad != dst_tensor.requires_grad:
        dst_tensor.requires_grad_(src_tensor.requires_grad)
