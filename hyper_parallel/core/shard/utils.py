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
"""Utility functions for Torch shard operations."""

from typing import Any, Optional

import os
import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch._ops import OpOverload, OpOverloadPacket

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import _get_slice_tensor_by_layout
from hyper_parallel.core.tensor_parallel.loss_parallel import _get_loss_parallel_strict
from hyper_parallel.core.tensor_parallel.loss_parallel_ops_common import (
    _check_context_and_layout,
    _get_local_tensor,
    _get_mesh_and_dim,
    _is_dtensor,
    _is_shard_on_last_dim,
    _validate_cross_entropy_params,
    _validate_mesh_and_shard,
)

def get_world_size() -> int:
    """Return Torch distributed world size, or WORLD_SIZE before init."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_cell_construct(cell):
    """Return the Torch module forward callable."""
    return cell.forward


def get_cells_and_names(cell):
    """Return Torch module names and instances."""
    return cell.named_modules()


def search_parameter_by_name(cell, param_name: str):
    """Find a parameter and its owning module by dotted name."""
    param_name = param_name.replace("self.", "")
    if param_name in cell._parameters:  # pylint: disable=protected-access
        return cell, param_name, cell._parameters[param_name]  # pylint: disable=protected-access

    if "." in param_name:
        cell_path, param_key = param_name.rsplit(".", 1)
        try:
            target_cell = cell.get_submodule(cell_path)
        except AttributeError:
            target_cell = None
        if target_cell is not None and param_key in target_cell._parameters:  # pylint: disable=protected-access
            return target_cell, param_key, target_cell._parameters[param_key]  # pylint: disable=protected-access

    for _, child_cell in cell.named_children():
        if isinstance(child_cell, nn.Module):
            result = search_parameter_by_name(child_cell, param_name)
            if result is not None:
                return result
    return None


def set_layout_into_parameter(param, layout):
    """Convert a local Torch parameter to a DTensor-backed parameter."""
    if isinstance(param, DTensor):
        raise ValueError(f"Parameter {param} has been configured layout, cannot be set repeatedly.")
    requires_grad = param.requires_grad
    param_dtensor = DTensor.from_local(
        _get_slice_tensor_by_layout(param, layout),
        layout.mesh,
        layout.alias_placements,
    )
    return Parameter(param_dtensor, requires_grad=requires_grad)


def update_parameter_by_name(result: tuple, new_param) -> bool:
    """Replace a parameter on its owning Torch module."""
    parent_cell, param_key, _ = result
    if param_key in parent_cell._parameters:  # pylint: disable=protected-access
        parent_cell._parameters[param_key] = new_param  # pylint: disable=protected-access
    else:
        parent_cell.register_parameter(param_key, new_param)
    return True


def get_op_name(func):
    """Return the registry name for a Torch callable or operator overload."""
    if hasattr(func, "__name__"):
        return func.__name__
    if isinstance(func, OpOverload):
        return func.name.split("::")[-1].split(".")[0]
    if isinstance(func, OpOverloadPacket):
        return func.name.split("::")[-1]
    func_str = str(func)
    if "built-in function" in func_str:
        return func_str.split()[-1].strip(">")
    if "function" in func_str:
        return func_str.split()[1]
    return "unknown_op"


def get_rank() -> int:
    """Return the current Torch distributed rank, or 0 before distributed init."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_group_local_rank(group=None) -> int:
    """Return local rank in a Torch process group, or 0 before distributed init."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    if group is None:
        return torch.distributed.get_rank()
    if hasattr(group, "rank"):
        return group.rank()
    return torch.distributed.get_group_rank(group, torch.distributed.get_rank())


def _is_floating_torch(tensor: Tensor) -> bool:
    """Check if PyTorch tensor is floating point."""
    return tensor.is_floating_point()


def distributed_cross_entropy(
        input_tensor: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
) -> Tensor:
    """Distributed cross_entropy entry used by shard dispatch."""
    # Defer the components import to preserve the lightweight models import boundary.
    from hyper_parallel.components.losses._vocab_parallel_cross_entropy import (  # pylint: disable=C0415
        DistributedCrossEntropyFunction,
    )

    input_dtensor = None
    mesh = None
    vocab_size = None

    if _is_dtensor(input_tensor):
        if not _is_shard_on_last_dim(input_tensor):
            raise ValueError(
                "input must be Shard(-1) on class dimension. "
                f"Got placements: {input_tensor.placements}"
            )
        input_dtensor = input_tensor
        mesh, _ = _get_mesh_and_dim(input_tensor)
        vocab_size = input_tensor.shape[-1]

    input_for_check = input_dtensor if input_dtensor is not None else input_tensor
    _check_context_and_layout(input_for_check)
    _validate_cross_entropy_params(
        input_tensor,
        target,
        weight,
        size_average,
        ignore_index,
        reduce,
        reduction,
        label_smoothing,
        _is_floating_torch,
    )

    if input_dtensor is None:
        raise ValueError(
            "input must be a DTensor when using loss_parallel. "
            f"Got type: {type(input_tensor)}"
        )

    input_local = _get_local_tensor(input_dtensor)
    local_vocab_size = input_local.shape[-1]
    if input_dtensor.ndim > 2:
        input_local = input_local.reshape(-1, local_vocab_size)
        target = target.reshape(-1)

    strict = _get_loss_parallel_strict()
    _validate_mesh_and_shard(input_dtensor, strict)
    return DistributedCrossEntropyFunction.apply(
        input_local,
        target,
        weight,
        ignore_index,
        reduction,
        vocab_size,
        mesh,
        0,
    )


def distributed_cross_entropy_from_op_call(
        op_call: Any,  # pylint: disable=unused-argument
        args: tuple,
        kwargs: dict,
):
    """Parse a cross_entropy op call and invoke the distributed Torch implementation."""
    input_tensor = args[0] if len(args) > 0 else kwargs.get("input")
    target = args[1] if len(args) > 1 else kwargs.get("target")
    weight = args[2] if len(args) > 2 else kwargs.get("weight")
    size_average = args[3] if len(args) > 3 else kwargs.get("size_average")
    ignore_index = args[4] if len(args) > 4 else kwargs.get("ignore_index", -100)
    reduce = args[5] if len(args) > 5 else kwargs.get("reduce")
    reduction = args[6] if len(args) > 6 else kwargs.get("reduction", "mean")
    label_smoothing = args[7] if len(args) > 7 else kwargs.get("label_smoothing", 0.0)

    return distributed_cross_entropy(
        input_tensor=input_tensor,
        target=target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
