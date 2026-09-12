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
"""Thread-local load-balance state shared by shard attention operators."""

import threading
from typing import Any, Optional, Tuple

import os
import torch
import torch.distributed.nn.functional as dist_func
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch._ops import OpOverload, OpOverloadPacket

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
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

_LB_OVERRIDE = threading.local()


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


def _set_lb_override(split_id: int, split_num: int) -> None:
    """Set per-thread split_id/split_num override for load-balance sub-calls."""
    _LB_OVERRIDE.split_id = split_id
    _LB_OVERRIDE.split_num = split_num


def _clear_lb_override() -> None:
    """Clear the load-balance override."""
    _LB_OVERRIDE.split_id = None
    _LB_OVERRIDE.split_num = None


def _get_lb_override() -> Tuple[Optional[int], Optional[int]]:
    """Return (split_id, split_num) if override is active, else (None, None)."""
    return getattr(_LB_OVERRIDE, "split_id", None), getattr(_LB_OVERRIDE, "split_num", None)


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


def _differentiable_all_reduce(tensor: Tensor, op: str, group) -> Tensor:
    """Run a differentiable Torch all-reduce."""
    reduce_op = torch.distributed.ReduceOp.MAX if op == "max" else torch.distributed.ReduceOp.SUM
    return dist_func.all_reduce(tensor, op=reduce_op, group=group)


def _is_floating_torch(tensor: Tensor) -> bool:
    """Check if PyTorch tensor is floating point."""
    return tensor.is_floating_point()


def _compute_vocab_start(vocab_size: int, tp_size: int, rank: int) -> int:
    """Compute the starting index for this rank's vocab shard."""
    chunk_size = (vocab_size + tp_size - 1) // tp_size
    return rank * chunk_size


def distributed_log_softmax(logits_local: Tensor, dim: int, mesh: DeviceMesh, mesh_dim: int = 0) -> Tensor:
    """Stable log-softmax on a class-sharded dimension."""
    max_local = logits_local.max(dim=dim, keepdim=True).values
    group = mesh.get_group(mesh_dim)
    max_global = _differentiable_all_reduce(max_local, op="max", group=group)

    exp_local = (logits_local - max_global).exp()
    sum_local = exp_local.sum(dim=dim, keepdim=True)
    sum_global = _differentiable_all_reduce(sum_local, op="sum", group=group)
    return logits_local - max_global - sum_global.log()


def distributed_nll_loss_forward(
        log_probs: Tensor,
        target: Tensor,
        weight: Optional[Tensor],
        ignore_index: int,
        reduction: str,
        vocab_start: int,
        vocab_end: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Index target classes and compute the local NLL contribution."""
    batch_size = target.numel()
    target_flat = target.flatten()
    target_mask = (target_flat >= vocab_start) & (target_flat < vocab_end)
    ignore_mask = target_flat != ignore_index
    target_mask = target_mask & ignore_mask

    if reduction == "none":
        loss = torch.zeros(batch_size, dtype=log_probs.dtype, device=log_probs.device)
    else:
        loss = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)
    total_weight = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)

    if target_mask.any():
        local_target = target_flat[target_mask] - vocab_start
        log_probs_2d = log_probs.reshape(-1, log_probs.shape[-1])
        row_indices = torch.where(target_mask)[0]
        selected_log_probs = log_probs_2d[row_indices, local_target]

        if weight is not None:
            global_target = target_flat[target_mask]
            sample_weights = weight[global_target]
            selected_log_probs = selected_log_probs * sample_weights
            total_weight = sample_weights.sum().reshape(1)
        else:
            total_weight = torch.tensor(
                target_mask.sum().item(), dtype=log_probs.dtype, device=log_probs.device
            ).reshape(1)

        nll = -selected_log_probs
        if reduction == "none":
            loss_flat = torch.zeros(batch_size, dtype=log_probs.dtype, device=log_probs.device)
            loss_flat[target_mask] = nll
            loss = loss_flat.reshape(target.shape)
        elif reduction == "sum":
            loss = nll.sum().unsqueeze(0)
        else:
            loss = nll.sum().unsqueeze(0)
    else:
        if reduction == "none":
            loss = torch.zeros(batch_size, dtype=log_probs.dtype, device=log_probs.device).reshape(target.shape)
        total_weight = torch.zeros(1, dtype=log_probs.dtype, device=log_probs.device)

    return loss, total_weight, target_mask, torch.tensor(vocab_start, dtype=torch.long, device=log_probs.device)


class DistributedCrossEntropyFunction(torch.autograd.Function):
    """Fused backward for distributed cross_entropy."""

    @staticmethod
    def forward(
            ctx: Any,
            input_local: Tensor,
            target: Tensor,
            weight: Optional[Tensor],
            ignore_index: int,
            reduction: str,
            vocab_size: int,
            mesh: DeviceMesh,
            mesh_dim: int,
    ) -> Tensor:
        """Forward pass."""
        local_vocab_size = input_local.shape[-1]
        rank = mesh.get_local_rank(mesh_dim)
        tp_size = mesh.size(mesh_dim)
        vocab_start = _compute_vocab_start(vocab_size, tp_size, rank)
        vocab_end = vocab_start + local_vocab_size

        log_probs_local = distributed_log_softmax(input_local, dim=-1, mesh=mesh, mesh_dim=mesh_dim)
        loss, total_weight, target_mask, vocab_start_tensor = distributed_nll_loss_forward(
            log_probs_local, target, weight, ignore_index, reduction, vocab_start, vocab_end
        )

        if reduction == "mean":
            group = mesh.get_group(mesh_dim)
            total_loss = _differentiable_all_reduce(loss, op="sum", group=group)
            total_weight_sum = _differentiable_all_reduce(total_weight, op="sum", group=group)
            ctx.save_for_backward(
                input_local, log_probs_local, target, weight, total_weight_sum, target_mask, vocab_start_tensor
            )
            ctx.reduction = reduction
            ctx.ignore_index = ignore_index
            ctx.vocab_size = vocab_size
            ctx.local_vocab_size = local_vocab_size
            ctx.mesh = mesh
            ctx.mesh_dim = mesh_dim
            ctx.vocab_start = vocab_start
            ctx.vocab_end = vocab_end
            if total_weight_sum.item() == 0:
                return torch.tensor(float("nan"), dtype=total_loss.dtype, device=total_loss.device)
            return total_loss / total_weight_sum

        if reduction == "sum":
            group = mesh.get_group(mesh_dim)
            total_loss = _differentiable_all_reduce(loss, op="sum", group=group)
            ctx.save_for_backward(
                input_local,
                log_probs_local,
                target,
                weight,
                torch.zeros(1, dtype=loss.dtype, device=loss.device),
                target_mask,
                vocab_start_tensor,
            )
            ctx.reduction = reduction
            ctx.ignore_index = ignore_index
            ctx.vocab_size = vocab_size
            ctx.local_vocab_size = local_vocab_size
            ctx.mesh = mesh
            ctx.mesh_dim = mesh_dim
            ctx.vocab_start = vocab_start
            ctx.vocab_end = vocab_end
            return total_loss

        ctx.save_for_backward(
            input_local,
            log_probs_local,
            target,
            weight,
            torch.zeros(1, dtype=loss.dtype, device=loss.device),
            target_mask,
            vocab_start_tensor,
        )
        ctx.reduction = reduction
        ctx.ignore_index = ignore_index
        ctx.vocab_size = vocab_size
        ctx.local_vocab_size = local_vocab_size
        ctx.mesh = mesh
        ctx.mesh_dim = mesh_dim
        ctx.vocab_start = vocab_start
        ctx.vocab_end = vocab_end
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Backward pass."""
        _, log_probs_local, target, weight, total_weight, _, _ = ctx.saved_tensors
        reduction = ctx.reduction
        ignore_index = ctx.ignore_index
        vocab_start = ctx.vocab_start
        vocab_end = ctx.vocab_end

        batch_size = target.numel()
        target_flat = target.flatten()
        softmax_local = log_probs_local.exp()
        ignore_mask = target_flat != ignore_index
        sample_weights = weight[target_flat] if weight is not None else None

        if reduction == "mean":
            grad_scale = grad_output / total_weight.clamp(min=1e-12)
        elif reduction == "sum":
            grad_scale = grad_output
        else:
            grad_scale = grad_output.flatten()

        in_vocab_mask = (target_flat >= vocab_start) & (target_flat < vocab_end) & ignore_mask
        if reduction == "none":
            grad_scale_expanded = grad_scale.unsqueeze(-1)
            if sample_weights is not None:
                grad_scale_expanded = grad_scale_expanded * sample_weights.unsqueeze(-1)
            grad_input = softmax_local * grad_scale_expanded
        else:
            if sample_weights is not None:
                grad_scale = grad_scale * sample_weights.unsqueeze(-1)
            grad_input = softmax_local * grad_scale.unsqueeze(-1)

        local_targets = torch.where(in_vocab_mask, target_flat - vocab_start, torch.zeros_like(target_flat))
        if in_vocab_mask.any():
            row_indices = torch.arange(batch_size, device=target.device, dtype=torch.long)
            if reduction == "none":
                grad_values = -grad_scale * sample_weights if sample_weights is not None else -grad_scale
            else:
                grad_values = -grad_scale.expand_as(target_flat)
            grad_input = grad_input.contiguous()
            grad_input[row_indices[in_vocab_mask], local_targets[in_vocab_mask]] += grad_values[in_vocab_mask]

        if not ignore_mask.all():
            if reduction == "none":
                grad_input[~ignore_mask] = 0.0
            else:
                ignore_indices_expanded = (~ignore_mask).unsqueeze(-1).expand_as(grad_input)
                grad_input[ignore_indices_expanded] = 0.0

        return grad_input, None, None, None, None, None, None, None


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
