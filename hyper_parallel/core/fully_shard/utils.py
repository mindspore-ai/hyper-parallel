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
"""Torch helpers, policies and mesh metadata for fully_shard.

The fully_shard core used to reach process-group management, module traversal and
tensor plumbing through the platform abstraction layer. Only the torch backend is
supported now, so those implementations live here and every caller calls them
directly.
"""
# pylint: disable=C9006,C9007
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.core.utils.communication import get_device_handle
from hyper_parallel.core.utils.communication import get_group_local_rank

DType = torch.dtype


# ---------------------------------------------------------------------------
# Module traversal
# ---------------------------------------------------------------------------

def get_modules(module: nn.Module):
    """Return every sub-module contained in ``module``.

    Args:
        module: The root module to traverse.

    Returns:
        An iterator over the module tree.
    """
    return module.modules()


def parameters_dict(cell: nn.Module):
    """Return every named parameter registered by the module tree.

    Args:
        cell: The root module to traverse.

    Returns:
        An iterator of ``(name, parameter)`` pairs.
    """
    return cell.named_parameters()


def buffers_dict(cell: nn.Module):
    """Return every named buffer registered by the module tree.

    Args:
        cell: The root module to traverse.

    Returns:
        An iterator of ``(name, buffer)`` pairs.
    """
    return cell.named_buffers()


# ---------------------------------------------------------------------------
# Tensor plumbing
# ---------------------------------------------------------------------------

def load_into_param(param: Tensor, data: Tensor) -> None:
    """Write ``data`` into ``param``, materialising a meta local tensor first.

    Args:
        param: The destination parameter, possibly a DTensor.
        data: The data to write into it.
    """
    from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=C0415

    if isinstance(param, DTensor):
        local = param._local_tensor  # pylint: disable=protected-access
        if local.is_meta:
            orig_requires_grad = param.requires_grad
            param._local_tensor = data  # pylint: disable=protected-access
            if data.requires_grad != orig_requires_grad:
                param.requires_grad_(orig_requires_grad)
        else:
            local.copy_(data)
    else:
        param.copy_(data)


def cast_fp_tensor(dtype: DType, x: Any) -> Any:
    """Cast a floating-point tensor to ``dtype``, leaving anything else alone.

    Args:
        dtype: The target dtype.
        x: The candidate tensor.

    Returns:
        Any: The cast tensor, or ``x`` unchanged when it is not a floating-point tensor.
    """
    if not isinstance(x, Tensor) or not torch.is_floating_point(x) or x.dtype == dtype:
        return x
    return x.to(dtype)


def apply_to_tensors(fn: Callable[[Tensor], Any], container: Any) -> Any:
    """Recursively apply ``fn`` to every tensor inside ``container``.

    Handles tensors, dataclasses, ordered and plain dicts, named tuples,
    lists, tuples, sets and packed sequences.

    Args:
        fn: The callable applied to each tensor.
        container: The structure to walk.

    Returns:
        Any: A structure of the same shape with every tensor replaced by ``fn``'s result.
    """

    def apply(x):
        if isinstance(x, Tensor):
            return fn(x)
        if hasattr(x, "__dataclass_fields__"):
            dc = replace(x)
            changes = {f.name: apply(getattr(dc, f.name)) for f in fields(dc)}
            return replace(dc, **changes)
        if isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        if isinstance(x, PackedSequence):
            apply(x.data)
            return x
        if isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        if isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields"):
            res = (apply(el) for el in x)
            return type(x)(*res)
        if isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        return x

    return apply(container)


def profiler_record(name: str):
    """Return the torch profiler annotation context for ``name``.

    Args:
        name: Label shown in profiler traces for the enclosed region.

    Returns:
        A context manager that records the region when the profiler is active.
    """
    return torch.profiler.record_function(name)


# ---------------------------------------------------------------------------
# Gradient-ready stream
# ---------------------------------------------------------------------------
# The handle and its stream are per-process singletons: at most one gradient
# reduction is in flight across the whole process, so the state belongs at
# module scope rather than on any scheduler or parameter object.
_GRAD_READY_STATE: Dict[str, Any] = {
    "handle": None,
    "post_process": None,
    "stream": None,
}


def _process_pending_grad_handle() -> None:
    """Wait for the in-flight gradient handle and run its post-process callback."""
    handle = _GRAD_READY_STATE["handle"]
    if handle is None:
        return
    handle.wait()
    post_process = _GRAD_READY_STATE["post_process"]
    if post_process is not None:
        post_process()


def _grad_ready_stream():
    """Return the stream used to order gradient-handle waits, creating it lazily."""
    if _GRAD_READY_STATE["stream"] is None:
        _GRAD_READY_STATE["stream"] = get_device_handle().Stream()
    return _GRAD_READY_STATE["stream"]


def grad_ready_stream():
    """Return the context manager that switches to the gradient-ready stream.

    Returns:
        The device stream context manager guarding the synchronisation stream.
    """
    return get_device_handle().stream(_grad_ready_stream())


def set_grad_reduce_handle(handle: Any, post_process: Optional[Callable[[], None]] = None) -> None:
    """Record a new in-flight gradient reduction handle.

    Any previously recorded handle is waited on first, so at most one handle is
    ever outstanding.

    Args:
        handle: The async work handle returned by the reduction collective.
        post_process: Callback run once ``handle`` completes, or ``None``.
    """
    with grad_ready_stream():
        _process_pending_grad_handle()
    _GRAD_READY_STATE["handle"] = handle
    _GRAD_READY_STATE["post_process"] = post_process


def wait_grad_handle() -> None:
    """Block until the in-flight gradient reduction handle completes and clear it."""
    handle = _GRAD_READY_STATE["handle"]
    if handle is None:
        return
    with grad_ready_stream():
        _process_pending_grad_handle()
        sync_event = _grad_ready_stream().record_event()
    sync_event.wait()
    _GRAD_READY_STATE["handle"] = None
    _GRAD_READY_STATE["post_process"] = None


# ---------------------------------------------------------------------------
# Policies and mesh metadata
# ---------------------------------------------------------------------------

@dataclass
class MixedPrecisionPolicy:
    """
    Configures mixed precision training for HSDP.

    This policy controls data type casting during forward/backward computation
    and gradient reduction, enabling memory savings and potential speedups.

    Attributes:
        param_dtype: Data type for parameter computation. If None, uses original dtype.
        reduce_dtype: Data type for gradient reduction. If None, uses param_dtype.
        output_dtype: Data type for module outputs. If None, no casting applied.
    """
    param_dtype: Optional[DType] = None
    reduce_dtype: Optional[DType] = None
    output_dtype: Optional[DType] = None
    cast_forward_inputs: bool = True
    apply_grad_on_fp32_main_grad: bool = False


@dataclass
class OffloadPolicy:
    """
    Base class for offload policies.

    This represents no offloading and serves as the default policy.
    Subclass this to implement custom offload strategies.
    """


@dataclass
class CPUOffloadPolicy(OffloadPolicy):
    """
    Offloads sharded parameters and gradients to CPU memory.

    When enabled, sharded parameters are kept on CPU and copied to device
    before all-gather. Gradients are copied back to CPU after backward.
    This reduces NPU memory usage at the cost of additional data transfers.

    Attributes:
        pin_memory: If True, pins CPU memory for faster H2D/D2H transfers
            and enables overlap with computation. Disable if CPU memory
            is constrained. (Default: True)
    """
    pin_memory: bool = True

@dataclass
class CommFusionPolicy():
    enable_comm_fusion: bool = False
    comm_fusion_zero_copy: bool = False


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    def __post_init__(self):
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError(
                "At least one of shard_mesh_dim and replicate_mesh_dim must not be None"
            )


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size: int = self.mesh.mesh_shape[self.shard_mesh_dim]
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank: int = get_group_local_rank(self.shard_process_group)


@dataclass
class DDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self):
        super().__post_init__()
        if self.replicate_mesh_dim is None:
            raise AssertionError("Expects non-None replicate_mesh_dim")
        self.replicate_mesh_size: int = self.mesh.mesh_shape[self.replicate_mesh_dim]
        self.replicate_process_group = self.mesh.get_group(self.replicate_mesh_dim)
        self.replicate_mesh_rank: int = get_group_local_rank(self.replicate_process_group)


@dataclass
class HSDPMeshInfo(FSDPMeshInfo, DDPMeshInfo):
    # pylint: disable=W0246
    def __post_init__(self):
        # Calls `FSDPMeshInfo` -> `DDPMeshInfo` -> `DataParallelMeshInfo`
        super().__post_init__()


@dataclass(frozen=True)
class SourceShardMetaInfo:
    """Describe a parameter's source TP/EP layout before fully_shard."""

    mesh: DeviceMesh
    placements: tuple[Placement, ...]
    origin_is_dtensor: bool = False


__all__ = [
    # Module traversal and tensor plumbing.
    "DType",
    "get_modules",
    "parameters_dict",
    "buffers_dict",
    "load_into_param",
    "cast_fp_tensor",
    "apply_to_tensors",
    "profiler_record",
    # Grad-reduce handle choreography.
    "grad_ready_stream",
    "set_grad_reduce_handle",
    "wait_grad_handle",
    # Precision/offload policies and mesh metadata.
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "CPUOffloadPolicy",
    "CommFusionPolicy",
    "DataParallelMeshInfo",
    "FSDPMeshInfo",
    "DDPMeshInfo",
    "HSDPMeshInfo",
    "SourceShardMetaInfo",
    # Re-exported from :mod:`hyper_parallel.core.utils.communication` because
    # callers here and in the tests reach them through this module.
    "get_device_handle",
    "get_group_local_rank",
]
