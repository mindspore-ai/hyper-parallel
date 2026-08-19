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
"""Common utilities for loss_parallel operations.

This module contains platform-agnostic helper functions and validation logic
shared between PyTorch and MindSpore implementations.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, TYPE_CHECKING

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.tensor_parallel.loss_parallel import is_loss_parallel_active

if TYPE_CHECKING:
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh

__all__ = [
    "_is_dtensor",
    "_is_shard_on_last_dim",
    "_is_replicate",
    "_get_mesh_and_dim",
    "_get_local_tensor",
    "_get_full_tensor",
    "_validate_target_type_base",
    "_validate_mesh_and_shard",
    "_validate_cross_entropy_params",
    "_check_context_and_layout",
]


def _is_dtensor(obj: Any) -> bool:
    """Check if object is a DTensor."""
    return isinstance(obj, DTensor)


def _is_shard_on_last_dim(dtensor: DTensor) -> bool:
    """Check if DTensor is Shard on the last dimension."""
    if not _is_dtensor(dtensor):
        return False
    placements = dtensor.placements
    if not placements:
        return False
    last_dim = len(dtensor.shape) - 1
    return any(
        isinstance(placement, Shard) and placement.dim in (-1, last_dim)
        for placement in placements
    )


def _is_replicate(dtensor: DTensor) -> bool:
    """Check if DTensor is Replicate."""
    if not _is_dtensor(dtensor):
        return False
    return all(isinstance(p, Replicate) for p in dtensor.placements)


def _get_mesh_and_dim(dtensor: DTensor) -> tuple[Optional["DeviceMesh"], Optional[int]]:
    """Get the mesh and mesh dimension that shard the class dimension."""
    mesh = dtensor.device_mesh
    last_dim = len(dtensor.shape) - 1
    for mesh_dim, placement in enumerate(dtensor.placements):
        if isinstance(placement, Shard) and placement.dim in (-1, last_dim):
            return mesh, mesh_dim
    return mesh, None


def _get_local_tensor(tensor):
    """Get DTensor local tensor, or return regular tensor."""
    if _is_dtensor(tensor):
        return tensor._local_tensor  # type: ignore  # pylint: disable=W0212
    return tensor


def _get_full_tensor(tensor):
    """Get full tensor (all_gather)."""
    if _is_dtensor(tensor):
        return tensor.full_tensor()  # type: ignore
    return tensor


def _validate_target_type_base(is_floating: bool) -> None:
    """Validate target type (base check).

    Args:
        is_floating: Whether target tensor has floating dtype.

    Raises:
        ValueError: If target is floating point (probabilistic).
    """
    if is_floating:
        raise ValueError(
            "Probabilistic target (float) is not supported in loss_parallel. "
            "Target must be class indices (int64)."
        )


def _validate_mesh_and_shard(dtensor: DTensor, strict: bool = True) -> None:
    """Validate mesh and Shard dimension.

    Args:
        dtensor: Input DTensor.
        strict: Whether to raise error or warn on validation failure.

    Raises:
        ValueError: If the class dimension is not sharded on exactly one mesh
            dimension in strict mode.
    """
    last_dim = len(dtensor.shape) - 1
    class_shard_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(dtensor.placements)
        if isinstance(placement, Shard) and placement.dim in (-1, last_dim)
    ]
    if len(class_shard_dims) != 1:
        message = (
            "Expected the class dimension to be sharded on exactly one mesh "
            f"dimension. Got placements: {dtensor.placements}"
        )
        if strict:
            raise ValueError(message)
        warnings.warn(f"{message}. This may cause incorrect results.")


def _validate_cross_entropy_params(
    input_tensor,
    target,
    weight: Optional[Any],
    size_average: Optional[bool],
    ignore_index: int,  # pylint: disable=W0613
    reduce: Optional[bool],
    reduction: str,
    label_smoothing: float,
    is_floating_fn,
) -> None:
    """Validate cross_entropy parameter support.

    Args:
        input_tensor: Input tensor.
        target: Target tensor.
        weight: Optional weight tensor.
        size_average: Deprecated parameter.
        ignore_index: Index to ignore (reserved for future use).
        reduce: Deprecated parameter.
        reduction: Reduction method.
        label_smoothing: Label smoothing factor.
        is_floating_fn: Platform-specific function to check if tensor is floating.

    Raises:
        ValueError: If parameters are invalid.
    """
    _validate_target_type_base(is_floating_fn(target))

    if _is_dtensor(input_tensor):
        if not _is_shard_on_last_dim(input_tensor):
            raise ValueError(
                "input must be Shard(-1) on class dimension. "
                f"Got placements: {input_tensor.placements}"
            )
    else:
        raise ValueError(
            "input must be a DTensor when using loss_parallel. "
            f"Got type: {type(input)}"
        )

    if weight is not None and _is_dtensor(weight):
        if not _is_replicate(weight):
            raise ValueError(
                "weight must be Replicate when it's a DTensor. "
                f"Got placements: {weight.placements}"
            )

    if size_average is not None or reduce is not None:
        warnings.warn(
            "size_average and reduce arguments are deprecated. "
            "Please use reduction='mean' or reduction='sum' instead.",
            DeprecationWarning,
        )

    if label_smoothing != 0.0:
        raise ValueError(
            "label_smoothing is not supported in loss_parallel. "
            "Please set label_smoothing=0.0 or disable loss_parallel."
        )

    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'none', 'mean', or 'sum'.")


def _check_context_and_layout(dtensor: DTensor) -> None:  # pylint: disable=W0613
    """Check context and layout.

    Args:
        dtensor: Input DTensor (reserved for future layout validation).

    Raises:
        ValueError: If not in loss_parallel context.
    """
    if not is_loss_parallel_active():
        raise ValueError(
            "Shard logits detected but not in loss_parallel context. "
            "Please wrap with loss_parallel() context manager."
        )
