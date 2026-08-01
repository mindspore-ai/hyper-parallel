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
"""Format-agnostic local-tensor extension contract for Torch fully_shard."""

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any, Literal, Protocol

import torch


_EXTENSION_MARKER = "_hp_fsdp_extension"
_REQUIRED_METHODS = (
    "hp_fsdp_to_dtensor",
    "fsdp_pre_all_gather",
    "fsdp_post_all_gather",
)


class FSDPLocalTensorExtension(Protocol):
    """Public contract for an external fully_shard local-tensor extension.

    HyperParallel owns even logical sharding through ``torch.chunk`` and all
    physical-tensor communication. The external package owns only conversion
    between its local logical tensor and physical communication tensors.

    ``fsdp_pre_all_gather(context)`` returns physical tensors plus opaque
    metadata. ``fsdp_post_all_gather(outputs, metadata, *, out=None)``
    returns ``(unsharded_tensor, inner_tensor)``. When ``out`` is supplied it
    must update and return that exact object. ``hp_fsdp_to_dtensor()`` only
    wraps the low-precision parameter. Fully-shard always wraps its BF16/FP32
    reduced gradient with HP's ordinary ``DTensor.from_local()``.
    """

    _hp_fsdp_extension: bool

    def hp_fsdp_to_dtensor(self, mesh: Any, placements: tuple) -> torch.Tensor:
        """Wrap the HP-created local logical shard for parameter storage."""

    def fsdp_pre_all_gather(
        self,
        context: "FSDPGatherContext",
    ) -> tuple[tuple[torch.Tensor, ...], Any]:
        """Expose physical local tensors for HP-managed all-gather."""

    def fsdp_post_all_gather(
        self,
        outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        *,
        out: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """Rebuild the unsharded logical tensor after communication."""


@dataclass(frozen=True)
class FSDPGatherContext:
    """Stable fully_shard context passed to a local-tensor all-gather extension.

    Args:
        phase: Training phase that requires the unsharded parameter.
        reshard_after_forward: Whether the owning fully_shard unit releases the
            unsharded parameter after forward.
        param_fqn: Fully qualified parameter name used for diagnostics.
    """

    phase: Literal["forward", "backward"]
    reshard_after_forward: bool
    param_fqn: str

    def __post_init__(self) -> None:
        """Validate context values at the scheduler-to-extension boundary."""
        if self.phase not in ("forward", "backward"):
            raise ValueError(
                "FSDPGatherContext.phase must be 'forward' or 'backward', "
                f"got {self.phase!r}."
            )
        if not isinstance(self.reshard_after_forward, bool):
            raise ValueError(
                "FSDPGatherContext.reshard_after_forward must be a bool, "
                f"got {type(self.reshard_after_forward).__name__}."
            )
        if not isinstance(self.param_fqn, str) or not self.param_fqn:
            raise ValueError("FSDPGatherContext.param_fqn must be a non-empty string.")


def is_fsdp_local_tensor_extension(local_tensor: Any) -> bool:
    """Return whether ``local_tensor`` explicitly enables the extension contract."""
    marker = getattr(local_tensor, _EXTENSION_MARKER, False)
    if not isinstance(marker, bool):
        raise ValueError(
            f"{_EXTENSION_MARKER} must be a bool on {type(local_tensor).__name__}, "
            f"got {type(marker).__name__}."
        )
    return marker


def validate_fsdp_local_tensor_extension(local_tensor: Any) -> bool:
    """Validate an extension declaration and return whether it is enabled.

    A tensor that exposes only part of the hook surface is rejected even when
    the explicit marker is missing. This prevents native-FSDP hooks from being
    consumed accidentally with incompatible HyperParallel semantics.

    Args:
        local_tensor: Local parameter tensor to inspect.

    Returns:
        Whether the tensor implements the HyperParallel extension contract.

    Raises:
        ValueError: If the marker or required callable methods are incomplete.
    """
    enabled = is_fsdp_local_tensor_extension(local_tensor)
    declared_methods = {
        method_name: callable(getattr(local_tensor, method_name, None))
        for method_name in _REQUIRED_METHODS
    }
    if not enabled:
        if any(declared_methods.values()):
            declared = [name for name, exists in declared_methods.items() if exists]
            raise ValueError(
                f"{type(local_tensor).__name__} declares fully_shard extension methods "
                f"{declared} but does not set {_EXTENSION_MARKER}=True."
            )
        return False

    missing = [name for name, exists in declared_methods.items() if not exists]
    if missing:
        raise ValueError(
            f"{type(local_tensor).__name__} sets {_EXTENSION_MARKER}=True but is missing "
            f"required callable methods: {missing}."
        )

    return True


def is_fsdp_flattenable(local_tensor: Any) -> bool:
    """Return whether a tensor may enter the ordinary flat-buffer path.

    Extension tensors can expose multiple physical communication tensors, so
    they always use the dedicated all-gather path rather than comm fusion.
    """
    return not validate_fsdp_local_tensor_extension(local_tensor)


def fsdp_to_dtensor(local_tensor: Any, mesh: Any, placements: tuple) -> Any:
    """Create the distributed wrapper selected by the local-tensor extension.

    Args:
        local_tensor: Validated extension local tensor.
        mesh: HyperParallel device mesh.
        placements: Placements for the distributed parameter.

    Returns:
        A DTensor-compatible wrapper produced by the extension.

    Raises:
        ValueError: If the returned object does not expose the DTensor contract
            required by fully_shard.
    """
    validate_fsdp_local_tensor_extension(local_tensor)
    distributed_tensor = local_tensor.hp_fsdp_to_dtensor(mesh, placements)
    if not isinstance(distributed_tensor, torch.Tensor):
        raise ValueError(
            f"{type(local_tensor).__name__}.hp_fsdp_to_dtensor() must return "
            "a torch.Tensor subclass so fully_shard can install it as an nn.Parameter."
        )
    required_attributes = ("layout", "to_local")
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(distributed_tensor, attribute)
    ]
    if missing:
        raise ValueError(
            f"{type(local_tensor).__name__}.hp_fsdp_to_dtensor() returned "
            f"{type(distributed_tensor).__name__}, which is missing {missing}."
        )
    return distributed_tensor


def fsdp_shard_tensor(
    full_tensor: Any,
    *,
    shard_dim: int,
    shard_rank: int,
    shard_world_size: int,
) -> torch.Tensor:
    """Use the HP-owned logical split and validate the external result.

    Extension tensors must support ``torch.chunk(..., dim=shard_dim)``,
    ``clone()``, and ``contiguous()`` while preserving their extension
    contract. HP owns rank selection and rejects uneven shards, so external
    packages never implement a second sharding policy.
    """
    if shard_world_size < 1 or not 0 <= shard_rank < shard_world_size:
        raise ValueError("Invalid fully_shard rank or world size.")
    if full_tensor.ndim == 0:
        raise NotImplementedError("fully_shard extension tensors cannot be scalar.")
    if full_tensor.size(shard_dim) % shard_world_size:
        raise NotImplementedError(
            "fully_shard extension tensors require even logical sharding: "
            f"shape={tuple(full_tensor.shape)}, dim={shard_dim}, "
            f"world_size={shard_world_size}."
        )
    shards = torch.chunk(full_tensor, shard_world_size, dim=shard_dim)
    if len(shards) != shard_world_size:
        raise ValueError("torch.chunk() returned an unexpected number of shards.")
    local_tensor = shards[shard_rank].clone().contiguous()
    if not isinstance(local_tensor, torch.Tensor):
        raise ValueError("Extension logical sharding must return a torch.Tensor.")
    validate_fsdp_local_tensor_extension(local_tensor)
    expected_shape = list(full_tensor.shape)
    expected_shape[shard_dim] //= shard_world_size
    if tuple(local_tensor.shape) != tuple(expected_shape):
        raise ValueError(
            "Extension logical sharding changed an unsupported dimension: "
            f"expected {tuple(expected_shape)}, got {tuple(local_tensor.shape)}."
        )
    return local_tensor


def fsdp_pre_all_gather(
    local_tensor: Any,
    context: FSDPGatherContext,
) -> tuple[tuple[torch.Tensor, ...], Any]:
    """Collect physical tensors and opaque metadata for extension all-gather.

    Extensions own the relationship between their logical tensor and physical
    communication tensors.  Fully-shard only validates and gathers the returned
    tensors, then gives the gathered buffers and metadata back to the extension.
    """
    validate_fsdp_local_tensor_extension(local_tensor)
    result = local_tensor.fsdp_pre_all_gather(context)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            f"{type(local_tensor).__name__}.fsdp_pre_all_gather() must return "
            "(tensors, metadata)."
        )
    tensors, metadata = result
    if not isinstance(tensors, Iterable):
        raise ValueError("fsdp_pre_all_gather() tensors must be an iterable.")
    tensors = tuple(tensors)
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError(
            "fsdp_pre_all_gather() must return one or more torch.Tensor values."
        )
    return tensors, metadata


def fsdp_post_all_gather(
    local_tensor: Any,
    gathered_tensors: tuple[torch.Tensor, ...],
    metadata: Any,
    *,
    out: Any = None,
) -> tuple[torch.Tensor, Any]:
    """Rebuild an extension tensor after all physical gathers complete.

    The extension receives only gathered physical tensors and its own metadata.
    It must return ``(unsharded_tensor, inner_tensor)``. ``inner_tensor`` is
    opaque to HyperParallel and allows an external implementation to retain
    its physical storage or auxiliary reconstruction state. Passing ``out``
    lets an extension reuse an existing unsharded representation when safe.
    """
    result = local_tensor.fsdp_post_all_gather(
        gathered_tensors,
        metadata,
        out=out,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            "fsdp_post_all_gather() must return (unsharded_tensor, inner_tensor)."
        )
    unsharded_tensor, inner_tensor = result
    if not isinstance(unsharded_tensor, torch.Tensor):
        raise ValueError(
            "fsdp_post_all_gather() unsharded_tensor must be a torch.Tensor."
        )
    if not is_fsdp_local_tensor_extension(unsharded_tensor):
        raise ValueError(
            "fsdp_post_all_gather() must rebuild a fully-shard extension tensor."
        )
    if out is not None and unsharded_tensor is not out:
        raise ValueError(
            "fsdp_post_all_gather(..., out=...) must return that exact out "
            "object so optimizer and autograd references remain valid."
        )
    return unsharded_tensor, inner_tensor
