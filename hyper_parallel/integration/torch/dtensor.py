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
"""Hyper-only tensor adapter for explicitly enabled Torch-framework integration.

This helper is not a Tensor subclass or an autograd bridge. Callers explicitly
use it at local-storage boundaries; native Torch type checks remain
unchanged. Construction and local access do not gather or reduce tensors.
"""

from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh as TorchMesh
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Replicate as TorchReplicate
from torch.distributed.tensor import Shard as TorchShard

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor as HyperDTensor
from hyper_parallel.core.dtensor.dtensor import (
    distribute_tensor as hyper_distribute_tensor,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def to_hyper_mesh(mesh: Any) -> DeviceMesh:
    """Reuse existing process groups when translating a Torch mesh descriptor."""
    if isinstance(mesh, DeviceMesh):
        return mesh
    if not isinstance(mesh, TorchMesh):
        raise TypeError("Expected a Torch or Hyper DeviceMesh.")
    return DeviceMesh.from_group(
        mesh.get_all_groups(),
        mesh.device_type,
        mesh=mesh.mesh,
        mesh_dim_names=mesh.mesh_dim_names,
    )


def to_hyper_placement(placement: Any) -> Any:
    """Translate supported placement descriptors without redistributing tensors."""
    if placement is None or isinstance(placement, (Shard, Replicate)):
        return placement
    # Exact classes avoid erasing additional semantics of special shard types.
    converters = {
        TorchShard: lambda value: Shard(value.dim),
        TorchReplicate: lambda value: Replicate(),
    }
    convert = converters.get(type(placement))
    if convert is not None:
        return convert(placement)
    raise NotImplementedError(
        "Torch compatibility currently supports Shard and Replicate only."
    )


def to_torch_view(value: Any) -> Any:
    """Return a detached Torch checkpoint view sharing a Hyper tensor's storage.

    No gather, reduction or autograd bridge is introduced. Ordinary values pass
    through unchanged. Only ordinary Shard and Replicate layouts are supported;
    special placements must not lose their ordering at this boundary.
    """
    if not isinstance(value, HyperDTensor):
        return value
    converters = {
        Shard: lambda placement: TorchShard(placement.dim),
        Replicate: lambda placement: TorchReplicate(),
    }
    placements = []
    for placement in value.placements:
        convert = converters.get(type(placement))
        if convert is None:
            raise NotImplementedError(
                f"Checkpoint views do not support {type(placement).__name__}; "
                "only ordinary Shard and Replicate placements are supported."
            )
        placements.append(convert(placement))
    mesh = value.device_mesh
    torch_mesh = TorchMesh.from_group(
        mesh.get_all_groups(),
        mesh.device_type,
        mesh=torch.tensor(mesh.mesh.tolist()),
        mesh_dim_names=mesh.mesh_dim_names,
    )
    return TorchDTensor.from_local(
        value.to_local().detach(),
        torch_mesh,
        placements,
        run_check=False,
        shape=value.shape,
        stride=(
            value.layout.tensor_stride
            if value.layout.tensor_stride is not None
            else torch.empty(value.shape, device="meta").stride()
        ),
    )


def _check_reference_partition(
    value: TorchDTensor, reference: HyperDTensor, placements: list
) -> None:
    """Require DCP to have loaded the owning parameter's rank-local partition."""
    # Shard subclasses carry partition metadata that must remain intact.
    reference_axes = [
        Shard(placement.dim) if type(placement) is Shard else placement  # pylint: disable=unidiomatic-typecheck
        for placement in reference.placements
    ]
    if (
        tuple(value.shape) != tuple(reference.shape)
        or tuple(value.to_local().shape) != tuple(reference.to_local().shape)
        or placements != reference_axes
        or value.device_mesh.mesh.tolist() != reference.device_mesh.mesh.tolist()
    ):
        raise ValueError(
            "Checkpoint shard does not match its reference parameter layout."
        )


def from_torch_view(value: Any, *, reference: HyperDTensor | None = None) -> Any:
    """Wrap a Torch checkpoint shard using Hyper without copying local storage.

    With a reference, retain its mesh and FSDP ceil-chunk placements after
    checking the global shape, local shape, rank order and partition axes.
    DCP owns any actual resharding before this conversion. Without a reference,
    translate only the ordinary Torch placement descriptors.
    """
    if not isinstance(value, TorchDTensor):
        return value
    placements = [to_hyper_placement(placement) for placement in value.placements]
    if reference is not None:
        _check_reference_partition(value, reference, placements)
        mesh, placements = reference.device_mesh, reference.placements
        stride = reference.layout.tensor_stride
    else:
        mesh, stride = to_hyper_mesh(value.device_mesh), value.stride()
    return HyperDTensor.from_local(
        value.to_local(), mesh, placements, shape=value.shape, stride=stride
    )


def is_dtensor(value: object) -> bool:
    """Recognize Hyper distributed tensors without modifying Torch type checks."""
    return isinstance(value, HyperDTensor)


class DTensorAdapter:
    """Expose Hyper local-storage operations without replacing training parameters.

    Args:
        tensor: Existing Hyper distributed tensor.

    Raises:
        TypeError: The input is not a supported DTensor.
    """

    def __init__(self, tensor: HyperDTensor) -> None:
        """Keep the existing Hyper tensor; do not convert its storage or layout."""
        if not is_dtensor(tensor):
            raise TypeError("DTensorAdapter requires a Hyper DTensor.")
        self.tensor = tensor

    def to_local(self) -> torch.Tensor:
        """Return existing local storage, without changing autograd semantics."""
        return self.tensor.to_local()

    def local_grad(self) -> torch.Tensor | None:
        """Return the local gradient, or None; do not reduce or create gradients."""
        grad = getattr(self.tensor, "main_grad", None)
        if grad is None:
            grad = self.tensor.grad
        return grad.to_local() if is_dtensor(grad) else grad

    def from_local(self, local: torch.Tensor) -> HyperDTensor:
        """Wrap a same-shaped local shard using Hyper and the reference layout.

        Autograd behavior remains that of Hyper's from_local.
        """
        self._validate_local(local)
        options = {"shape": tuple(self.tensor.shape), "run_check": False}
        if self.tensor.layout.tensor_stride is not None:
            options["stride"] = self.tensor.layout.tensor_stride
        return HyperDTensor.from_local(
            local, self.tensor.device_mesh, self.tensor.placements, **options
        )

    @torch.no_grad()
    def copy_local_(self, local: torch.Tensor) -> None:
        """Copy a local payload in place, preserving parameter and storage identity.

        The caller owns stream synchronization and global layout correctness.
        Device and dtype must match so this boundary introduces no implicit cast
        or device transfer. Distributed inputs must be explicitly unwrapped.
        """
        self._validate_local(local)
        target = self.to_local()
        if target.dtype != local.dtype or target.device != local.device:
            raise ValueError("Local payload must match the reference device and dtype.")
        target.copy_(local)

    @torch.no_grad()
    def copy_from_full_(
        self, full: torch.Tensor, *, src_data_rank: int | None = None
    ) -> None:
        """Distribute a full payload with Hyper and copy its local shard.

        The caller owns checkpoint reading, rank broadcasts and dtype/device
        conversion. With src_data_rank=None all ranks supply identical data and
        the backend slices locally. An integer requests backend source-rank
        communication; every mesh member must participate. Layout support and
        collective semantics are those of the backend's distribute_tensor.
        """
        if not isinstance(full, torch.Tensor) or isinstance(
            full, (HyperDTensor, TorchDTensor)
        ):
            raise TypeError("Expected a plain full Tensor.")
        if tuple(full.shape) != tuple(self.tensor.shape):
            raise ValueError("Full payload shape must match the logical global shape.")
        distributed = hyper_distribute_tensor(
            full,
            self.tensor.device_mesh,
            self.tensor.placements,
            src_data_rank=src_data_rank,
        )
        self.copy_local_(distributed.to_local())

    def _validate_local(self, local: torch.Tensor) -> None:
        if not isinstance(local, torch.Tensor) or isinstance(
            local, (HyperDTensor, TorchDTensor)
        ):
            raise TypeError("Expected a plain local Tensor, not a distributed tensor.")
        if tuple(local.shape) != tuple(self.to_local().shape):
            raise ValueError(
                f"Local shape {tuple(local.shape)} does not match {tuple(self.to_local().shape)}."
            )
