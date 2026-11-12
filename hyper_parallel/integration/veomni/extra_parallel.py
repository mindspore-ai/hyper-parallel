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
"""ExtraParallel boundary descriptors without changing model-owned parallelism."""

import math
from functools import lru_cache
from types import MethodType

import torch
from torch.distributed.tensor import DTensor as TorchDTensor

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.integration.torch.dtensor import from_torch_view, to_hyper_mesh


def bind_gradient_divisor(module: torch.nn.Module, group_size: int) -> None:
    """Translate Torch's custom reduction divisor to Hyper's native averaged reduction.

    Hyper already averages over its FSDP/HSDP domain. Multiplying the result
    by domain_size/divisor gives the requested sum/divisor, including EP.
    The bindings belong to this explicitly adapted module, not the core class.
    """

    def set_gradient_divide_factor(self: torch.nn.Module, factor: float) -> None:
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (float, int))
            or not math.isfinite(factor)
            or factor <= 0
        ):
            raise ValueError("Gradient divide factor must be a finite positive number.")
        self.set_gradient_scaling_factor(group_size / factor)

    module.set_gradient_divide_factor = MethodType(set_gradient_divide_factor, module)
    module.set_reduce_scatter_divide_factor = module.set_gradient_divide_factor


@lru_cache(maxsize=128)
def _clipping_layout(mesh_key: tuple, placements: tuple, shape: tuple, dtype) -> Layout:
    """Cache metadata only; identity keeps equal-topology process groups distinct."""
    layout = Layout.from_device_mesh(to_hyper_mesh(mesh_key[1]))
    layout.set_placements(placements)
    layout.placement_to_tensor_map(len(shape))
    layout.set_tensor_meta(shape, torch.empty(shape, device="meta").stride(), dtype)
    return layout


def _clipping_view(local: torch.Tensor, mesh, placements: tuple, shape: tuple) -> DTensor:
    """Bind current storage without copying layout metadata or retaining gradients."""
    layout = _clipping_layout((id(mesh), mesh), placements, shape, local.dtype)
    return DTensor.from_local_with_layout(local, layout, shape=shape)


def _bind_clip_gradient(view: DTensor, parameter: torch.nn.Parameter) -> None:
    """Expose the current gradient, retaining FP32 main-gradient semantics."""
    main_gradient = getattr(parameter, "main_grad", None)
    gradient = main_gradient if main_gradient is not None else parameter.grad
    if gradient is None:
        return
    local = gradient.to_local() if isinstance(gradient, (DTensor, TorchDTensor)) else gradient
    if isinstance(parameter, TorchDTensor):
        gradient_view = DTensor.from_local(
            local, view.device_mesh, view.placements, shape=view.shape,
            stride=view.layout.tensor_stride, run_check=False,
        )
    else:
        gradient_view = _clipping_view(local, view.device_mesh, view.placements, tuple(view.shape))
    if main_gradient is not None:
        view.main_grad = gradient_view
    else:
        view.grad = gradient_view


def parameters_for_clipping(model: torch.nn.Module) -> list:
    """Describe every shard to Hyper clipping, sharing live gradient storage.

    VeOmni intentionally drops the EP axis from FSDP-managed expert parameters;
    its SpecInfo retains that axis. Restore the descriptor only for clipping.
    Persistent Torch DTensors remain model-owned and are exposed as local views.
    No parameter is replaced and no data collective is added by these views.
    """
    specifications = getattr(model, "_fqn2spec_info", {})
    if not specifications:
        parameters = list(model.parameters())
        if not any(isinstance(parameter, TorchDTensor) for parameter in parameters):
            return parameters
    result = []
    for name, parameter in model.named_parameters():
        specification = specifications.get(name)
        if isinstance(parameter, TorchDTensor):
            view = from_torch_view(parameter)
            view.requires_grad_(parameter.requires_grad)
        elif (
            isinstance(parameter, DTensor)
            and specification is not None
            and specification.persistent_fsdp_shard_dim is None
            and specification.placement.is_shard()
            and specification.para_fsdp_mesh is not None
        ):
            mesh = specification.para_fsdp_mesh
            placements = [Replicate() for _ in range(mesh.ndim)]
            placements[-2] = Shard(specification.fsdp_shard_dim)
            placements[-1] = Shard(specification.placement.dim)
            shape = list(parameter.shape)
            # SpecInfo.para_mesh slices a new Torch mesh on every access.
            # The restored descriptor already places EP on its final axis.
            shape[specification.placement.dim] *= mesh.size(-1)
            local_parameter = parameter.to_local().detach()
            local_parameter.requires_grad_(parameter.requires_grad)
            view = _clipping_view(
                local_parameter,
                mesh,
                tuple(placements),
                tuple(shape),
            )
        else:
            result.append(parameter)
            continue
        _bind_clip_gradient(view, parameter)
        result.append(view)
    return result
