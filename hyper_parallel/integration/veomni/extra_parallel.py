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
from types import MethodType, SimpleNamespace

import torch
from torch.distributed.tensor import DTensor as TorchDTensor

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.integration.torch.dtensor import to_torch_view


def _fsdp_clipping_view(model: torch.nn.Module) -> object:
    """Keep Torch's DTensor norm arithmetic without replacing live parameters."""
    views = []
    mesh_cache = {}
    detached_spec_cache = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        gradient = to_torch_view(gradient, mesh_cache=mesh_cache, detached_spec_cache=detached_spec_cache)
        if gradient is not None and isinstance(parameter, DTensor) and not isinstance(gradient, TorchDTensor):
            reference = to_torch_view(parameter, mesh_cache=mesh_cache)
            gradient = TorchDTensor.from_local(
                gradient.detach(), reference.device_mesh, reference.placements,
                run_check=False, shape=reference.shape, stride=reference.stride(),
            )
        views.append(SimpleNamespace(grad=gradient))
    return SimpleNamespace(parameters=lambda: iter(views))


def native_clipping_view(model: torch.nn.Module) -> object:
    """Expose live local gradients and ownership to VeOmni's native clipper.

    The proxy owns descriptors only: gradient scaling writes into the original
    storage, including FP32 main gradients. Plain FSDP retains Torch DTensor
    norm arithmetic. EP and CPU offload use VeOmni's local reductions, where
    missing or empty gradients contribute zero on every peer.
    """
    specifications = getattr(model, "_fqn2spec_info", {})
    cpu_offload = getattr(model, "_fsdp_cpu_offload_enabled", False)
    if not specifications and not cpu_offload:
        return _fsdp_clipping_view(model)
    groups = {"non_extra_parallel": []}
    replicated, persistent = set(), set()
    views = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = getattr(parameter, "main_grad", None)
        if gradient is None:
            gradient = parameter.grad
        if isinstance(gradient, (DTensor, TorchDTensor)):
            gradient = gradient.to_local()
        if gradient is None or gradient.numel() == 0:
            gradient = torch.zeros((), device=parameter.device, dtype=torch.float32)
        view = SimpleNamespace(grad=gradient.detach())
        views.append(view)
        specification = specifications.get(name)
        mesh = getattr(parameter, "device_mesh", None)
        dimensions = getattr(mesh, "mesh_dim_names", ()) or ()
        owner = next((dim[:-5] for dim in dimensions if dim.endswith("_fsdp")), "non_extra_parallel")
        if specification is not None and specification.persistent_fsdp_shard_dim is not None:
            owner = specification.para_name
            persistent.add(id(view))
        elif specification is not None and specification.placement.is_replicate():
            replicated.add(id(view))
        groups.setdefault(owner, []).append(view)
    if not specifications:
        return SimpleNamespace(parameters=lambda: iter(views), _fsdp_cpu_offload_enabled=cpu_offload)
    return SimpleNamespace(
        parameters=lambda: iter(views),
        _extra_parallel_param_groups=groups,
        _ep_replicated_lora_param_ids=replicated,
        _persistent_extra_parallel_param_ids=persistent,
    )


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
    # VeOmni's current NPU EP builder still calls this name, including on Torch 2.10.
    module.set_reduce_scatter_divide_factor = module.set_gradient_divide_factor
