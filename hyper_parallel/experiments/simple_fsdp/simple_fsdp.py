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
"""SimpleFSDP: compiler-friendly FSDP via autograd all-gather.

This module implements a small experimental SimpleFSDP path. It shards every
parameter across a data-parallel process group and replaces parameter access
with an autograd-aware all-gather. When the model is compiled with
``torch.compile(model, fullgraph=True)``, Dynamo can see the parameter
all-gather in the graph. The paired backward reduce-scatter is provided by
``AllGatherRSFunction``.

This is intentionally narrower than HyperParallel's production ``fully_shard``:
it is a spike for validating the TorchTitan-style graph-training direction
before DTensor ``redistribute`` grows full autograd semantics.
"""
import sys
from typing import Any

import torch
import torch.distributed as dist  # pylint: disable=C0415
import torch.nn as nn

from hyper_parallel.experiments.simple_fsdp.autograd_comm import all_gather_rs

_wrap_class_cache: dict[tuple[type, tuple[tuple[str, int], ...], int, int], type] = {}


def _get_process_group(mesh: Any) -> Any:
    """Extract a PyTorch process group from a mesh-like object."""
    if hasattr(mesh, "get_group"):
        return mesh.get_group()
    if hasattr(mesh, "_process_group"):
        return mesh._process_group
    return mesh


def _get_group_rank_and_world_size(mesh: Any) -> tuple[Any, int, int]:
    """Return ``(group, group_local_rank, group_world_size)`` for ``mesh``.

    Args:
        mesh: HyperParallel ``DeviceMesh``, PyTorch ``DeviceMesh``, raw
            process group, or ``None`` for the default process group.

    Returns:
        Tuple containing process group, local rank inside that group, and
        group world size.

    Raises:
        RuntimeError: If torch distributed has not been initialized.
    """
    if not dist.is_initialized():
        raise RuntimeError("simple_fsdp requires torch.distributed to be initialized.")
    group = _get_process_group(mesh)
    return group, dist.get_rank(group), dist.get_world_size(group)


def _normalize_shard_dim(param_name: str, param: nn.Parameter, shard_dim: int, world_size: int) -> int:
    """Validate and normalize the sharding dimension for one parameter."""
    if param.is_meta:
        raise ValueError(
            f"simple_fsdp does not support meta parameters yet; got meta parameter {param_name!r}."
        )
    if param.ndim == 0:
        raise ValueError(
            f"simple_fsdp cannot shard scalar parameter {param_name!r}; "
            "keep it replicated or exclude it before wrapping."
        )

    normalized_dim = shard_dim if shard_dim >= 0 else param.ndim + shard_dim
    if normalized_dim < 0 or normalized_dim >= param.ndim:
        raise ValueError(
            f"shard_dim={shard_dim} is out of range for parameter {param_name!r} "
            f"with shape {tuple(param.shape)}."
        )
    if param.shape[normalized_dim] % world_size != 0:
        raise ValueError(
            f"Parameter {param_name!r} shape {tuple(param.shape)} is not evenly shardable "
            f"on dim {normalized_dim} across world_size={world_size}."
        )
    return normalized_dim


def _shard_parameter(
    param_name: str,
    param: nn.Parameter,
    local_rank: int,
    world_size: int,
    shard_dim: int,
) -> tuple[nn.Parameter, int]:
    """Create the local parameter shard and return its normalized shard dim."""
    normalized_dim = _normalize_shard_dim(param_name, param, shard_dim, world_size)
    shard_size = param.shape[normalized_dim] // world_size
    shard_start = local_rank * shard_size
    shard = param.detach().narrow(normalized_dim, shard_start, shard_size).clone().detach()
    return nn.Parameter(shard, requires_grad=param.requires_grad), normalized_dim


def _register_fsdp_parametrization(
    module: nn.Module,
    param_shard_dims: dict[str, int],
    group: Any,
    world_size: int,
) -> None:
    """Replace parameter access with autograd all-gather via a class swap."""
    param_name_to_property = {}
    for param_name, param_shard_dim in param_shard_dims.items():

        def _make_property(p_name: str, process_group: Any, group_size: int, dim: int):
            @property
            def prop(self):
                shard = self._parameters[p_name]
                if shard is None or shard.numel() == 0:
                    return shard
                return all_gather_rs(shard, process_group, group_size, dim)

            return prop

        param_name_to_property[param_name] = _make_property(
            param_name, group, world_size, param_shard_dim
        )

    cache_key = (
        module.__class__,
        tuple(sorted(param_shard_dims.items())),
        id(group),
        world_size,
    )
    if cache_key in _wrap_class_cache:
        wrapper_cls = _wrap_class_cache[cache_key]
    else:
        wrapper_cls = type(
            f"SimpleFSDP{module.__class__.__name__}{len(_wrap_class_cache)}",
            (module.__class__,),
            param_name_to_property,
        )
        sys.modules[wrapper_cls.__module__].__dict__[wrapper_cls.__name__] = wrapper_cls
        _wrap_class_cache[cache_key] = wrapper_cls
    module.__class__ = wrapper_cls


def simple_fsdp(
    model: nn.Module,
    mesh: Any,
    shard_dim: int = 0,
) -> nn.Module:
    """Apply experimental SimpleFSDP to ``model``.

    For every direct parameter in every submodule:

    1. Slice the parameter along ``shard_dim`` using the local rank in the
       data-parallel process group.
    2. Replace the parameter with the local shard.
    3. Swap the module class so reading the parameter all-gathers the full
       tensor, with backward reduce-scattering gradients back to the shard.

    Args:
        model: The model to wrap.
        mesh: Device mesh or process group defining the data-parallel ranks.
        shard_dim: Dimension along which to shard each parameter.

    Returns:
        The same model object with sharded parameters and class-swapped modules.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized.
        ValueError: If any parameter cannot be evenly sharded on ``shard_dim``.
    """
    group, local_rank, world_size = _get_group_rank_and_world_size(mesh)

    for module in list(model.modules()):
        if "SimpleFSDP" in module.__class__.__name__:
            continue

        params_dict = dict(module.named_parameters(recurse=False))
        if not params_dict:
            continue

        param_shard_dims = {}
        for param_name, param in params_dict.items():
            if param is None or param.numel() == 0:
                continue
            shard, normalized_dim = _shard_parameter(
                param_name, param, local_rank, world_size, shard_dim
            )
            module.register_parameter(param_name, shard)
            param_shard_dims[param_name] = normalized_dim

        if param_shard_dims:
            _register_fsdp_parametrization(module, param_shard_dims, group, world_size)

    return model
