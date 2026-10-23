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
"""Read-only model execution identity without native initialization or tensor copies."""

from __future__ import annotations

from collections import Counter
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _json_metadata(value: Any) -> Any:
    """Copy metadata while leaving tensor contents and runtime handles untouched."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _execution_modules(model: torch.nn.Module) -> dict[str, Any]:
    """Collect explicitly provided execution-state metadata from every named module."""
    result = {}
    for name, module in model.named_modules():
        provider = getattr(module, "execution_state", None)
        if callable(provider):
            result[name] = _json_metadata(provider())
    return result


def _loaded_native_libraries() -> list[str]:
    """Report already loaded Multicore libraries without loading or hashing them."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    paths = {line.split()[-1] for line in lines if line.split()}
    return sorted(path for path in paths if path.startswith("/") and ".so" in path
                  and any(name in path for name in ("multicore", "mega_moe", "shmem")))


def _independent_groups(modules: dict[str, Any]) -> list[dict[str, Any]]:
    """Flag potentially reusable groups without assuming that layers execute serially."""
    candidates: dict[str, dict[str, list[str]]] = {}
    for name, state in modules.items():
        if state.get("closed"):
            continue
        configuration = repr((state.get("type"), state.get("configuration"), state.get("ep_ranks")))
        groups = candidates.setdefault(configuration, {})
        groups.setdefault(str(state.get("resource_group")), []).append(name)
    return [{"code": "independent_compatible_groups", "modules": [name for names in groups.values() for name in names],
             "message": "Share execution resources before first forward only if these modules execute serially."}
            for groups in candidates.values() if len(groups) > 1]


def _parameter_metadata(model: torch.nn.Module) -> dict[str, Any]:
    """Describe parameter kinds without reading values or collecting local shards."""
    parameters = list(model.named_parameters())
    return {
        "types": dict(Counter(f"{type(value).__module__}.{type(value).__qualname__}" for _, value in parameters)),
        "dtypes": dict(Counter(str(value.dtype) for _, value in parameters)),
        "meta": [name for name, value in parameters if value.is_meta],
        "layouts": [_parameter_layout(name, value) for name, value in parameters],
    }


def _parameter_layout(name: str, value: torch.Tensor) -> dict[str, Any]:
    """Inspect public logical-layout metadata without converting a DTensor to a shard."""
    mesh = getattr(value, "device_mesh", None)
    placements = getattr(value, "placements", None)
    return {
        "name": name, "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "requires_grad": value.requires_grad,
        "mesh_shape": list(mesh.shape) if mesh is not None else None,
        "mesh_dims": list(mesh.mesh_dim_names) if mesh is not None and mesh.mesh_dim_names is not None else None,
        "placements": [str(placement) for placement in placements] if placements is not None else None,
    }


def inspect_execution(model: torch.nn.Module, *, optimizer: Any = None) -> dict[str, Any]:
    """Describe resolved execution metadata without allocating or communicating.

    This is an observation, not proof of initialization, numerical correctness or
    FSDP ownership. Tensor optimizer options are described by type, not copied.

    Args:
        model: Constructed model, including meta models and unbound executors.
        optimizer: Optional optimizer exposing its actual param_groups.

    Returns:
        JSON-compatible identity, parameter kinds, executor states and advisory findings.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    modules = _execution_modules(model)
    package = importlib.util.find_spec("hyper_parallel")
    result = {
        "schema_version": 1,
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "torch_version": str(torch.__version__), "hyper_parallel_source": package.origin if package else None,
        "native_libraries": _loaded_native_libraries(),
        "environment": {key: os.environ.get(key) for key in (
            "ASCEND_LAUNCH_BLOCKING", "CUDA_LAUNCH_BLOCKING", "HCCL_DETERMINISTIC",
            "FLASH_ATTENTION_DETERMINISTIC", "HYPER_PARALLEL_LIB_PATH")},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "parameters": _parameter_metadata(model),
        "execution_modules": modules,
        "resource_group_count": len({state["resource_group"] for state in modules.values()
                                     if "resource_group" in state and not state.get("closed")}),
        "findings": _independent_groups(modules),
    }
    if optimizer is not None:
        keys = ("fused", "foreach", "lr", "betas", "eps", "weight_decay", "capturable")
        result["optimizer"] = {
            "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "param_groups": [{key: _json_metadata(group.get(key)) for key in keys}
                             for group in optimizer.param_groups],
        }
    return result
