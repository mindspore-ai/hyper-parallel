# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""DTensor backend compatibility layer for the optimizer module.

Provides lazy exports (PEP 562) for DTensor, DeviceMesh, Shard, Replicate, 
and StridedShard based on the detected backend ('torch' or 'hyper').
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch.distributed._tensor as torch_dt

logger = logging.getLogger(__name__)

# Global backend flag
_DTENSOR_BACKEND: str = "hyper"  # "hyper" or "torch"


class _NeverMatch:
    """Safe fallback class that always returns False for ``isinstance()``."""
    __slots__ = ()


# Lazy-export cache
_LAZY_CACHE: Dict[str, Any] = {}


def _invalidate_lazy_cache() -> None:
    """Clear the lazy-export cache to rebuild on next access."""
    _LAZY_CACHE.clear()


def detect_dtensor_backend(
        adamw_params: List[Any],
        muon_params: List[Any],
) -> str:
    """Detect and set the DTensor backend ('torch' or 'hyper') from parameter lists."""
    global _DTENSOR_BACKEND  # pylint: disable=global-statement

    sample_param = _extract_first_param(muon_params)

    if sample_param is None:
        sample_param = _extract_first_param(adamw_params)

    if sample_param is None:
        logger.info_rank0("No parameters found for backend detection; defaulting to 'hyper'.")
        _DTENSOR_BACKEND = "hyper"
        _invalidate_lazy_cache()
        return _DTENSOR_BACKEND

    param_cls_module = type(sample_param).__module__
    if param_cls_module.startswith("torch.distributed"):
        _DTENSOR_BACKEND = "torch"
    else:
        _DTENSOR_BACKEND = "hyper"

    logger.info_rank0("Detected DTensor backend: '%s'.", _DTENSOR_BACKEND)
    _invalidate_lazy_cache()
    return _DTENSOR_BACKEND


def _extract_first_param(param_groups: List[Any]) -> Any:
    """Return the first parameter from a list of param groups, or None."""
    for group in param_groups:
        params = group.get("params", []) if isinstance(group, dict) else []
        for p in params:
            return p

    for p in param_groups:
        return p

    return None


# Accessor functions
def get_dtensor_cls():
    """Return the DTensor class for the active backend."""
    if _DTENSOR_BACKEND == "torch":
        return torch_dt.DTensor
    from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel
    return DTensor


def get_device_mesh_cls():
    """Return the DeviceMesh class for the active backend."""
    if _DTENSOR_BACKEND == "torch":
        from torch.distributed.device_mesh import DeviceMesh  # pylint: disable=import-outside-toplevel
        return DeviceMesh
    from hyper_parallel.core.dtensor.device_mesh import DeviceMesh  # pylint: disable=import-outside-toplevel
    return DeviceMesh


def get_shard_cls():
    """Return the Shard placement class for the active backend."""
    if _DTENSOR_BACKEND == "torch":
        from torch.distributed._tensor.placement_types import Shard  # pylint: disable=import-outside-toplevel
        return Shard
    from hyper_parallel.core.dtensor.placement_types import Shard  # pylint: disable=import-outside-toplevel
    return Shard


def get_replicate_cls():
    """Return the Replicate placement class for the active backend."""
    if _DTENSOR_BACKEND == "torch":
        from torch.distributed._tensor.placement_types import Replicate  # pylint: disable=import-outside-toplevel
        return Replicate
    from hyper_parallel.core.dtensor.placement_types import Replicate  # pylint: disable=import-outside-toplevel
    return Replicate


def get_strided_shard_cls():
    """Return the StridedShard placement class. Returns _NEVER_MATCH for 'torch'."""
    if _DTENSOR_BACKEND == "torch":
        return _NeverMatch

    from hyper_parallel.core.dtensor.placement_types import StridedShard  # pylint: disable=import-outside-toplevel
    return StridedShard


# DTensor union type resolver
def _import_hyper_dtensor():
    """Import hyper DTensor class; return torch DTensor as fallback."""
    try:
        from hyper_parallel.core.dtensor.dtensor import DTensor  # pylint: disable=import-outside-toplevel
        return DTensor
    except ImportError:
        return torch_dt.DTensor


def _resolve_dtensor_union():
    """Build ``torch_dt.DTensor | hyper_dt.DTensor`` on demand."""
    return torch_dt.DTensor | _import_hyper_dtensor()


def to_local_if_dtensor(tensor: Any) -> Any:
    """Return the local shard if `tensor` is a DTensor, otherwise return as-is."""
    # Use resolver directly for internal module lookups instead of lazy-loaded DTensor
    dtensor_type = _LAZY_CACHE.get("DTensor") or _resolve_dtensor_union()
    return tensor.to_local() if isinstance(tensor, dtensor_type) else tensor


# lazy exports
_LAZY_RESOLVERS = {
    "DTensor": _resolve_dtensor_union,
    "DeviceMesh": get_device_mesh_cls,
    "Shard": get_shard_cls,
    "Replicate": get_replicate_cls,
    "StridedShard": get_strided_shard_cls,
}


def __getattr__(name):  # type: ignore[no-untyped-def]  # pylint: disable=invalid-name
    """Resolve module attributes on first access."""
    resolver = _LAZY_RESOLVERS.get(name)
    if resolver is not None:
        value = _LAZY_CACHE.get(name)
        if value is None:
            value = resolver()
            _LAZY_CACHE[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # type: ignore[no-untyped-def]  # pylint: disable=invalid-name
    """Include lazy-exported names in dir() for IDE autocomplete."""
    return list(globals().keys()) + list(_LAZY_RESOLVERS.keys())
