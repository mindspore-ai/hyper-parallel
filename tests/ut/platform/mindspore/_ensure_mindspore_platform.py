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
"""Reset cached :func:`get_platform` for MindSpore UTs (import before ``hyper_parallel``)."""
from __future__ import annotations

import importlib
import os
from typing import Collection


def _force_cpu_device_target() -> None:
    """Force MindSpore to CPU device so UTs never touch Ascend runtime.

    Tensor ops such as ``narrow`` / ``copy_`` / ``contiguous`` / ``add_`` /
    ``asnumpy`` initialize the configured device backend. Default
    ``device_target`` on Ascend-equipped hosts is ``"Ascend"``, which calls
    ``rtGetDeviceCount`` / ``Get soc name`` and fails on CPU-only CI runners.
    Pinning the context to CPU keeps the UTs hardware-agnostic.
    """
    import mindspore as ms  # pylint: disable=import-outside-toplevel

    ms.set_context(device_target="CPU")


def _reset_platform_singleton() -> None:
    os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
    import hyper_parallel.platform.platform as _platform_mod  # pylint: disable=import-outside-toplevel

    _platform_mod.platform = None
    _force_cpu_device_target()


def _bind_platform_globals(module_names: Collection[str]) -> None:
    """Reassign module-level ``platform`` (and common tensor aliases) without ``reload``."""
    from hyper_parallel.platform import get_platform  # pylint: disable=import-outside-toplevel

    plat = get_platform()
    for name in module_names:
        mod = importlib.import_module(name)
        if hasattr(mod, "platform"):
            mod.platform = plat
        if hasattr(mod, "Tensor"):
            mod.Tensor = plat.Tensor
        if hasattr(mod, "Module"):
            mod.Module = plat.Module


# Submodules that do ``platform = get_platform()`` at import time; must stay in sync after platform switches.
_FULLY_SHARD_PLATFORM_BIND_MODULES: tuple[str, ...] = (
    "hyper_parallel.core.dtensor.device_mesh",
    "hyper_parallel.core.fully_shard.hsdp_scheduler",
    "hyper_parallel.core.fully_shard.hsdp_utils",
    "hyper_parallel.core.fully_shard.hsdp_state",
    "hyper_parallel.core.fully_shard.hsdp_param",
    "hyper_parallel.core.fully_shard.utils",
    "hyper_parallel.core.fully_shard.api",
)


def ensure_mindspore_platform_for_fully_shard() -> None:
    """Point fully_shard / scheduler / device_mesh at MindSpore without reloading (keeps Torch UTs safe)."""
    _reset_platform_singleton()
    _bind_platform_globals(_FULLY_SHARD_PLATFORM_BIND_MODULES)


def ensure_mindspore_platform_for_device_mesh() -> None:
    """Rebind :mod:`hyper_parallel.core.dtensor.device_mesh` ``Tensor`` to MindSpore."""
    _reset_platform_singleton()
    _bind_platform_globals(("hyper_parallel.core.dtensor.device_mesh",))


def ensure_mindspore_platform_for_context_parallel() -> None:
    """Rebind async context parallel platform aliases."""
    _reset_platform_singleton()
    _bind_platform_globals(("hyper_parallel.core.context_parallel.async_context_parallel",))


def ensure_mindspore_platform_default() -> None:
    """Clear singleton only; next :func:`get_platform` uses ``HYPER_PARALLEL_PLATFORM``."""
    _reset_platform_singleton()


def ensure_mindspore_platform_for_shard_and_dtensor() -> None:
    """Rebind DTensor / layout / ``custom_shard`` platform handles for MindSpore."""
    _reset_platform_singleton()
    _bind_platform_globals(
        (
            "hyper_parallel.core.dtensor.device_mesh",
            "hyper_parallel.core.dtensor.layout",
            "hyper_parallel.core.dtensor.dtensor",
            "hyper_parallel.core.shard.custom_shard",
        )
    )


def restore_torch_platform_for_ut() -> None:
    """Restore PyTorch-backed module globals after MindSpore-only tests (best-effort)."""
    os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
    import hyper_parallel.platform.platform as _platform_mod  # pylint: disable=import-outside-toplevel

    _platform_mod.platform = None
    _bind_platform_globals(
        (
            "hyper_parallel.core.dtensor.device_mesh",
            "hyper_parallel.core.dtensor.layout",
            "hyper_parallel.core.dtensor.dtensor",
            "hyper_parallel.core.shard.custom_shard",
            "hyper_parallel.core.expert_parallel.expert_parallel",
            *_FULLY_SHARD_PLATFORM_BIND_MODULES,
            "hyper_parallel.core.context_parallel.async_context_parallel",
        )
    )
    try:
        from tests.ut.dmodule._ensure_torch_dmodule import (  # pylint: disable=import-outside-toplevel
            ensure_torch_platform_for_dmodule,
        )

        ensure_torch_platform_for_dmodule()
    except ImportError:
        pass
