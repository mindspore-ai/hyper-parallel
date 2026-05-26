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
"""Bind PyTorch platform and reload ``hyper_parallel.dmodule`` (M1 is torch-only)."""

from __future__ import annotations

import importlib
import os
import sys

_DMODULE_RELOAD_ORDER: tuple[str, ...] = (
    "hyper_parallel.dmodule.module",
    "hyper_parallel.dmodule.model",
    "hyper_parallel.dmodule.model_spec",
    "hyper_parallel.dmodule",
)


def _dmodule_module_needs_reload() -> bool:
    """True if ``dmodule.Module`` was imported while MindSpore was the active platform."""
    mod_name = "hyper_parallel.dmodule.module"
    if mod_name not in sys.modules:
        return False
    dmodule_module = sys.modules[mod_name]
    return not callable(getattr(dmodule_module.Module, "children", None))


def ensure_torch_platform_for_dmodule() -> None:
    """Point ``dmodule.Module`` at ``torch.nn.Module`` after MindSpore UTs switch platform.

    ``hyper_parallel.dmodule.module`` captures ``platform.Module`` at import time.
    MindSpore tests collected earlier leave a cached MindSpore platform and a stale
    ``Module`` base missing ``children()``; reload dmodule submodules only when needed.
    """
    os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
    import hyper_parallel.platform.platform as _platform_mod  # pylint: disable=import-outside-toplevel

    _platform_mod.platform = None
    if not _dmodule_module_needs_reload():
        return
    for name in _DMODULE_RELOAD_ORDER:
        if name not in sys.modules:
            continue
        importlib.reload(sys.modules[name])
