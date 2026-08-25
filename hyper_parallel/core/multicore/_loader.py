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
"""Locate and load the component-owned multicore payload."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

_VENDOR_NAME = "hyper_parallel_multicore_nn"


class NativeComponentUnavailableError(RuntimeError):
    """Report an unavailable or incorrectly activated multicore component."""


def require_multicore_environment() -> Path:
    """Require the packaged custom OPP environment prepared by ``set_env.bash``."""
    component_root = _component_root()
    vendor_root = component_root / "vendors" / _VENDOR_NAME
    if not vendor_root.is_dir():
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-PAYLOAD-MISSING] component=multicore vendor={vendor_root}. "
            "Inspect the current build log; for source/PYTHONPATH development, run "
            "./build.sh --multicore all."
        )
    vendor_root = vendor_root.resolve()
    op_api_root = (vendor_root / "op_api" / "lib").resolve()
    missing_variables = []
    if not _environment_contains_path("ASCEND_CUSTOM_OPP_PATH", vendor_root):
        missing_variables.append("ASCEND_CUSTOM_OPP_PATH")
    if not _environment_contains_path("LD_LIBRARY_PATH", op_api_root):
        missing_variables.append("LD_LIBRARY_PATH")
    if missing_variables:
        set_env_script = component_root / "set_env.bash"
        missing_text = ",".join(missing_variables)
        if any(module in sys.modules for module in ("mindspore", "torch", "torch_npu")):
            raise NativeComponentUnavailableError(
                "[HP-NATIVE-OPP-ACTIVATION-TOO-LATE] component=multicore "
                f"missing={missing_text}. MindSpore/Torch/torch_npu has already been imported; "
                f"exit the current Python process, run source {set_env_script}, and start a new process."
            )
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-OPP-NOT-ACTIVATED] component=multicore "
            f"missing={missing_text}. Run source {set_env_script} before starting Python with multicore."
        )
    return vendor_root.resolve()


def get_multicore_paths(framework: str) -> tuple[Path, Path]:
    """Return the unified vendor root and selected framework adapter."""
    if framework not in {"mindspore", "torch"}:
        raise ValueError(f"Unsupported multicore framework: {framework!r}.")
    vendor_root = require_multicore_environment()
    framework_root = _component_root() / "framework" / framework
    candidates = (
        [framework_root / "hyper_parallel_mega_moe_ms.so"]
        if framework == "mindspore"
        else sorted(framework_root.glob("hyper_parallel_mega_moe_pta*.so"))
    )
    vendor_library = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    if not vendor_library.is_file() or len(candidates) != 1 or not candidates[0].is_file():
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-FRAMEWORK-TARGET-UNAVAILABLE] component=multicore framework={framework} "
            f"root={_component_root()}. The current wheel/PYTHONPATH payload does not include an adapter for this "
            "framework; rebuild with the corresponding --multicore target and inspect the build log."
        )
    return vendor_root, candidates[0].resolve()


def preload_vendor_library(vendor_root: Path) -> None:
    """Load the exact component-owned ACLNN library without global SONAME search."""
    library_path = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    try:
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-VENDOR-LOAD-FAILED] library={library_path} error={error}."
        ) from error


def load_cpython_extension(module_name: str, adapter_path: Path) -> ModuleType:
    """Load one ABI-specific CPython extension from the component payload."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise NativeComponentUnavailableError(f"Cannot create a module spec for {adapter_path}.")
    module = None
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError) as error:
        if module is not None and sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-ADAPTER-LOAD-FAILED] component=multicore "
            f"module={module_name} library={adapter_path} error={error}. "
            "Check the Python/framework/CANN version combination and rebuild the matching adapter."
        ) from error
    except Exception:
        if module is not None and sys.modules.get(module_name) is module:
            sys.modules.pop(module_name, None)
        raise
    return module


def _component_root() -> Path:
    """Return the installed or source-build multicore payload root."""
    packaged = Path(__file__).resolve().parent / "lib"
    if packaged.is_dir():
        return packaged
    repository_root = Path(__file__).resolve().parents[3]
    if (repository_root / "setup.py").is_file():
        source_build = (
            repository_root / "build" / "native" / "payload" / "hyper_parallel"
            / "core" / "multicore" / "lib"
        )
        if source_build.is_dir():
            return source_build
    return packaged


def _environment_contains_path(variable: str, expected: Path) -> bool:
    """Return whether an environment path list contains ``expected``."""
    configured = [value for value in os.environ.get(variable, "").split(os.pathsep) if value]
    return str(expected.resolve()) in {str(Path(value).resolve()) for value in configured}
