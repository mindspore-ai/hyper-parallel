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
"""Scoped vLLM import contracts for CPU adapter tests, without a runtime server."""

import importlib.util
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


class CompilationMode(IntEnum):
    """Only disabled compilation is used by the CPU adapter fixtures."""

    NONE = 0


def _backend_not_mocked(*_args: Any, **_kwargs: Any) -> Any:
    """Fail if a test reaches an external vLLM operation without its own double."""
    raise AssertionError("CPU adapter tests must mock the vLLM backend operation")


def _support_torch_compile(**_kwargs: Any) -> Any:
    """Keep the real adapter class when graph compilation is disabled for CPU tests."""
    return lambda cls: cls


def load_adapter_modules() -> tuple[ModuleType, ModuleType]:
    """Load real adapter code against external API doubles, then restore import state."""
    names = (
        "vllm", "vllm.compilation", "vllm.compilation.decorators", "vllm.config",
        "vllm.distributed", "vllm.model_executor", "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.weight_utils", "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
    )
    modules = {name: ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules["vllm.config"].VllmConfig = SimpleNamespace
    modules["vllm.compilation.decorators"].support_torch_compile = _support_torch_compile
    modules["vllm.distributed"].get_tp_group = _backend_not_mocked
    modules["vllm.model_executor.model_loader.weight_utils"].default_weight_loader = _backend_not_mocked
    modules["vllm.model_executor.layers.attention"].Attention = _backend_not_mocked

    package_name = "rl.roles.rollout.consistency_models.qwen3"
    source = Path(next(iter(importlib.util.find_spec(package_name).submodule_search_locations)))
    loaded = {}
    with pytest.MonkeyPatch.context() as patcher:
        for name, module in modules.items():
            patcher.setitem(sys.modules, name, module)
        for name in ("attention", "model"):
            # Transformers inspects the defining module; retain only test-owned aliases.
            alias = f"tests.ut.rl.rollout._qwen3_{name}_under_test"
            spec = importlib.util.spec_from_file_location(alias, source / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            patcher.setitem(sys.modules, f"{package_name}.{name}", module)
            spec.loader.exec_module(module)
            loaded[name] = module
    return loaded["model"], loaded["attention"]
