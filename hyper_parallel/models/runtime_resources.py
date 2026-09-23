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
"""Explicit model-owned runtime resources, independent of Trainer backends."""

from __future__ import annotations

from torch import nn  # pylint: disable=forbidden-backend-import


class ModelRuntimeModule:
    """Opt a module into preparation and deterministic resource cleanup.

    Implementations group compatible resources themselves. Hooks must not
    execute model forwards or change registered parameters. Unprepared and
    already closed modules must tolerate cleanup.
    """

    @staticmethod
    def prepare_runtime_group(modules: list[ModelRuntimeModule]) -> None:
        """Prepare same-type modules before their first forward."""
        raise NotImplementedError

    def close_runtime(self) -> None:
        """Release resources after backward and before process-group teardown."""
        raise NotImplementedError


class ModelRuntimeResources:
    """Own one model's explicitly participating modules for one training run."""

    def __init__(self, model: nn.Module) -> None:
        """Collect unique modules without importing any optional backend.

        Args:
            model: Final model after replacement, parallelization and materialization.
        """
        self._modules = [module for module in model.modules() if isinstance(module, ModelRuntimeModule)]
        self._prepared = False
        self._closed = False

    def prepare(self) -> None:
        """Prepare groups once; roll back all participants if preparation fails."""
        if self._closed:
            raise RuntimeError("Cannot prepare closed model runtime resources")
        if self._prepared:
            return
        groups = {}
        for module in self._modules:
            groups.setdefault(type(module), []).append(module)
        try:
            for module_type, modules in groups.items():
                module_type.prepare_runtime_group(modules)
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise RuntimeError(f"Runtime preparation and cleanup failed: {cleanup_error}") from error
            raise
        self._prepared = True

    def close(self) -> None:
        """Close every participant, retaining failures for an explicit retry."""
        if self._closed:
            return
        errors = []
        pending = []
        for module in reversed(self._modules):
            try:
                module.close_runtime()
            except Exception as error:
                errors.append(error)
                pending.append(module)
        self._modules = list(reversed(pending))
        self._closed = not pending
        if errors:
            raise RuntimeError(f"Failed to close {len(errors)} model runtime resource(s): {errors}") from errors[0]
