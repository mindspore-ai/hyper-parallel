# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Parallel style abstractions for declarative module parallelism."""
from abc import ABC, abstractmethod
from typing import Dict

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module


class ParallelStyle(ABC):
    """Abstract base class for parallel styles applied to nn.Module submodules.

    Subclasses implement ``apply`` to wrap a module with the desired
    parallel communication behaviour (e.g. all-to-all for context parallel).
    """

    @abstractmethod
    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply this parallel style to *module* in-place and return it.

        Args:
            module: The submodule to be parallelised.
            device_mesh: The device mesh describing the cluster topology.

        Returns:
            The (possibly wrapped) module with parallelism applied.
        """


def _get_submodule(root: Module, path: str) -> Module:
    """Navigate *root* by the dot-separated *path* and return the submodule.

    Args:
        root: Top-level module.
        path: Dot-separated attribute path, e.g. ``"attn.core_attn"``.

    Returns:
        The submodule at *path*.

    Raises:
        AttributeError: If any segment of *path* does not exist on the module.
    """
    parts = path.split(".")
    current = root
    for part in parts:
        current = getattr(current, part)
    return current


def parallelize_module(
    module: Module,
    device_mesh: DeviceMesh,
    parallelize_plan: Dict[str, ParallelStyle],
) -> Module:
    """Apply declarative parallel styles to submodules of *module*.

    Each key in *parallelize_plan* is a dot-separated path to a submodule
    (e.g. ``"attn.core_attn"``), and each value is a :class:`ParallelStyle`
    instance that will be applied to that submodule.

    Args:
        module: The root module whose submodules will be parallelised.
        device_mesh: The device mesh to pass to each :class:`ParallelStyle`.
        parallelize_plan: Mapping from submodule path to :class:`ParallelStyle`.

    Returns:
        The root *module* with parallel styles applied to the specified
        submodules (modifications are done in-place via hooks).

    Example:
        >>> from hyper_parallel import DeviceMesh, ContextParallel
        >>> from hyper_parallel.parallel_style import parallelize_module
        >>> mesh = DeviceMesh("npu", (2, 4), mesh_dim_names=("dp", "cp"))
        >>> parallelize_module(model, mesh, {
        ...     "attn.core_attn": ContextParallel(seq_dim=1, head_dim=2),
        ... })
    """
    for submodule_path, style in parallelize_plan.items():
        submodule = _get_submodule(module, submodule_path)
        style.apply(submodule, device_mesh)
    return module
