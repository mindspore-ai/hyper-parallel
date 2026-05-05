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
"""High-level API to apply parallel styles to modules (aligned with PyTorch ``parallelize_module``)."""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from fnmatch import fnmatch
from typing import Iterator, Optional, Union

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _mesh_resources
from hyper_parallel.core.tensor_parallel.style import ParallelStyle
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module

__all__ = ["parallelize_module"]


def _named_children(module: Module):
    """Immediate child modules: PyTorch ``nn.Module.named_children`` or MindSpore ``Cell.name_cells``."""
    if hasattr(module, "named_children"):
        return module.named_children()
    return module.name_cells().items()


@contextmanager
def _tensor_parallel_mesh_context(device_mesh: DeviceMesh) -> Iterator[DeviceMesh]:
    """Internal: same thread-local stack as ``with device_mesh:`` for ``parallelize_module(..., None)``.

    Prefer user code using ``with mesh:``; this exists for tests and library helpers.
    """
    with device_mesh:
        yield device_mesh


def _validate_tp_mesh_dim(device_mesh: DeviceMesh) -> None:
    """Require a 1-D mesh, matching PyTorch tensor-parallel constraints."""
    if device_mesh.ndim > 1:
        raise ValueError(
            f"Tensor Parallel only accepts a 1D DeviceMesh, but found {device_mesh.ndim}D! "
            f'If you have a 2-D or N-D device_mesh, consider passing in device_mesh["tp"] '
            f'or another 1-D sub-mesh slice (e.g. mesh["cp"]).'
        )


def parallelize_module(  # type: ignore[return]
    module: Module,
    device_mesh: Optional[DeviceMesh] = None,
    parallelize_plan: Optional[Union[ParallelStyle, dict[str, ParallelStyle]]] = None,
    *,
    src_data_rank: Optional[int] = 0,
) -> Module:
    """Apply parallel styles to *module* or submodules (PyTorch-compatible interface).

    Behaviour follows ``torch.distributed.tensor.parallel.parallelize_module``:

    - *device_mesh* should normally be passed explicitly. Omitting it (``None``) requires an
      active mesh context: ``with mesh:`` (see :meth:`hyper_parallel.core.dtensor.device_mesh.DeviceMesh.__enter__`)
      or :func:`_tensor_parallel_mesh_context` for tests/libraries.
    - *parallelize_plan* may be a single :class:`ParallelStyle` (applied to *module*)
      or a dict mapping submodule paths to styles. Path segments support ``fnmatch``
      patterns (e.g. ``\"layers.*\"``) like PyTorch FQN rules.
    - Only **1-D** :class:`DeviceMesh` is accepted; slice a sub-mesh from a multi-dim mesh first.
    - *src_data_rank* is stored on the style (``style.src_data_rank``) before ``apply``; styles
      that shard parameters from a logical global tensor may use it (see PyTorch TP).

    Note:
        When ``parallelize_plan`` is a single :class:`ParallelStyle` (not a dict), this
        function modifies it in-place by setting ``parallelize_plan.src_data_rank``.
        The caller should be aware that the passed object will be mutated.

    Args:
        module: Root module to parallelize.
        device_mesh: Mesh for this TP/CP slice. Use ``None`` only inside ``with mesh:`` (or
            :func:`_tensor_parallel_mesh_context`) so ``_mesh_resources.get_current_mesh()`` resolves
            (see PyTorch ``distribute_module``).
        parallelize_plan: A :class:`ParallelStyle` or dict ``{path: ParallelStyle}``.
        src_data_rank: Source rank for global tensor semantics; ``None`` means use local data only
            (PyTorch parity). Default ``0``.

    Returns:
        *module* after in-place parallelization.
    """
    if device_mesh is None:
        device_mesh = _mesh_resources.get_current_mesh()
    _validate_tp_mesh_dim(device_mesh)

    if parallelize_plan is None:
        warnings.warn(
            "No parallelize_plan is provided and auto-parallel is not supported "
            "at the moment, so this parallelize_module call will do nothing.",
            stacklevel=2,
        )
        return module

    if isinstance(parallelize_plan, ParallelStyle):
        parallelize_plan.src_data_rank = src_data_rank
        return parallelize_plan.apply(module, device_mesh)
    if isinstance(parallelize_plan, dict):

        def _apply_path(
            current_module: Module,
            atoms: list[str],
            style: ParallelStyle,
            src_rank: Optional[int],
        ) -> bool:
            atom = atoms[0]
            matched_children = list(
                filter(
                    lambda t, pattern=atom: fnmatch(t[0], pattern),
                    _named_children(current_module),
                )
            )
            applied = False
            for _, submodule in matched_children:
                if len(atoms) == 1:
                    parallelize_module(
                        submodule,
                        device_mesh,
                        style,
                        src_data_rank=src_rank,
                    )
                    applied = True
                else:
                    applied = _apply_path(submodule, atoms[1:], style, src_rank) or applied
            return applied

        for module_path, parallelize_style in parallelize_plan.items():
            if not isinstance(parallelize_style, ParallelStyle):
                raise TypeError(
                    "Expect ParallelStyle values in parallelize_plan dict, but got "
                    f"{type(parallelize_style)} for path '{module_path}'."
                )
            path_splits = module_path.split(".")
            if module_path == "" or any(path == "" for path in path_splits):
                raise ValueError(
                    f"Expect module path to be non-empty dot-separated atoms, but got '{module_path}'."
                )
            if not _apply_path(module, path_splits, parallelize_style, src_data_rank):
                warnings.warn(
                    f"parallelize_plan path '{module_path}' has no matches, so this path is skipped.",
                    stacklevel=2,
                )
        return module
    raise TypeError(
        "Expect Union[ParallelStyle, Dict[str, ParallelStyle]] for"
        f" parallelize_plan, {type(parallelize_plan)} found!"
    )
