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
"""Process-local compatibility patches used only by the dry-run tool."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch._subclasses.fake_tensor import unset_fake_temporarily
from transformers.modeling_utils import ContextManagers

from hyper_parallel import init_empty_weights


_DEFER_MODEL_MATERIALIZATION: ContextVar[bool] = ContextVar(
    "dryrun_defer_model_materialization",
    default=False,
)
_MUON_LAYOUT_STAGE_STATE: ContextVar[Any] = ContextVar(
    "dryrun_muon_layout_stage_state",
    default=None,
)
_INSTALLED = False
# Model-specific environment patches, registered by example/plugin modules at
# import time (same contract as value-dependency handlers). The generic
# compatibility layer never imports model-specific code; the dependency
# direction is plugins -> dry-run adapters, not the reverse.
_ENVIRONMENT_PATCHES: dict[str, Any] = {}
_ENVIRONMENT_PATCHES_INSTALLED: set[str] = set()


def register_environment_patch(name: str, installer: Any) -> None:
    """Register a model-specific process-local patch for dry-run environments.

    Args:
        name: Unique patch identity, e.g. ``"deepseek_v41.csa2"``.
        installer: Zero-argument callable applying the patch idempotently.
    """
    if not callable(installer):
        raise TypeError("environment patch installer must be callable")
    _ENVIRONMENT_PATCHES[name] = installer
    # A patch registered after install_dryrun_compatibility (import order is
    # the plugin's choice) still has to take effect for this process.
    if _INSTALLED and name not in _ENVIRONMENT_PATCHES_INSTALLED:
        installer()
        _ENVIRONMENT_PATCHES_INSTALLED.add(name)


@contextmanager
def model_build_context() -> Iterator[None]:
    """Build a model without materializing weights or reading device storage."""
    token = _DEFER_MODEL_MATERIALIZATION.set(True)
    try:
        yield
    finally:
        _DEFER_MODEL_MATERIALIZATION.reset(token)


def is_model_materialization_deferred() -> bool:
    """Return whether the current build is owned by dry-run."""
    return _DEFER_MODEL_MATERIALIZATION.get()


def _install_model_build_patches() -> None:
    """Keep shape-only construction out of AutoModel business branches."""
    from hyper_parallel.models._transformers import auto_model, model_builder

    if getattr(model_builder, "_dryrun_compatibility_installed", False):
        return

    original_materialize = model_builder._materialize_and_load_model
    original_init_context = auto_model._model_init_context

    def materialize_without_storage(model: nn.Module, **kwargs: Any) -> nn.Module:
        if is_model_materialization_deferred():
            return model
        return original_materialize(model, **kwargs)

    def model_init_context(is_meta_device: bool) -> Any:
        if not is_model_materialization_deferred():
            return original_init_context(is_meta_device)

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            from transformers.initialization import no_init_weights

        contexts = [unset_fake_temporarily(), no_init_weights(), init_empty_weights()]
        return ContextManagers(contexts)

    model_builder._materialize_and_load_model = materialize_without_storage
    model_builder.model_build_context = model_build_context
    model_builder.is_model_materialization_deferred = is_model_materialization_deferred
    auto_model._model_init_context = model_init_context
    model_builder._dryrun_compatibility_installed = True


def _install_fake_subgroup_patch() -> None:
    """Create optimizer replica subgroups directly on PyTorch's fake backend."""
    from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer

    if getattr(BaseDistributedOptimizer, "_dryrun_subgroup_patch_installed", False):
        return
    original = BaseDistributedOptimizer._get_or_create_sub_pg

    def get_or_create_sub_pg(
            self: Any,
            parent_pg: dist.ProcessGroup,
            sub_size: int,
    ) -> dist.ProcessGroup:
        if dist.get_backend(parent_pg) != "fake":
            return original(self, parent_pg, sub_size)
        parent_ranks = tuple(sorted(dist.get_process_group_ranks(parent_pg)))
        cache_key = (parent_ranks, sub_size)
        cached = self._split_sub_pg_cache.get(cache_key)
        if cached is not None:
            for subgroup in cached.values():
                if subgroup is not None:
                    return subgroup

        rank = dist.get_rank()
        subgroups = {}
        local_group = None
        for index in range(len(parent_ranks) // sub_size):
            ranks = list(parent_ranks[index * sub_size:(index + 1) * sub_size])
            if rank not in ranks:
                subgroups[index] = None
                continue
            local_group = dist.new_group(ranks=ranks, use_local_synchronization=True)
            subgroups[index] = local_group
        self._split_sub_pg_cache[cache_key] = subgroups
        if local_group is None:
            raise RuntimeError(f"Rank {rank} is absent from optimizer subgroup {parent_ranks}")
        return local_group

    BaseDistributedOptimizer._get_or_create_sub_pg = get_or_create_sub_pg
    BaseDistributedOptimizer._dryrun_subgroup_patch_installed = True


def _infer_gather_rank_shapes(
        layout: Any,
        global_shape: tuple[int, ...],
        coordinate: tuple[int, ...],
        shard_axes: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Derive successive all-gather shapes from DTensor slice geometry."""
    from itertools import product

    from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout

    shapes = {}
    for rank, mesh_coordinate in enumerate(
            product(*(range(size) for size in layout.mesh_shape))
    ):
        area = infer_slice_area_by_layout(layout, rank, global_shape)
        shapes[mesh_coordinate] = tuple(end - start for start, end in area)
    result = []
    for mesh_dim, tensor_dim in shard_axes:
        group_size = layout.mesh_shape[mesh_dim]
        peers = [
            coordinate[:mesh_dim] + (rank,) + coordinate[mesh_dim + 1:]
            for rank in range(group_size)
        ]
        result.append(tuple(shapes[peer] for peer in peers))
        next_shapes = {}
        for key, shape in shapes.items():
            group_peers = [
                key[:mesh_dim] + (rank,) + key[mesh_dim + 1:]
                for rank in range(group_size)
            ]
            output = list(shape)
            output[tensor_dim] = sum(shapes[peer][tensor_dim] for peer in group_peers)
            next_shapes[key] = tuple(output)
        shapes = next_shapes
    return tuple(result)


def _install_muon_metadata_patch() -> None:
    """Supply FakeTensor Muon shard metadata without changing optimizer code."""
    from hyper_parallel.core.dtensor.layout import Layout
    from hyper_parallel.core.optimizer import muon
    from hyper_parallel.core.optimizer import muon_shard
    from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor

    if getattr(muon_shard, "_dryrun_metadata_patch_installed", False):
        return
    original_build = muon_shard.build_param_shard_metadata_for_group
    original_all_gather = muon_shard._allgather_shapes_for_group

    def all_gather_shapes_for_group(
            local_shapes: list[tuple[int, ...]],
            shard_pg: Any,
            shard_size: int,
            device: torch.device,
    ) -> list[list[tuple[int, ...]]]:
        state = _MUON_LAYOUT_STAGE_STATE.get()
        if state is None:
            return original_all_gather(local_shapes, shard_pg, shard_size, device)
        layout_shapes, stage = state
        _MUON_LAYOUT_STAGE_STATE.set((layout_shapes, stage + 1))
        return [
            [stages[stage][rank] for stages in layout_shapes]
            for rank in range(shard_size)
        ]

    def build_param_shard_metadata_for_group(hsdp_group: Any) -> dict[Any, Any]:
        layout_spec = hsdp_group.layout_spec
        if (
                layout_spec is None
                or not layout_spec.shard_axes
                or not hsdp_group.records
                or not all(
                    isinstance(getattr(record.param, "layout", None), Layout)
                    for record in hsdp_group.records
                )
        ):
            return original_build(hsdp_group)
        layout_shapes = []
        for record in hsdp_group.records:
            parameter = record.param
            coordinate = tuple(parameter.device_mesh.get_coordinate())
            stages = _infer_gather_rank_shapes(
                parameter.layout,
                tuple(parameter.shape),
                coordinate,
                layout_spec.shard_axes,
            )
            inferred = stages[0][coordinate[layout_spec.shard_axes[0][0]]]
            local_shape = tuple(to_local_if_dtensor(parameter.data).shape)
            if inferred != local_shape:
                raise ValueError(
                    f"Muon shard shape disagrees with layout: inferred={inferred}, "
                    f"local={local_shape}, global={tuple(parameter.shape)}"
                )
            layout_shapes.append(stages)
        token = _MUON_LAYOUT_STAGE_STATE.set((layout_shapes, 0))
        try:
            return original_build(hsdp_group)
        finally:
            _MUON_LAYOUT_STAGE_STATE.reset(token)

    muon_shard.build_param_shard_metadata_for_group = build_param_shard_metadata_for_group
    muon.build_param_shard_metadata_for_group = build_param_shard_metadata_for_group
    muon_shard._allgather_shapes_for_group = all_gather_shapes_for_group
    muon_shard._dryrun_metadata_patch_installed = True


def install_dryrun_compatibility() -> None:
    """Install all process-local compatibility patches used by dry-run."""
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return
    _install_model_build_patches()
    _install_fake_subgroup_patch()
    _install_muon_metadata_patch()
    for name, installer in _ENVIRONMENT_PATCHES.items():
        if name not in _ENVIRONMENT_PATCHES_INSTALLED:
            installer()
            _ENVIRONMENT_PATCHES_INSTALLED.add(name)
    _INSTALLED = True


__all__ = [
    "install_dryrun_compatibility",
    "is_model_materialization_deferred",
    "model_build_context",
    "register_environment_patch",
]
