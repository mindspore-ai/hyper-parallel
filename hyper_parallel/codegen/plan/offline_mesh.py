# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Offline mesh: a static stand-in for a real distributed ``DeviceMesh``.

``ShardingPlanner.plan()`` consumes a mesh through a narrow contract —
``mesh_dim_names`` + ``mesh_shape`` for axis validation and planning
(``getattr`` guarded, see ``_validate_dtensor_axes`` /
``_build_mesh_dim_names`` in ``components.distributed.sharding_planner``),
and ``size()`` / ``__getitem__()`` for the per-axis lookups the generated
runtime performs later.  A real ``DeviceMesh`` requires an initialized
process group — on the codegen host (CPU, no ``torchrun``) the default
``hccl`` backend fails at construction.  The offline mesh provides the same
*shape* interface over the parallel dims projected from the YAML, so the
plan can be derived (and frozen into meta) before any training process
group exists.

The plan coordinate system still excludes DP placements, but the offline mesh
must carry the runtime DP domain so planner checks that depend on the dense
region (notably extended EP for MoE) see the same topology as native trainer:

- ``dp`` is declared with the data-parallel size derived from ``WORLD_SIZE``;
- ``tp`` / ``cp`` axes are declared whenever the config declares them
  (present even at size 1 so the planner's validation can see them);
- ``ep`` is **never** declared as a mesh axis.  An explicit ``"ep"`` axis
  selects old-style EP in the planner (``_mark_hf_native_moe``), which
  leaves a TP key on expert weights. The HF-native extended expert-parallel
  path derives the EP group from the dense region and must NOT see an ``"ep"``
  axis. ``ep_size`` still reaches
  the planner through ``plan(..., ep_size=...)``, so EP is planned; it
  just is not a coordinate of the frozen plan;
- ``pp`` contributes to ``ndim`` only; it never appears in
  ``mesh_dim_names`` (the planner rejects pipeline parallelism with extended
  expert parallelism, and the plan coordinate system is a single dp slice).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hyper_parallel.codegen.spec.types import ParallelDimsSpec


@dataclass(frozen=True)
class OfflineMesh:
    """Static mesh with the surface ``ShardingPlanner.plan()`` needs.

    Immutable and JSON-safe: it carries only axis names + sizes, never a
    process group.  ``__getitem__("tp")`` returns the single-axis sub-mesh
    the generated runtime addresses as ``mesh["tp"]``.
    """

    mesh_dim_names: tuple[str, ...] = ()
    mesh_shape: tuple[int, ...] = ()
    _axis_sizes: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if len(self.mesh_dim_names) != len(self.mesh_shape):
            raise ValueError(
                f"mesh_dim_names {self.mesh_dim_names!r} and mesh_shape "
                f"{self.mesh_shape!r} must have the same length"
            )
        object.__setattr__(
            self, "_axis_sizes", dict(zip(self.mesh_dim_names, self.mesh_shape)),
        )

    # -- interface consumed by the planner / generated runtime --------------

    def size(self, dim: Optional[str] = None) -> int:
        """Total size, or the size of one named axis (1 when absent)."""
        if dim is None:
            total = 1
            for size in self.mesh_shape:
                total *= size
            return total
        return self._axis_sizes.get(dim, 1)

    def __getitem__(self, dim: str) -> "OfflineMesh":
        """Return the single-axis sub-mesh for ``dim`` (``mesh["tp"]``).

        Unknown axis names yield an empty (size-1) mesh rather than raising,
        matching ``DeviceMesh.__getitem__``'s tolerance for axes that do not
        exist on this mesh.
        """
        size = self._axis_sizes.get(dim)
        if size is None:
            return OfflineMesh()
        return OfflineMesh(mesh_dim_names=(dim,), mesh_shape=(size,))

    @property
    def ndim(self) -> int:
        """Number of mesh dimensions."""
        return len(self.mesh_dim_names)


def build_offline_mesh(parallel_dims: ParallelDimsSpec) -> OfflineMesh:
    """Build the static mesh for the projected parallel dims.

    Axis rules (see the module docstring):

    - ``dp`` / ``cp`` / ``tp`` present when the config declares them (even at size 1);
    - ``ep`` never named — naming it would select old-style EP and leave a
      TP key on fused expert weights; ``ep_size`` is passed to ``plan()``
      instead so the planner derives the expert mesh;
    - ``pp`` counted into ``ndim`` but never named;
    """
    names: list[str] = []
    sizes: list[int] = []
    if parallel_dims.dp_size >= 1:
        names.append("dp")
        sizes.append(parallel_dims.dp_size)
    if parallel_dims.cp_size >= 1:
        names.append("cp")
        sizes.append(parallel_dims.cp_size)
    if parallel_dims.tp_size >= 1:
        names.append("tp")
        sizes.append(parallel_dims.tp_size)
    if parallel_dims.pp_size > 1:
        names.append("pp")
        sizes.append(parallel_dims.pp_size)
    return OfflineMesh(mesh_dim_names=tuple(names), mesh_shape=tuple(sizes))


__all__ = ["OfflineMesh", "build_offline_mesh"]
