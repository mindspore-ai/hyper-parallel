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
"""Declarative sharding specs and placement resolution for dmodule.

Provides :class:`ShardingConfig`, :class:`LocalMapConfig`, and
:func:`resolve_placements` to convert :data:`~hyper_parallel.dmodule.types.NamedPlacement`
maps into mesh-axis-ordered placement lists for
:meth:`~hyper_parallel.dmodule.module.Module.parallelize`.

Example:
    Column-parallel weight with replicated activations::

        from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
        from hyper_parallel.dmodule.sharding import ShardingConfig
        from hyper_parallel.dmodule.types import MeshAxisName

        ShardingConfig(
            state_shardings={"weight": {MeshAxisName.TP: Shard(0)}},
            in_dst_shardings={"x": {MeshAxisName.TP: Replicate()}},
        )
"""

from dataclasses import dataclass, field

from hyper_parallel.core.dtensor.placement_types import Placement, Replicate
from hyper_parallel.dmodule.types import MeshAxisName, NamedPlacement

__all__ = [
    "LocalMapConfig",
    "NamedPlacement",
    "ShardingConfig",
    "resolve_placements",
]


@dataclass(kw_only=True, slots=True)
class LocalMapConfig:
    """Gradient placement specs for ``local_map`` modules (wired in a future milestone).

    Args:
        in_grad_placements: One :class:`~hyper_parallel.dmodule.types.NamedPlacement`
            per forward input that uses ``local_map``.

    Example::

        from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
        from hyper_parallel.dmodule.sharding import LocalMapConfig, ShardingConfig
        from hyper_parallel.dmodule.types import MeshAxisName

        sharding = ShardingConfig(
            state_shardings={"weight": {MeshAxisName.TP: Shard(0)}},
            local_map=LocalMapConfig(
                in_grad_placements=({MeshAxisName.TP: Replicate()},),
            ),
        )
    """

    in_grad_placements: tuple[NamedPlacement, ...]

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (debug/logging).

        Returns:
            Dict with a ``repr`` field for the whole spec.
        """
        return {"repr": repr(self)}


@dataclass(kw_only=True, slots=True)
class ShardingConfig:
    """Declarative sharding for one :class:`~hyper_parallel.dmodule.module.Module`.

    Field names for ``in_*`` / ``out_*`` must match ``forward`` argument names
    (positional args are mapped by name inside :meth:`Module.parallelize`).

    Args:
        state_shardings: Parameter path (e.g. ``"weight"``) → :data:`NamedPlacement`.
        in_src_shardings: Optional source layout for each forward input before adapt.
        in_dst_shardings: Target layout for each forward input (redistribute if needed).
        out_src_shardings: Optional source layout(s) for forward output(s).
        out_dst_shardings: Target layout for a single tensor output.
        local_map: Optional :class:`LocalMapConfig` (not yet applied in ``parallelize``).

    Example:
        TP column-parallel weight with replicated activations::

            from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
            from hyper_parallel.dmodule.sharding import ShardingConfig
            from hyper_parallel.dmodule.types import MeshAxisName

            ShardingConfig(
                state_shardings={
                    "weight": {MeshAxisName.TP: Shard(0)},
                },
                in_dst_shardings={
                    "x": {MeshAxisName.TP: Replicate()},
                },
                out_dst_shardings={MeshAxisName.TP: Shard(-1)},
            )
    """

    state_shardings: dict[str, NamedPlacement] = field(default_factory=dict)
    in_src_shardings: dict[str, NamedPlacement] | None = None
    in_dst_shardings: dict[str, NamedPlacement] | None = None
    out_src_shardings: NamedPlacement | tuple[NamedPlacement, ...] | None = None
    out_dst_shardings: NamedPlacement | None = None
    local_map: LocalMapConfig | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (debug/logging).

        Returns:
            Dict with a ``repr`` field for the whole spec.
        """
        return {"repr": repr(self)}


def _axis_key(axis: MeshAxisName | str) -> str:
    """Normalize a mesh axis key to a plain string for lookup."""
    if isinstance(axis, MeshAxisName):
        return axis.value
    return str(axis)


def resolve_placements(
    named: NamedPlacement,
    mesh_axis_names: tuple[str, ...],
) -> list[Placement]:
    """Convert :data:`NamedPlacement` to an ordered placement list for DTensor APIs.

    The returned list aligns with ``mesh_axis_names`` (same order and length).
    Axes listed on the mesh but omitted from *named* default to
    :class:`~hyper_parallel.core.dtensor.placement_types.Replicate`. Keys in *named*
    that are not present on the mesh (including case mismatches) raise
    :class:`ValueError`.

    Args:
        named: Placement per mesh axis name (:class:`MeshAxisName` or lowercase str).
        mesh_axis_names: Names from ``DeviceMesh.mesh_dim_names`` in axis order.

    Returns:
        One :class:`~hyper_parallel.core.dtensor.placement_types.Placement` per mesh
        axis, suitable for :func:`~hyper_parallel.core.dtensor.dtensor.distribute_tensor`
        and ``DTensor.redistribute``.

    Raises:
        ValueError: If *named* declares an axis that does not appear on the mesh.

    Example:
        Partial TP spec on a ``(dp, tp)`` mesh — ``dp`` is filled with ``Replicate``::

            from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
            from hyper_parallel.dmodule.sharding import resolve_placements
            from hyper_parallel.dmodule.types import MeshAxisName

            mesh_axis_names = ("dp", "tp")
            named = {MeshAxisName.TP: Shard(0)}

            placements = resolve_placements(named, mesh_axis_names)
            assert placements[0].is_replicate()
            assert placements[1].is_shard(0)

        Case-sensitive matching — ``"TP"`` does not match ``"tp"``::

            resolve_placements({"TP": Shard(0)}, ("tp",))  # raises ValueError
    """
    normalized: dict[str, Placement] = {
        _axis_key(axis): placement for axis, placement in named.items()
    }
    result: list[Placement] = []
    for axis_name in mesh_axis_names:
        if axis_name in normalized:
            result.append(normalized[axis_name])
        else:
            result.append(Replicate())

    declared = set(normalized.keys())
    extra = declared - set(mesh_axis_names)
    if extra:
        raise ValueError(f"NamedPlacement has axes not in mesh: {extra}")
    return result
