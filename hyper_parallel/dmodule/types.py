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
"""Shared type primitives for declarative protocol sharding.

Provides mesh axis names and :data:`NamedPlacement` for
:class:`~hyper_parallel.dmodule.sharding.ShardingConfig` and
:func:`~hyper_parallel.dmodule.sharding.resolve_placements`.

Example:
    Build a per-axis placement map and pass it to :class:`ShardingConfig`::

        from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
        from hyper_parallel.dmodule.types import MeshAxisName

        weight_sharding = {
            MeshAxisName.TP: Shard(0),
            MeshAxisName.DP: Replicate(),
        }
"""

from enum import Enum

from hyper_parallel.core.dtensor.placement_types import Placement


class StrEnum(str, Enum):
    """``str`` + ``Enum`` for Python versions before 3.11.

    Members compare equal to their string values, which helps when matching
    :attr:`~hyper_parallel.core.dtensor.device_mesh.DeviceMesh.mesh_dim_names`.

    Example::

        MeshAxisName.TP == "tp"
        str(MeshAxisName.TP) == "tp"
    """


class MeshAxisName(StrEnum):
    """Standard names for :class:`~hyper_parallel.core.dtensor.device_mesh.DeviceMesh` axes.

    Use ``axis`` / ``axes`` when referring to a mesh dimension name; use ``dim``
    for tensor dimensions. Values are lowercase strings and must match
    ``mesh_dim_names`` passed to :func:`~hyper_parallel.core.dtensor.device_mesh.init_device_mesh`
    (matching is case-sensitive).

    Example:
        Create a 2-D mesh and reference axes in config::

            from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
            from hyper_parallel.dmodule.types import MeshAxisName

            mesh = init_device_mesh(
                "npu",
                mesh_shape=(2, 4),
                mesh_dim_names=(MeshAxisName.DP.value, MeshAxisName.TP.value),
            )
            # mesh.mesh_dim_names == ("dp", "tp")

        Use enum members in :class:`~hyper_parallel.dmodule.sharding.ShardingConfig`::

            from hyper_parallel.core.dtensor.placement_types import Shard
            from hyper_parallel.dmodule.types import MeshAxisName

            tp_colwise = {MeshAxisName.TP: Shard(0)}
    """

    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    FSDP = "fsdp"
    TP = "tp"
    CP = "cp"
    PP = "pp"
    EP = "ep"
    EFSDP = "efsdp"


#: Map from mesh axis name to :class:`~hyper_parallel.core.dtensor.placement_types.Placement`.
#:
# Keys are :class:`MeshAxisName` (or the same lowercase strings). Values describe
# how a tensor is placed on that mesh axis. Convert to an ordered placement list
# with :func:`~hyper_parallel.dmodule.sharding.resolve_placements`.
#
# Example::
#
#     from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
#     from hyper_parallel.dmodule.types import MeshAxisName, NamedPlacement
#
#     named: NamedPlacement = {
#         MeshAxisName.DP: Replicate(),
#         MeshAxisName.TP: Shard(0),
#     }
NamedPlacement = dict[MeshAxisName, Placement]

__all__ = [
    "MeshAxisName",
    "NamedPlacement",
    "Placement",
    "StrEnum",
]
