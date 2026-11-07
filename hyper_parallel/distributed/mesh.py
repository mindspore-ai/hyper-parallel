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
"""Mesh domains and the distributed-setup container for AutoModels.

Split out of the former ``components/distributed/infrastructure.py`` in
stage 7 (05 §15.11 step 3): the mesh topology objects are AutoModels-side
model-construction inputs, while process-group lifecycle helpers moved to
``hyper_parallel.trainer.runtime.distributed`` /
``hyper_parallel.trainer.runtime.logging``.
"""

__all__ = [
    "DistributedSetup",
    "LogicalDeviceMesh",
    "MeshContext",
    "ParallelLayout",
    "ParallelTopologySpec",
    "infer_minimum_world_size",
    "resolve_parallel_layout",
]

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import torch.distributed as dist

from hyper_parallel import init_device_mesh


def _compatible_parallel_degree(value: Any) -> Any:
    """Preserve the Trainer's historical minimum parallel degree of one."""
    return max(1, value)


def _positive_world_size(value: Any) -> int:
    """Validate the process-group or inferred world size."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"world_size must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class ParallelTopologySpec:
    """Normalized parallel degrees shared by runtime and offline planning."""

    tp_size: int
    cp_size: int
    pp_size: int
    ep_size: int
    dp_shard_size: int
    edp_shard_size: int
    sequence_parallel: bool
    loss_parallel: bool

    @classmethod
    def from_config(cls, config: Any) -> "ParallelTopologySpec":
        """Read parallel degrees using the Trainer's compatibility semantics."""
        accelerator = config.accelerator
        fsdp_config = config.fsdp_config
        return cls(
            tp_size=_compatible_parallel_degree(accelerator.tp_size),
            cp_size=_compatible_parallel_degree(accelerator.cp_size),
            pp_size=_compatible_parallel_degree(accelerator.pp_size),
            ep_size=_compatible_parallel_degree(accelerator.ep_size),
            dp_shard_size=_compatible_parallel_degree(fsdp_config.dp_shard_size),
            edp_shard_size=_compatible_parallel_degree(fsdp_config.edp_shard_size),
            sequence_parallel=bool(accelerator.sequence_parallel),
            loss_parallel=bool(accelerator.loss_parallel),
        )


@dataclass(frozen=True)
class ParallelLayout:
    """Resolved parallel topology independent of devices and process groups."""

    spec: ParallelTopologySpec
    world_size: int
    dp_size: int
    dp_replicate_size: int

    @property
    def dense_world_size(self) -> int:
        """Return the rank count in one pipeline stage."""
        return self.dp_size * self.spec.cp_size * self.spec.tp_size


def resolve_parallel_layout(
    spec: ParallelTopologySpec, world_size: int
) -> ParallelLayout:
    """Validate a world size and resolve its DP/FSDP/EP layout."""
    world_size = _positive_world_size(world_size)
    non_dp_size = spec.tp_size * spec.cp_size * spec.pp_size
    if world_size % non_dp_size != 0:
        raise ValueError(
            f"world_size {world_size} is not divisible by non-DP size "
            f"{non_dp_size}"
        )
    dp_size = world_size // non_dp_size
    fsdp_data_parallel_size = dp_size * spec.cp_size
    if fsdp_data_parallel_size % spec.dp_shard_size != 0:
        raise ValueError(
            "DP+CP size "
            f"{fsdp_data_parallel_size} is not divisible by FSDP shard size "
            f"{spec.dp_shard_size}"
        )
    dense_world_size = dp_size * spec.cp_size * spec.tp_size
    if dense_world_size % spec.ep_size != 0:
        raise ValueError(
            f"expert domain size ({dense_world_size}) must be divisible by "
            f"ep_size ({spec.ep_size})"
        )
    expert_data_parallel_size = dense_world_size // spec.ep_size
    if expert_data_parallel_size % spec.edp_shard_size != 0:
        raise ValueError(
            f"expert data-parallel size ({expert_data_parallel_size}) must be "
            f"divisible by edp_shard_size ({spec.edp_shard_size})"
        )
    return ParallelLayout(
        spec=spec,
        world_size=world_size,
        dp_size=dp_size,
        dp_replicate_size=fsdp_data_parallel_size // spec.dp_shard_size,
    )


def infer_minimum_world_size(spec: ParallelTopologySpec) -> int:
    """Return the smallest world size satisfying every parallel contract."""
    dense_world_size = math.lcm(
        spec.tp_size * spec.cp_size,
        spec.ep_size,
        spec.tp_size * spec.dp_shard_size,
        spec.ep_size * spec.edp_shard_size,
    )
    return dense_world_size * spec.pp_size


@dataclass(frozen=True)
class LogicalDeviceMesh:
    """Process-group-free subset of the DeviceMesh planning interface."""

    mesh_dim_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical mesh shape."""
        return self.mesh_shape

    def __getitem__(
        self, mesh_dim_names: Union[str, Tuple[str, ...]]
    ) -> "LogicalDeviceMesh":
        """Return a logical submesh selected by dimension name."""
        names = (
            (mesh_dim_names,)
            if isinstance(mesh_dim_names, str)
            else tuple(mesh_dim_names)
        )
        missing = tuple(name for name in names if name not in self.mesh_dim_names)
        if missing:
            raise KeyError(f"logical mesh has no dimensions {missing!r}")
        shape_by_name = dict(zip(self.mesh_dim_names, self.mesh_shape))
        return LogicalDeviceMesh(
            mesh_dim_names=names,
            mesh_shape=tuple(shape_by_name[name] for name in names),
        )

    def size(self, mesh_dim: Optional[Union[int, str]] = None) -> int:
        """Return the total or selected logical dimension size."""
        if mesh_dim is None:
            return math.prod(self.mesh_shape)
        index = (
            self.mesh_dim_names.index(mesh_dim)
            if isinstance(mesh_dim, str)
            else mesh_dim
        )
        return self.mesh_shape[index]


def _init_topology_mesh(mesh_kwargs: dict[str, Any]) -> Any:
    """Create a mesh while preserving compatibility with lightweight test fakes."""
    supported_arguments = inspect.signature(init_device_mesh).parameters
    filtered_kwargs = {
        name: value
        for name, value in mesh_kwargs.items()
        if name in supported_arguments
    }
    return init_device_mesh(**filtered_kwargs)


@dataclass
class MeshContext:
    """Runtime topology and the mesh domains consumed by dual-mode Trainer."""
    device_mesh: Any = None
    fsdp_non_moe_mesh: Any = None
    fsdp_moe_mesh: Any = None
    dp_size: int = 1
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    edp_shard_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    dp_rank: int = 0
    tp_rank: int = 0
    cp_rank: int = 0
    pp_rank: int = 0
    ep_rank: int = 0
    sequence_parallel: bool = False
    loss_parallel: bool = False

    @classmethod
    def from_layout(cls, layout: ParallelLayout) -> "MeshContext":
        """Create common mesh metadata from a resolved parallel layout."""
        spec = layout.spec
        return cls(
            dp_size=layout.dp_size,
            dp_replicate_size=layout.dp_replicate_size,
            dp_shard_size=spec.dp_shard_size,
            edp_shard_size=spec.edp_shard_size,
            tp_size=spec.tp_size,
            cp_size=spec.cp_size,
            pp_size=spec.pp_size,
            ep_size=spec.ep_size,
            sequence_parallel=spec.sequence_parallel,
            loss_parallel=spec.loss_parallel,
        )

    @property
    def pp_enabled(self) -> bool:
        """Whether pipeline parallelism is enabled (pp_size > 1)."""
        return self.pp_size > 1

    @property
    def dp_cp_mesh(self) -> Any:
        """DP+CP joint mesh for all-reduce.

        Mirrors the derivation in ``FinetuneRecipe.setup()``: take the DP
        dimension together with CP (if present) and flatten to 1-D.
        """
        if self.device_mesh is None:
            return None
        dim_names = self.device_mesh.mesh_dim_names or ()
        selected_dims = tuple(dim_name for dim_name in ("dp", "cp") if dim_name in dim_names)
        if not selected_dims:
            selected_dims = tuple(
                dim_name
                for dim_name in ("dp_replicate", "dp_shard", "cp")
                if dim_name in dim_names
            )
        if not selected_dims:
            return None
        sub = (
            self.device_mesh[selected_dims[0]]
            if len(selected_dims) == 1
            else self.device_mesh[selected_dims]
        )
        if sub.ndim > 1:
            sub = sub.flatten("dp_cp")
        return sub

    @property
    def cp_mesh(self) -> Any:
        """CP submesh for CP utilities (e.g. shard_batch_for_cp).

        Returns ``None`` when context parallelism is disabled (cp_size == 1)
        or no DeviceMesh is available.
        """
        if self.device_mesh is None:
            return None
        dim_names = self.device_mesh.mesh_dim_names or ()
        if "cp" in dim_names:
            return self.device_mesh["cp"]
        return None

    def build_meshs(self, device_type: str, world_size: int) -> None:
        """Build the device, dense-FSDP and expert-FSDP mesh domains.

        The device mesh preserves the existing (dp, cp, tp) rank order. Dense
        FSDP flattens the DP+CP domain and reshapes it according to the
        configured shard size, while expert FSDP derives an (edp, ep)
        topology from the flattened device domain.

        Args:
            device_type: Device type accepted by init_device_mesh.
            world_size: Number of ranks in the process group.

        Raises:
            ValueError: If the configured topology cannot cover world_size or
                cannot form the requested expert mesh.
        """
        dense_world_size = self.dp_size * self.cp_size * self.tp_size
        expected_world_size = dense_world_size * self.pp_size
        if expected_world_size != world_size:
            raise ValueError(
                "dp_size * cp_size * tp_size * pp_size "
                f"must equal world_size ({expected_world_size} != {world_size})"
            )
        fsdp_data_parallel_size = self.dp_size * self.cp_size
        configured_fsdp_size = self.dp_replicate_size * self.dp_shard_size
        if configured_fsdp_size != fsdp_data_parallel_size:
            raise ValueError(
                "dp_replicate_size * dp_shard_size must equal dp_size * cp_size "
                f"({configured_fsdp_size} != {fsdp_data_parallel_size})"
            )

        init_backend = dist.is_initialized()
        stage_rank_start = 0
        if init_backend and self.pp_size > 1:
            stage_rank_start = (dist.get_rank() // dense_world_size) * dense_world_size
        stage_rank_list = tuple(
            range(stage_rank_start, stage_rank_start + dense_world_size)
        )
        device_mesh_kwargs = {
            "device_type": device_type,
            "mesh_shape": (self.dp_size, self.cp_size, self.tp_size),
            "mesh_dim_names": ("dp", "cp", "tp"),
            "init_backend": init_backend,
        }
        if init_backend:
            device_mesh_kwargs["rank_list"] = stage_rank_list
        self.device_mesh = _init_topology_mesh(device_mesh_kwargs)

        dense_rank_list = (
            tuple(self.device_mesh.rank_list)
            if hasattr(self.device_mesh, "rank_list")
            else None
        )
        dense_mesh_kwargs = {
            "device_type": device_type,
            "mesh_shape": (
                self.dp_replicate_size,
                self.dp_shard_size,
                self.tp_size,
            ),
            "mesh_dim_names": ("fsdp_replicate", "fsdp_shard", "tp"),
            "init_backend": init_backend,
        }
        if init_backend and dense_rank_list is not None:
            dense_mesh_kwargs["rank_list"] = dense_rank_list
        self.fsdp_non_moe_mesh = _init_topology_mesh(dense_mesh_kwargs)
        _validate_dense_tp_rank_layout(self.device_mesh, self.fsdp_non_moe_mesh)

        self._build_expert_parallel_mesh(device_type, init_backend, dense_rank_list)

    def _build_expert_parallel_mesh(
        self,
        device_type: str,
        init_backend: bool,
        dense_rank_list: Optional[Tuple[int, ...]],
    ) -> None:
        """Build the expert-parallel FSDP mesh when expert parallelism is enabled."""
        self.fsdp_moe_mesh = None
        if self.ep_size <= 1:
            return
        expert_domain_size = math.prod(self.device_mesh.mesh_shape)
        if expert_domain_size % self.ep_size != 0:
            raise ValueError(
                f"expert domain size ({expert_domain_size}) must be divisible by "
                f"ep_size ({self.ep_size})"
            )
        edp_size = expert_domain_size // self.ep_size
        if edp_size % self.edp_shard_size != 0:
            raise ValueError(
                f"expert data-parallel size ({edp_size}) must be divisible by "
                f"edp_shard_size ({self.edp_shard_size})"
            )
        edp_replicate_size = edp_size // self.edp_shard_size
        expert_mesh_shape = (
            (edp_replicate_size, self.edp_shard_size, self.ep_size)
            if edp_replicate_size > 1
            else (self.edp_shard_size, self.ep_size)
        )
        expert_mesh_names = (
            ("edp_replicate", "edp_shard", "ep")
            if edp_replicate_size > 1
            else ("edp_shard", "ep")
        )
        expert_mesh_kwargs = {
            "device_type": device_type,
            "mesh_shape": expert_mesh_shape,
            "mesh_dim_names": expert_mesh_names,
            "init_backend": init_backend,
        }
        if init_backend and dense_rank_list is not None:
            expert_mesh_kwargs["rank_list"] = dense_rank_list
        self.fsdp_moe_mesh = _init_topology_mesh(expert_mesh_kwargs)


def _validate_dense_tp_rank_layout(device_mesh: Any, fsdp_non_moe_mesh: Any) -> None:
    """Ensure both TP child meshes describe the same groups and local rank."""
    if not hasattr(device_mesh, "rank_list") or not hasattr(fsdp_non_moe_mesh, "rank_list"):
        return
    device_tp_mesh = device_mesh["tp"]
    fsdp_tp_mesh = fsdp_non_moe_mesh["tp"]
    if tuple(device_tp_mesh.rank_list) != tuple(fsdp_tp_mesh.rank_list):
        raise ValueError("device_mesh['tp'] and fsdp_non_moe_mesh['tp'] rank groups differ")
    if device_tp_mesh.get_local_rank() != fsdp_tp_mesh.get_local_rank():
        raise ValueError("device_mesh['tp'] and fsdp_non_moe_mesh['tp'] local ranks differ")


@dataclass
class DistributedSetup:
    """Unified distributed configuration container.

    Following design doc 06 §3.
    """
    mesh_context: MeshContext = field(default_factory=MeshContext)
    strategy_config: Any = None
    # Normalized ``{match: ModuleShardingSpec}`` mapping (or None) — the
    # Trainer desugars YAML PlanOverride entries via
    # entries_to_plan_overrides (when-filtering against the built mesh
    # included) BEFORE the AutoModels build pipeline runs; AutoModels never
    # sees the raw Trainer YAML DTO list (05 §15.2.6).
    plan_overrides: Any = None
    # Normalized module replacement rules (tuple[ModuleReplacementSpec])
    # desugared Trainer-side via entries_to_module_replacements.
    module_replacements: Any = None
    # Non-forward method patches installed after module replacement and before planning.
    method_replacements: Any = None
    # TrainingConfig.training.low_precision. Kept on the setup so model
    # construction and sharding consume one resolved policy object.
    low_precision_config: Any = None
    fp32_main_params: bool = False

    @classmethod
    def from_config(
        cls, config: Any, mesh_context: MeshContext, *, strategy_enabled: bool
    ) -> "DistributedSetup":
        """Assemble model-build configuration independently of mesh execution.

        Args:
            config: Resolved training configuration.
            mesh_context: Topology prepared by the caller.
            strategy_enabled: Whether to retain the configured FSDP strategy.

        Returns:
            Setup with common model-build policies populated.
        """
        return cls(
            mesh_context=mesh_context,
            strategy_config=config.fsdp_config if strategy_enabled else None,
            low_precision_config=getattr(
                getattr(config, "training", None), "low_precision", None
            ),
            fp32_main_params=config.optimizer.fp32_main_params,
        )
