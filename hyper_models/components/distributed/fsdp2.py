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
"""FSDP2 integration for the dual-mode Trainer."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from hyper_parallel import DeviceMesh, HSDPModule, Replicate, fully_shard
from hyper_parallel.core.dtensor.placement_types import Partial, Placement
import hyper_parallel.core.fully_shard.utils as fully_shard_utils
from hyper_parallel.platform import get_platform
from hyper_models.components.distributed.config import FSDP2Config
from hyper_models.components.distributed.infrastructure import MeshContext
from hyper_models.components.distributed.source_shard import FSDP_OWNED_DIMS

logger = logging.getLogger(__name__)
platform = get_platform()
ModuleClass = platform.Module
ParameterClass = platform.Parameter

SourceShardInfoByFQN: TypeAlias = Mapping[  # pylint: disable=invalid-name
    str, tuple[tuple[Placement, ...], DeviceMesh]
]
SourceShardInfoByParam: TypeAlias = dict[  # pylint: disable=invalid-name
    ParameterClass,
    "fully_shard_utils.SourceShardMetaInfo",
]


@dataclass(frozen=True)
class _WrapModuleInfo:
    """One transformer block selected by the configured wrap policy."""

    fqn: str
    module: ModuleClass


def _is_expert_source_mesh(mesh: DeviceMesh | None) -> bool:
    """Whether a source mesh carries the expert (EP) axis."""
    return mesh is not None and "ep" in (getattr(mesh, "mesh_dim_names", None) or ())


class FSDP2Manager:
    """Configure and apply nested FSDP2 wrapping for a dual-mode model."""

    def __init__(self, config: FSDP2Config, mesh_context: MeshContext) -> None:
        """Initialize the manager from resolved YAML config and runtime mesh.

        Args:
            config: Resolved top-level ``fsdp_config``.
            mesh_context: Runtime topology containing the shared world mesh.
        """
        self.config = config
        self.mesh_context = mesh_context

    def _build_source_shard_info_by_param(
        self,
        model: ModuleClass,
        source_shard_info: SourceShardInfoByFQN | None,
    ) -> SourceShardInfoByParam | None:
        """Resolve stable FQNs to parameter identities after model rewrites.

        Activation-checkpoint wrappers may add an internal child-module prefix,
        but the supported wrapper contract removes that prefix from
        ``named_modules``/``named_parameters``. Therefore planner metadata
        produced before checkpointing remains resolvable here, while FSDP still
        receives the final parameter objects that it will manage.

        Args:
            model: Model after sharding and checkpoint rewrites, before layer
                compile is installed.
            source_shard_info: Source-layout metadata produced by the TP planner.

        Returns:
            Parameter-keyed metadata for later ``fully_shard`` calls, or
            ``None`` when tensor parallelism is disabled.

        Raises:
            ValueError: If TP metadata is missing, references an unknown FQN,
                contains ``Partial``, or gives tied aliases conflicting layouts.
        """
        if source_shard_info is None:
            if self.mesh_context.tp_size > 1:
                raise ValueError("TP is enabled but source_shard_info is missing")
            return None
        if self.mesh_context.tp_size > 1 and not source_shard_info:
            raise ValueError("TP is enabled but source_shard_info is empty")

        parameters_by_fqn = dict(model.named_parameters(remove_duplicate=False))
        metadata_by_parameter: SourceShardInfoByParam = {}
        for parameter_fqn, (placements, source_mesh) in source_shard_info.items():
            parameter = parameters_by_fqn.get(parameter_fqn)
            if parameter is None:
                raise ValueError(
                    f"source_shard_info contains unknown parameter FQN: {parameter_fqn}"
                )
            if any(isinstance(placement, Partial) for placement in placements):
                raise ValueError(
                    f"source_shard_info does not support Partial placement: {parameter_fqn}"
                )

            source_shard_info = fully_shard_utils.SourceShardMetaInfo(  # pylint: disable=no-member
                mesh=source_mesh,
                placements=tuple(placements),
                origin_is_dtensor=False,
            )
            previous_source_shard_info = metadata_by_parameter.get(parameter)
            if previous_source_shard_info is not None and (
                previous_source_shard_info.mesh is not source_shard_info.mesh
                or previous_source_shard_info.placements != source_shard_info.placements
            ):
                raise ValueError(
                    "Tied parameter aliases have conflicting source layouts; "
                    f"conflict found at {parameter_fqn}"
                )
            metadata_by_parameter[parameter] = source_shard_info
        return metadata_by_parameter

    def _resolve_replicate_params(
        self,
        model: ModuleClass,
    ) -> set[ParameterClass] | None:
        """Resolve configured replicate-parameter FQNs on the final model."""
        if not self.config.replicate_params:
            return None

        parameters_by_fqn = dict(model.named_parameters(remove_duplicate=False))
        replicate_params = set()
        for parameter_fqn in self.config.replicate_params:
            parameter = parameters_by_fqn.get(parameter_fqn)
            if parameter is None:
                raise ValueError(
                    "replicate_params contains unknown parameter FQN: "
                    f"{parameter_fqn}"
                )
            replicate_params.add(parameter)
        return replicate_params

    def _build_fsdp_actual_mesh(self, expert: bool = False) -> DeviceMesh:
        """Select the prebuilt dense or expert FSDP child mesh."""
        if expert and self.mesh_context.fsdp_moe_mesh is None:
            raise ValueError("Expert FSDP units require MeshContext.fsdp_moe_mesh")
        if expert:
            expert_mesh = self.mesh_context.fsdp_moe_mesh
            if "edp_replicate" in (expert_mesh.mesh_dim_names or ()):
                return expert_mesh[("edp_replicate", "edp_shard")]
            return expert_mesh["edp_shard"]

        if not expert and self.mesh_context.fsdp_non_moe_mesh is not None:
            dense_mesh = self.mesh_context.fsdp_non_moe_mesh
            if self.mesh_context.dp_replicate_size > 1:
                return dense_mesh[("fsdp_replicate", "fsdp_shard")]
            return dense_mesh["fsdp_shard"]

        # Compatibility path for callers that construct a MeshContext around
        # the old combined world mesh directly.
        world_mesh = self.mesh_context.device_mesh
        if world_mesh is None:
            raise ValueError("FSDP2 requires MeshContext.fsdp_non_moe_mesh")
        mesh_dim_names = world_mesh.mesh_dim_names
        if mesh_dim_names is None:
            raise ValueError("FSDP2 requires named DeviceMesh dimensions")
        shard_dim_names = tuple(
            dim_name for dim_name in ("dp_shard", "cp") if dim_name in mesh_dim_names
        )
        if not shard_dim_names:
            raise ValueError("FSDP2 requires a dp_shard or cp dimension in the world mesh")
        shard_mesh = (
            world_mesh[shard_dim_names[0]]
            if len(shard_dim_names) == 1
            else world_mesh[shard_dim_names].flatten("fsdp_shard")
        )
        if "dp_replicate" not in mesh_dim_names:
            return shard_mesh
        return DeviceMesh.concatenate([world_mesh["dp_replicate"], shard_mesh])

    def _build_fully_shard_kwargs(
        self,
        fsdp_mesh: DeviceMesh,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build shared sublayer kwargs and the root-specialized copy."""
        if self.config.mp_policy is None:
            mp_policy = fully_shard_utils.MixedPrecisionPolicy()
        else:
            mp_policy = fully_shard_utils.MixedPrecisionPolicy(
                param_dtype=self.config.mp_policy.param_dtype,
                reduce_dtype=self.config.mp_policy.reduce_dtype,
                output_dtype=self.config.mp_policy.output_dtype,
            )
        if self.config.offload_policy is None:
            offload_policy = fully_shard_utils.OffloadPolicy()
        else:
            offload_policy = fully_shard_utils.CPUOffloadPolicy(
                pin_memory=self.config.offload_policy.pin_memory
            )

        fsdp_sublayer_kwargs = {
            "mesh": fsdp_mesh,
            "reshard_after_forward": self.config.reshard_after_forward,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "comm_fusion": self.config.comm_fusion,
            "comm_fusion_zero_copy": self.config.comm_fusion_zero_copy,
        }
        fsdp_root_kwargs = fsdp_sublayer_kwargs.copy()
        fsdp_root_kwargs["reshard_after_forward"] = False
        return fsdp_sublayer_kwargs, fsdp_root_kwargs

    @staticmethod
    def _find_transformer_block_modules(
        model: ModuleClass,
    ) -> tuple[list[_WrapModuleInfo], set[int]]:
        """Find transformer blocks under HF gradient-checkpointing containers."""
        wrap_modules = []
        wrapped_module_ids = set()
        for container_fqn, container in model.named_modules():
            if id(container) in wrapped_module_ids:
                continue
            if not hasattr(container, "gradient_checkpointing"):
                continue
            for child_name, child in container.named_children():
                blocks = list(child.children())
                if not blocks:
                    continue
                child_fqn = f"{container_fqn}.{child_name}" if container_fqn else child_name
                for block_index, block in enumerate(blocks):
                    if id(block) in wrapped_module_ids:
                        continue
                    wrapped_module_ids.add(id(block))
                    wrap_modules.append(
                        _WrapModuleInfo(f"{child_fqn}.{block_index}", block)
                    )
        return wrap_modules, wrapped_module_ids

    def _find_expert_wrap_modules(
        self,
        model: ModuleClass,
        metadata_by_parameter: SourceShardInfoByParam | None,
        wrapped_module_ids: set[int],
    ) -> list[_WrapModuleInfo]:
        """Find nested expert containers whose parameters use the expert mesh."""
        if metadata_by_parameter is None or self.mesh_context.fsdp_moe_mesh is None:
            return []
        expert_parameters = {
            parameter
            for parameter, source_shard_info in metadata_by_parameter.items()
            if _is_expert_source_mesh(source_shard_info.mesh)
        }
        expert_module_fqns = {
            parameter_fqn.split(".experts.", 1)[0] + ".experts"
            for parameter_fqn, parameter in model.named_parameters(remove_duplicate=False)
            if parameter in expert_parameters and ".experts." in parameter_fqn
        }
        module_by_fqn = dict(model.named_modules())
        expert_wrap_modules = []
        for expert_module_fqn in sorted(expert_module_fqns):
            expert_module = module_by_fqn.get(expert_module_fqn)
            if expert_module is None or id(expert_module) in wrapped_module_ids:
                continue
            wrapped_module_ids.add(id(expert_module))
            expert_wrap_modules.append(
                _WrapModuleInfo(expert_module_fqn, expert_module)
            )
        return expert_wrap_modules

    def _find_wrap_modules(
        self,
        model: ModuleClass,
        metadata_by_parameter: SourceShardInfoByParam | None = None,
    ) -> list[_WrapModuleInfo]:
        """Find HF transformer blocks selected by ``transformer_block`` policy."""
        wrap_modules, wrapped_module_ids = self._find_transformer_block_modules(model)
        wrap_modules.extend(
            self._find_expert_wrap_modules(
                model,
                metadata_by_parameter,
                wrapped_module_ids,
            )
        )
        if not wrap_modules:
            raise ValueError(
                "wrap_policy='transformer_block' did not find a non-empty "
                "block container under an HF gradient_checkpointing module"
            )
        return wrap_modules

    @staticmethod
    def _resolve_parameter_owners(
        model: ModuleClass,
        wrap_modules: list[_WrapModuleInfo],
    ) -> dict[ParameterClass, ModuleClass]:
        """Assign each parameter to its deepest common FSDP wrap module."""
        aliases_by_parameter: dict[ParameterClass, list[str]] = defaultdict(list)
        for parameter_fqn, parameter in model.named_parameters(remove_duplicate=False):
            aliases_by_parameter[parameter].append(parameter_fqn)

        owner_by_parameter = {}
        for parameter, parameter_fqns in aliases_by_parameter.items():
            common_wrap_modules = [
                wrap_module
                for wrap_module in wrap_modules
                if all(
                    parameter_fqn.startswith(f"{wrap_module.fqn}.")
                    for parameter_fqn in parameter_fqns
                )
            ]
            if common_wrap_modules:
                owner_by_parameter[parameter] = max(
                    common_wrap_modules,
                    key=lambda wrap_module: wrap_module.fqn.count("."),
                ).module
            else:
                owner_by_parameter[parameter] = model
        return owner_by_parameter

    def _get_default_source_shard_info(self) -> "fully_shard_utils.SourceShardMetaInfo":
        """Build all-Replicate source metadata covering every non-FSDP mesh axis.

        Used to complete source_shard_info for parameters the plan does not mention;
        the axis set must match the per-parameter entries so that all parameters
        in one ``fully_shard`` unit share the same source-mesh dimensionality.
        """
        world_mesh = self.mesh_context.fsdp_non_moe_mesh or self.mesh_context.device_mesh
        if world_mesh is None or world_mesh.mesh_dim_names is None:
            raise ValueError("TP metadata completion requires a named world mesh")
        source_dim_names = tuple(
            dim_name
            for dim_name in world_mesh.mesh_dim_names
            if dim_name not in FSDP_OWNED_DIMS
        )
        if not source_dim_names:
            raise ValueError(
                "TP is enabled but the world mesh has no non-FSDP source dimension"
            )
        source_mesh = (
            world_mesh
            if tuple(world_mesh.mesh_dim_names) == source_dim_names
            else world_mesh[source_dim_names[0]]
            if len(source_dim_names) == 1
            else world_mesh[source_dim_names]
        )
        return fully_shard_utils.SourceShardMetaInfo(  # pylint: disable=no-member
            mesh=source_mesh,
            placements=tuple(Replicate() for _ in source_dim_names),
            origin_is_dtensor=False,
        )

    def _uses_expert_mesh(self, metadata_by_parameter: SourceShardInfoByParam | None) -> bool:
        """Return whether one FSDP unit owns routed-expert source metadata."""
        if metadata_by_parameter is None or self.mesh_context.fsdp_moe_mesh is None:
            return False
        return any(
            _is_expert_source_mesh(source_shard_info.mesh)
            for source_shard_info in metadata_by_parameter.values()
        )

    def _build_managed_source_shard_info(
        self,
        owner: ModuleClass,
        owner_by_parameter: Mapping[ParameterClass, ModuleClass],
        metadata_by_parameter: SourceShardInfoByParam | None,
    ) -> SourceShardInfoByParam | None:
        """Select and complete metadata for parameters managed by one wrap call."""
        if metadata_by_parameter is None:
            return None

        default_source_shard_info = None
        if (
            self.mesh_context.fsdp_non_moe_mesh is not None
            or self.mesh_context.device_mesh is not None
        ):
            default_source_shard_info = self._get_default_source_shard_info()
        managed_source_shard_info = {}
        for parameter, parameter_owner in owner_by_parameter.items():
            if parameter_owner is not owner:
                continue
            source_shard_info = metadata_by_parameter.get(parameter)
            if source_shard_info is None:
                if default_source_shard_info is None:
                    raise ValueError(
                        "source_shard_info is present but does not cover an FSDP-managed parameter"
                    )
                source_shard_info = default_source_shard_info
            managed_source_shard_info[parameter] = source_shard_info
        return managed_source_shard_info

    @staticmethod
    def _build_managed_replicate_params(
        owner: ModuleClass,
        owner_by_parameter: Mapping[ParameterClass, ModuleClass],
        replicate_params: set[ParameterClass] | None,
    ) -> set[ParameterClass] | None:
        """Select configured replicate parameters owned by one FSDP module."""
        if replicate_params is None:
            return None
        return {
            parameter
            for parameter, parameter_owner in owner_by_parameter.items()
            if parameter_owner is owner and parameter in replicate_params
        }

    def _configure_source_layout_gradient_scaling(
        self,
        hsdp_module: HSDPModule,
        managed_source_shard_info: SourceShardInfoByParam | None,
    ) -> bool:
        """Restore global-mean gradients for one source-layout FSDP unit.

        ``fully_shard`` defaults units whose parameters all have source-layout
        metadata to SUM reduction. The dual-mode Trainer's token-weighted loss
        is multiplied by ``dp_size`` to compensate normal FSDP AVG reduction,
        so a SUM-reduced unit must apply the inverse factor exactly once.

        Returns:
            Whether a gradient scaling factor was configured for the unit.
        """
        dp_size = self.mesh_context.dp_size
        if dp_size <= 1 or not managed_source_shard_info:
            return False
        hsdp_module.set_gradient_scaling_factor(1.0 / dp_size)
        return True

    def _configure_prefetch(self, hsdp_modules: list[HSDPModule]) -> None:
        """Configure forward and backward prefetch targets by list depth."""
        for module_index, hsdp_module in enumerate(hsdp_modules):
            forward_end = module_index + self.config.forward_prefetch_depth + 1
            hsdp_module.set_modules_to_forward_prefetch(
                hsdp_modules[module_index + 1:forward_end]
            )

            backward_start = max(
                0, module_index - self.config.backward_prefetch_depth
            )
            hsdp_module.set_modules_to_backward_prefetch(
                list(reversed(hsdp_modules[backward_start:module_index]))
            )

    @staticmethod
    def _order_wrap_modules(
        model: ModuleClass,
        wrap_modules: list[_WrapModuleInfo],
    ) -> list[_WrapModuleInfo]:
        """Order FSDP units according to module traversal/forward declaration order."""
        module_order = {
            id(module): module_index
            for module_index, (_, module) in enumerate(model.named_modules())
        }
        return sorted(
            wrap_modules,
            key=lambda wrap_module: module_order.get(id(wrap_module.module), len(module_order)),
        )

    def parallelize(
        self,
        model: ModuleClass,
        source_shard_info: SourceShardInfoByFQN | None = None,
        *,
        compile_hooks_enabled: bool = False,
    ) -> ModuleClass:
        """Apply configured child FSDP units, root FSDP, and prefetch links.

        Args:
            model: Model to wrap in place.
            source_shard_info: FQN-keyed source-layout metadata returned by
                ``apply_sharding_plan``. The manager resolves parameter
                identities before applying nested FSDP units.
            compile_hooks_enabled: Whether scheduler forward hooks should stay
                outside Dynamo capture for decoder-layer compilation.

        Returns:
            The model enhanced with the HyperParallel HSDPModule interface.

        Raises:
            ValueError: If TP is enabled without FQN metadata or configured
                parameter FQNs cannot be resolved.
        """
        if not isinstance(compile_hooks_enabled, bool):
            raise ValueError(
                "compile_hooks_enabled must be a bool, got "
                f"{type(compile_hooks_enabled).__name__}"
            )
        metadata_by_parameter = self._build_source_shard_info_by_param(
            model,
            source_shard_info,
        )
        replicate_params = self._resolve_replicate_params(model)

        wrap_modules = self._find_wrap_modules(model, metadata_by_parameter)
        owner_by_parameter = self._resolve_parameter_owners(model, wrap_modules)
        dense_fsdp_kwargs, dense_root_kwargs = self._build_fully_shard_kwargs(
            self._build_fsdp_actual_mesh()
        )

        wrapping_order = sorted(
            wrap_modules,
            key=lambda wrap_module: wrap_module.fqn.count("."),
            reverse=True,
        )
        gradient_scaled_units = 0
        for wrap_module in wrapping_order:
            managed_source_shard_info = self._build_managed_source_shard_info(
                wrap_module.module,
                owner_by_parameter,
                metadata_by_parameter,
            )
            fsdp_sublayer_kwargs = dense_fsdp_kwargs
            if self._uses_expert_mesh(managed_source_shard_info):
                if self.mesh_context.fsdp_moe_mesh is None:
                    raise ValueError("Expert TP metadata requires MeshContext.fsdp_moe_mesh")
                fsdp_sublayer_kwargs, _ = self._build_fully_shard_kwargs(
                    self._build_fsdp_actual_mesh(expert=True)
                )
            fully_shard(  # pylint: disable=unexpected-keyword-arg
                wrap_module.module,
                source_shard_infos=managed_source_shard_info,
                compile_hooks_enabled=compile_hooks_enabled,
                replicate_params=self._build_managed_replicate_params(
                    wrap_module.module,
                    owner_by_parameter,
                    replicate_params,
                ),
                **fsdp_sublayer_kwargs,
            )
            if self._configure_source_layout_gradient_scaling(
                wrap_module.module,
                managed_source_shard_info,
            ):
                gradient_scaled_units += 1

        root_source_shard_info = self._build_managed_source_shard_info(
            model,
            owner_by_parameter,
            metadata_by_parameter,
        )
        if self._uses_expert_mesh(root_source_shard_info):
            raise ValueError(
                "Routed expert parameters must belong to a nested experts FSDP unit"
            )
        fully_shard(  # pylint: disable=unexpected-keyword-arg
            model,
            source_shard_infos=root_source_shard_info,
            compile_hooks_enabled=compile_hooks_enabled,
            replicate_params=self._build_managed_replicate_params(
                model,
                owner_by_parameter,
                replicate_params,
            ),
            **dense_root_kwargs,
        )
        if self._configure_source_layout_gradient_scaling(model, root_source_shard_info):
            gradient_scaled_units += 1
        if gradient_scaled_units:
            logger.info(
                "Applied DP gradient scaling factor %s to %d source-layout FSDP2 units",
                1.0 / self.mesh_context.dp_size,
                gradient_scaled_units,
            )
        ordered_wrap_modules = self._order_wrap_modules(model, wrap_modules)
        if self.config.enable_fsdp2_prefetch:
            self._configure_prefetch(
                [wrap_module.module for wrap_module in ordered_wrap_modules]
            )
        logger.info(
            "Applied FSDP2 to %d transformer blocks and the root module",
            len(wrap_modules),
        )
        return model


def _instantiate_fsdp2(*, config: Any, mesh_context: MeshContext) -> FSDP2Manager | None:
    """Create an FSDP2 manager for the resolved strategy config."""
    if config is None:
        return None
    if isinstance(config, FSDP2Config):
        return FSDP2Manager(config=config, mesh_context=mesh_context)
    raise ValueError(f"FSDP2Manager requires FSDP2Config, got {type(config).__name__}")
