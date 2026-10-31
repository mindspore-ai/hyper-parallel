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
"""Codegen-only additions to native sharding-plan discovery."""

from contextlib import contextmanager
from fnmatch import fnmatchcase
from typing import Any, Iterator, Optional

from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed._builder.default_templates import _multi_dim
from hyper_parallel.distributed._builder.planner import ShardingPlanner
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec, NamedPlacement
from hyper_parallel.distributed.tensor_parallel.param_role import ParamRole
from hyper_parallel.models._transformers import auto_model


_CODEGEN_LEAF_SEGMENTS = frozenset({"linear_fc1", "linear_fc2"})


def _override_declared_params(plan_overrides: dict[str, Any], model: Any) -> set[str]:
    """Return parameter FQNs already owned by exact or glob overrides."""
    module_names = tuple(name for name, _ in model.named_modules())
    declared = set()
    for key, spec in plan_overrides.items():
        matches = (
            (name for name in module_names if fnmatchcase(name, key))
            if any(char in key for char in "*?[")
            else (key,)
        )
        for module_fqn in matches:
            prefix = f"{module_fqn}." if module_fqn else ""
            declared.update(prefix + name for name in (spec.params or {}))
    return declared


class CodegenShardingPlanner(ShardingPlanner):
    """Add audit-only coverage without changing the Trainer's native planner."""

    def _infer_boundary_type(self, fqn: str, group: Any) -> str:
        """Let Megatron-style projection leaves merge into their MLP owner."""
        if fqn.rsplit(".", 1)[-1].lower() in _CODEGEN_LEAF_SEGMENTS:
            return "unknown"
        return super()._infer_boundary_type(fqn, group)

    def _group_by_boundary(self, param_roles: dict[str, ParamRole]) -> dict[str, list[tuple[str, ParamRole]]]:
        """Keep unknown root leftovers attached to each real parameter owner."""
        groups = super()._group_by_boundary(param_roles)
        normalized: dict[str, list[tuple[str, ParamRole]]] = {}
        for module_fqn, group in groups.items():
            if self._infer_boundary_type(module_fqn, group) != "unknown":
                normalized[module_fqn] = group
                continue
            for param_fqn, role in group:
                owner = param_fqn.rsplit(".", 1)[0]
                normalized.setdefault(owner, []).append((param_fqn, role))
        return normalized

    def _derive_boundary_specs(self, plan: Any, boundary_groups: Any, **kwargs: Any) -> None:
        """Append explicit replicated audit entries after native derivation."""
        super()._derive_boundary_specs(plan, boundary_groups, **kwargs)
        if not self._derive:
            return
        model = kwargs["model"]
        parameters_by_name = dict(model.named_parameters())
        override_declared = _override_declared_params(self._plan_overrides, model)
        for boundary_fqn, group in boundary_groups.items():
            if boundary_fqn in plan.modules:
                continue
            fallback = self._build_replicated_fallback_spec(
                boundary_fqn,
                group,
                parameters_by_name,
                override_declared,
            )
            if fallback is not None:
                plan.modules[boundary_fqn] = fallback

    @staticmethod
    def _build_replicated_fallback_spec(
        boundary_fqn: str,
        group: list[tuple[str, ParamRole]],
        parameters_by_name: dict[str, Any],
        override_declared: set[str],
    ) -> Optional[ModuleShardingSpec]:
        """Declare communication-free metadata for replicated leftovers."""
        replicable = (ParamRole.REPLICATED, ParamRole.NORM, ParamRole.BIAS)
        if any(role not in replicable for _, role in group):
            return None
        prefix = f"{boundary_fqn}." if boundary_fqn else ""
        params: dict[str, NamedPlacement] = {}
        for param_fqn, _ in group:
            parameter = parameters_by_name.get(param_fqn)
            if parameter is None or not parameter.requires_grad or param_fqn in override_declared:
                continue
            relative = param_fqn[len(prefix):] if prefix else param_fqn
            params[relative] = _multi_dim(tp=Replicate(), cp=Replicate(), ep=Replicate())
        if not params:
            return None
        return ModuleShardingSpec(params=params, in_src={}, in_dst={}, is_boundary=False)


@contextmanager
def use_codegen_planner_overlay() -> Iterator[None]:
    """Upgrade planners created inside this single-process codegen build."""
    original = auto_model.instantiate_infrastructure

    def instantiate_infrastructure(*args: Any, **kwargs: Any) -> Any:
        """Upgrade only an exact native planner created inside this context."""
        planner, fsdp_manager = original(*args, **kwargs)
        if type(planner) is ShardingPlanner:  # pylint: disable=unidiomatic-typecheck
            planner.__class__ = CodegenShardingPlanner
        return planner, fsdp_manager

    auto_model.instantiate_infrastructure = instantiate_infrastructure
    try:
        yield
    finally:
        auto_model.instantiate_infrastructure = original


__all__ = ["CodegenShardingPlanner", "use_codegen_planner_overlay"]
