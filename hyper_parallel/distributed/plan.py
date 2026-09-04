# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""plan: the model-level sharding plan (05 §3.1 canonical).

:class:`ShardingPlan` is the single authoritative model-level plan of
AutoModels — it is NOT a port of the legacy ``core/shard/sharding_plan.py``
(the two data models are incompatible and must never import each other).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hyper_parallel.distributed.recipe_spec import (
    ModuleShardingSpec,
    NamedPlacement,
)


@dataclass
class ShardingPlan:
    """Complete sharding plan for a model (05 §3.1)."""
    # {module_fqn: ModuleShardingSpec} — contains only modules with is_boundary=True
    modules: Dict[str, ModuleShardingSpec] = field(default_factory=dict)

    # Global switches
    sequence_parallel: bool = True
    loss_parallel: bool = False

    # Special parameter handlers: {module_fqn.param_name: handler_name}
    special_handlers: Dict[str, str] = field(default_factory=dict)

    # Mesh dimension names (consistent with DeviceMesh.mesh_dim_names)
    mesh_dim_names: Tuple[str, ...] = ()

    # Tied-weight pairs: [(fqn_a, fqn_b)], parameters sharing storage
    # (embed_tokens <-> lm_head).
    tied_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def _format_named_placement(self, named: Optional[NamedPlacement]) -> str:
        """Format a named placement as a compact one-line dictionary."""
        if not named:
            return "{}"
        items = [
            (getattr(axis, "value", axis), placement)
            for axis, placement in named.items()
            if not self.mesh_dim_names or getattr(axis, "value", axis) in self.mesh_dim_names
        ]
        return "{" + ", ".join(f"{axis}: {placement!r}" for axis, placement in items) + "}"

    @staticmethod
    def _format_injection_callable(obj: Any) -> str:
        """Format a callable, Target, or registry name for display."""
        if obj is None:
            return "-"
        path = getattr(obj, "_target_path", None)
        if path:
            return path
        if isinstance(obj, str):
            return obj
        return getattr(obj, "__qualname__", repr(obj))

    def _append_parameter_explanation(self, lines: List[str], spec: ModuleShardingSpec) -> None:
        """Append parameter, TP-local attribute, and deferred-bias details."""
        if spec.params:
            lines.append("  parameter sharding:")
            for param_name, named in spec.params.items():
                lines.append(f"    {param_name}: {self._format_named_placement(named)}")
        else:
            lines.append("  parameter sharding: none ({} = this boundary shards no parameters, I/O stitching only)")
        attr_plan = spec._tp_local_attr_plan  # pylint: disable=protected-access
        if attr_plan is not None and (attr_plan.auto_divide or attr_plan.user_divide):
            lines.append("  TP-local attribute division:")
            if attr_plan.auto_divide:
                lines.append("    auto(D-17): " + ", ".join(attr_plan.auto_divide))
            if attr_plan.user_divide:
                lines.append("    user(plan_overrides): " + ", ".join(attr_plan.user_divide))
        if spec._deferred_bias_params:  # pylint: disable=protected-access
            lines.append(
                "  deferred bias (D-22, no bias inside the region, added exactly once after the TP reduction): "
                + ", ".join(spec._deferred_bias_params)  # pylint: disable=protected-access
            )

    def _append_communication_explanation(self, lines: List[str], spec: ModuleShardingSpec) -> None:
        """Append compiled input and output redistribution operations."""
        # Lazy import keeps plan.py import-light; PrecompiledBoundary lives
        # in _builder since changeset 4c.
        from hyper_parallel.distributed._builder.precompiled_boundary import (  # pylint: disable=C0415
            PrecompiledBoundary,
        )

        boundary = PrecompiledBoundary(spec, None, self.mesh_dim_names)
        if boundary.in_plan:
            lines.append("  input communication plan (in_src -> in_dst):")
            for op in boundary.in_plan:
                tag = "passthrough" if op.collective_type == "identity" else op.collective_type
                lines.append(
                    f"    {op.arg_name}: {tuple(map(repr, op.src_placements))} -> "
                    f"{tuple(map(repr, op.dst_placements))}  [{tag}]"
                )
        if boundary.out_plan:
            lines.append("  output communication plan (out_src -> out_dst):")
            for op in boundary.out_plan:
                lines.append(
                    f"    {op.arg_name}(tuple[{op.arg_index}]): {tuple(map(repr, op.src_placements))} -> "
                    f"{tuple(map(repr, op.dst_placements))}  [{op.collective_type}]"
                )
        if not boundary.in_plan and not boundary.out_plan:
            lines.append("  boundary communication: none")

    def _append_injection_explanation(self, lines: List[str], spec: ModuleShardingSpec) -> None:
        """Append compute-injection declarations and dispatch semantics."""
        injection = []
        if spec.local_compute_fn is not None:
            injection.append(f"local_compute_fn={self._format_injection_callable(spec.local_compute_fn)}")
        if spec.inner_wrapper is not None:
            injection.append(
                f"inner_wrapper={self._format_injection_callable(spec.inner_wrapper)}"
                f"(target={spec.inner_target or 'auto'})"
            )
        if spec.inner_out_src is not None:
            injection.append(f"inner_out_src={spec.inner_out_src}")
        if spec.region_dispatch is not None:
            meaning = (
                "black-box managed (propagation check skipped inside the region, declarative re-wrap)"
                if spec.region_dispatch is False
                else "dispatch-through (real validation under validate is enabled)"
            )
            injection.append(f"region_dispatch={spec.region_dispatch} -> {meaning}")
        if injection:
            lines.append("  injection: " + "; ".join(injection))
        else:
            lines.append("  injection: none (ordinary boundary, dispatch-through under validate)")

    def _append_handler_explanation(self, lines: List[str], boundary_fqn: str) -> None:
        """Append special parameter handlers belonging to one boundary."""
        handlers = {
            key: value for key, value in self.special_handlers.items() if key.startswith(boundary_fqn + ".")
        }
        for key, handler in handlers.items():
            lines.append(f"  special handling: {key[len(boundary_fqn) + 1:]} -> {handler}")

    def explain(self, fqn: Optional[str] = None) -> str:
        """Human-readable introspection report of this plan (usability tool).

        Per boundary: the parameter sharding table (param → placement), the
        compiled boundary communication plan (in/out RedistOps — which
        tensor, from which layout to which, which collective), the injection
        declarations and their resolution result, and any special-handler
        entries. All of this information already lives in the plan/specs —
        this is purely a formatting outlet. The intended learning path:
        read your own model's actual sharding first, then reverse-engineer
        the concepts — instead of building the layout mental model from
        docs up front.

        Args:
            fqn: optional exact boundary FQN — report just that boundary;
                None reports all boundaries.
        """
        lines = [
            "=== ShardingPlan introspection report ===",
            f"mesh_dim_names={self.mesh_dim_names}  "
            f"sequence_parallel={self.sequence_parallel}  "
            f"loss_parallel={self.loss_parallel}",
            f"boundaries: {len(self.modules)}  tied_pairs: "
            + (", ".join(f"{a}<->{b}" for a, b in self.tied_pairs) or "none"),
        ]
        if fqn is not None and fqn not in self.modules:
            lines.append(
                f"\n[!] {fqn!r} is not a boundary of this plan "
                "(existing boundaries listed above)")
            return "\n".join(lines)
        selected = (
            {fqn: self.modules[fqn]} if fqn is not None else self.modules)

        for name, spec in selected.items():
            lines.append(f"\n[{name}]")
            self._append_parameter_explanation(lines, spec)
            self._append_communication_explanation(lines, spec)
            self._append_injection_explanation(lines, spec)
            self._append_handler_explanation(lines, name)
        return "\n".join(lines)
