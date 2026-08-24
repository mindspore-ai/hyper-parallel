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
"""precompiled_boundary: compile-time communication planning (05 §4.3).

RedistOp / PrecompiledBoundary compile the placement differences of
in_src→in_dst and out_src→out_dst into sequences of RedistOps. An optional
lowerer may attach local-tensor execution operations while the boundary keeps
the generic DTensor redistribution fallback.

API adaptation (differences from the 05 doc pseudocode; actual in-house
DTensor signatures):
- ``DTensor.from_local(local, mesh, placements)``: no run_check parameter;
- ``dt.redistribute(mesh, placements)``: mesh is the first argument, no async_op.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Placement, Shard
from hyper_parallel.auto_models.components.distributed.sharding_config import resolve_placements

logger = logging.getLogger(__name__)


def _classify_collective(src, dst) -> str:
    """Derive the communication type from placements (a debug/profiling label,
    not a communication-path selector).

    Only differing dimensions are compared. Identity placement dimensions do
    not participate in classification, so a Shard-to-Replicate transition is
    classified as all_gather.
    """
    if tuple(src) == tuple(dst):
        return "identity"
    diff_src = tuple(s for s, d in zip(src, dst) if s != d)
    diff_dst = tuple(d for s, d in zip(src, dst) if s != d)

    has_shard_src = any(isinstance(p, Shard) for p in diff_src)
    has_partial_src = any(isinstance(p, Partial) for p in diff_src)
    has_shard_dst = any(isinstance(p, Shard) for p in diff_dst)
    all_replicate_dst = all(
        not isinstance(p, (Shard, Partial)) for p in diff_dst
    )

    if has_partial_src and has_shard_dst:
        return "reduce_scatter"
    if has_partial_src and all_replicate_dst:
        return "all_reduce"
    if has_shard_src and all_replicate_dst:
        return "all_gather"
    return "redistribute"


def _get_arg(args, kwargs, name, idx, default=None):
    if name in kwargs:
        return kwargs[name]
    if idx is not None and idx < len(args):
        return args[idx]
    return default


def _set_arg(args, kwargs, name, idx, value):
    if name in kwargs:
        kwargs[name] = value
        return args, kwargs
    if idx is not None and idx < len(args):
        args = list(args)
        args[idx] = value
        return tuple(args), kwargs
    kwargs[name] = value
    return args, kwargs


class BoundaryExecutionOp(Protocol):
    """Operation attached by a boundary lowerer and executed on local tensors."""

    def execute(self, tensor: Any) -> Any:
        """Execute one pre-lowered placement transition."""


BoundaryOpLowerer = Callable[
    [Sequence[Placement], Sequence[Placement]],
    Optional[BoundaryExecutionOp],
]


@dataclass
class RedistOp:
    """A single precompiled redistribute operation (05 §4.3.1).

    ``collective_type`` is a debug/profiling label. Validation and transitions
    rejected by the injected lowerer use ``DTensor.redistribute()``; accepted
    transitions use ``execution_op`` directly on local tensors.
    """
    arg_name: str
    arg_index: Optional[int]
    mesh: object  # DeviceMesh
    src_placements: Tuple[Placement, ...]
    dst_placements: Tuple[Placement, ...]
    collective_type: str
    execution_op: Optional[BoundaryExecutionOp] = None

    def execute(self, tensor: torch.Tensor, *, as_dtensor: bool = False) -> Any:
        """Execute the communication.

        Args:
            tensor: input local tensor (or DTensor).
            as_dtensor: True → return a DTensor (validate mode), False → return a local tensor.
        """
        if self.collective_type == "identity":
            if isinstance(tensor, DTensor):
                # validate (as_dtensor=True) keeps the DTensor; production
                # returns local — an identity op's input may come from a
                # from_local re-wrap in a local region (MoE/CP wrapper), so
                # the boundary exit must unwrap it.
                return tensor if as_dtensor else tensor.to_local()
            if as_dtensor:
                return DTensor.from_local(
                    tensor, self.mesh, tuple(self.src_placements))
            return tensor

        if not as_dtensor and self.execution_op is not None:
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            return self.execution_op.execute(local_tensor)

        # Unified path: zero-copy wrap → redistribute → optional to_local
        if isinstance(tensor, DTensor):
            dt = tensor
        else:
            dt = DTensor.from_local(tensor, self.mesh, tuple(self.src_placements))
        dt = dt.redistribute(self.mesh, tuple(self.dst_placements))
        return dt if as_dtensor else dt.to_local()


class PrecompiledBoundary:
    """Compile-time communication plan (05 §4.3.3): two RedistOp sequences, in_plan/out_plan."""

    def __init__(
        self,
        spec: Any,
        mesh: Any,
        mesh_dim_names: Sequence[str],
        *,
        op_lowerer: Optional[BoundaryOpLowerer] = None,
    ) -> None:
        """Compile the in/out RedistOp plans for one module boundary.

        Args:
            spec: ModuleShardingSpec carrying the in_src/in_dst/out_src/out_dst
                placement contracts.
            mesh: DeviceMesh the redistributions execute on.
            mesh_dim_names: Ordered mesh dimension names used to resolve
                placement shorthands.
            op_lowerer: Optional lowerer that turns a (src, dst) placement
                transition into a local-tensor execution op.
        """
        self.spec = spec
        self.mesh = mesh
        self.mesh_dim_names = tuple(mesh_dim_names)
        self._op_lowerer = op_lowerer
        self.in_plan = self._compile_input_plan(spec, mesh, self.mesh_dim_names)
        self.out_plan = self._compile_output_plan(spec, mesh, self.mesh_dim_names)

    def _lower_execution_op(self, src, dst) -> Optional[BoundaryExecutionOp]:
        """Delegate optional execution lowering without owning backend details."""
        return self._op_lowerer(src, dst) if self._op_lowerer is not None else None

    # ── Compilation ─────────────────────────────────────────────────────

    def _compile_input_plan(self, spec, mesh, mesh_dim_names):
        """Compile the input communication plan from in_src → in_dst (identity
        dimensions naturally compile to pass-through ops)."""
        plan = []
        in_src = spec.in_src or {}   # tolerate None: a hand-written spec (debug
        in_dst = spec.in_dst or {}   # shortcut) may not be normalized by the plan
        # (the input-side semantics of "inherit when not declared")
        all_names = set(in_src.keys()) | set(in_dst.keys())
        for name in sorted(all_names):
            src_p = tuple(resolve_placements(
                in_src.get(name, {}), mesh_dim_names))
            dst_p = tuple(resolve_placements(
                in_dst.get(name, {}), mesh_dim_names))
            plan.append(RedistOp(
                arg_name=name,
                arg_index=None,
                mesh=mesh,
                src_placements=src_p,
                dst_placements=dst_p,
                collective_type=_classify_collective(src_p, dst_p),
                execution_op=self._lower_execution_op(src_p, dst_p),
            ))
        return plan

    def _compile_output_plan(self, spec, mesh, mesh_dim_names):
        """Compile the output communication plan from out_src → out_dst (identity
        skipped, multi-output supported).

        arg_index source priority: (1) explicit order in spec.out_names;
        (2) key order of out_src.
        out_src=None or out_dst=None → nothing is compiled.
        """
        if spec.out_src is None or spec.out_dst is None:
            return []

        out_names = getattr(spec, "out_names", None) or list(spec.out_src.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}

        plan = []
        all_names = set(spec.out_src.keys()) | set(spec.out_dst.keys())
        for name in sorted(all_names):
            src_p = tuple(resolve_placements(
                spec.out_src.get(name, {}), mesh_dim_names))
            dst_p = tuple(resolve_placements(
                spec.out_dst.get(name, {}), mesh_dim_names))
            if src_p == dst_p:
                continue  # identity, no communication needed
            plan.append(RedistOp(
                arg_name=name,
                arg_index=name_to_idx.get(name, 0),
                mesh=mesh,
                src_placements=src_p,
                dst_placements=dst_p,
                collective_type=_classify_collective(src_p, dst_p),
                execution_op=self._lower_execution_op(src_p, dst_p),
            ))
        return plan

    # ── Runtime execution ───────────────────────────────────────────────

    def redistribute_inputs(
        self,
        args: Sequence[Any],
        kwargs: dict,
        *,
        as_dtensor: bool = False,
    ) -> Tuple[Sequence[Any], dict]:
        """Execute input redistribution. as_dtensor=True → return DTensors (validate mode).

        When an arg is not found in args/kwargs (None) the op is skipped —
        e.g. embed's in_src key "input" differs from the actual kwargs name
        "input_ids" and is identity.
        """
        for op in self.in_plan:
            arg = _get_arg(args, kwargs, op.arg_name, op.arg_index, default=None)
            if arg is None:
                continue
            result = op.execute(arg, as_dtensor=as_dtensor)
            args, kwargs = _set_arg(args, kwargs, op.arg_name, op.arg_index, result)
        return args, kwargs

    def redistribute_outputs(self, outputs: Any, *, as_dtensor_input: bool = False) -> Any:
        """Execute output redistribution (single Tensor output / multi-output
        tuple, order preserved, same structure returned).

        as_dtensor_input=True → inputs are already DTensors (validate mode).
        """
        is_sequence = isinstance(outputs, (tuple, list))
        outputs_list = list(outputs) if is_sequence else [outputs]
        for op in self.out_plan:
            idx = op.arg_index if op.arg_index is not None else 0
            if idx >= len(outputs_list):
                logger.warning(
                    "PrecompiledBoundary: out_plan expects output '%s' at index %d, "
                    "but module returned only %d outputs. Skipping.",
                    op.arg_name, idx, len(outputs_list),
                )
                continue
            tensor = outputs_list[idx]
            if tensor is None:
                continue
            # as_dtensor_input=True (validate) → keep DTensors for out_dst
            # validation; otherwise return local (production / final boundary exit).
            outputs_list[idx] = op.execute(tensor, as_dtensor=as_dtensor_input)
        if isinstance(outputs, tuple):
            return tuple(outputs_list)
        if isinstance(outputs, list):
            return outputs_list
        return outputs_list[0]
