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
"""Bind a native live plan to source-materialized generated forwards."""

from __future__ import annotations

import functools
from typing import Any

from hyper_parallel.distributed._builder.applier import (
    _boundary_post_order_key,
    _get_cp_submesh,
    _get_tp_submesh,
)
from hyper_parallel.distributed._builder.forward_rewriter import (
    _bind_input_indices,
    _descendant_boundary_fqns,
    _install_bias_suppression,
    _is_vocab_parallel_embed,
    _keep_loss_parallel_output,
    _maybe_add_deferred_biases,
    _rewrap_local_outputs,
    _wrap_inner_attention,
    _wrap_local_region_forward,
    _wrap_production_forward,
    _wrap_vocab_parallel_embedding,
)
from hyper_parallel.distributed._builder.parameter_sharding import _resolve_module
from hyper_parallel.distributed._builder.precompiled_boundary import PrecompiledBoundary
from hyper_parallel.distributed._builder.rule_resolver import (
    _resolve_inner_target,
    _resolve_local_compute_fn,
)
from hyper_parallel.distributed._builder.tp_collective_lowering import create_tp_collective_lowerer


class _GeneratedTPOperators:
    """Expose emitted bare TP operations through the native live mesh."""

    def __init__(self, lowerer: Any) -> None:
        """Store the native TP collective lowerer."""
        self._lowerer = lowerer

    @staticmethod
    def to_local(tensor: Any) -> Any:
        """Return a local tensor when a DTensor-like value is supplied."""
        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

    def all_gather(self, tensor: Any, dim: int) -> Any:
        """Run differentiable TP all-gather."""
        return self._execute("all_gather", tensor, tensor_dim=dim)

    def all_reduce(self, tensor: Any) -> Any:
        """Run differentiable TP sum all-reduce."""
        return self._execute("all_reduce", tensor)

    def reduce_scatter(self, tensor: Any, dim: int) -> Any:
        """Run differentiable TP sum reduce-scatter."""
        return self._execute("reduce_scatter", tensor, tensor_dim=dim)

    def _execute(self, kind: str, tensor: Any, tensor_dim: int | None = None) -> Any:
        if self._lowerer is None:
            raise RuntimeError(f"generated TP operation {kind!r} requires an active tp mesh")
        operation = self._lowerer.execution_op(kind, tensor_dim=tensor_dim)
        return operation.execute(tensor)


class _GeneratedBoundary:
    """Adapt emitted boundary calls to the native compiled boundary API."""

    def __init__(
        self,
        compiled: PrecompiledBoundary,
        spec: Any,
        mesh: Any,
        mesh_dim_names: Any,
        label: str,
    ) -> None:
        """Store the live-plan objects used by emitted boundary operations."""
        self.compiled = compiled
        self._spec = spec
        self._mesh = mesh
        self._mesh_dim_names = mesh_dim_names
        self._label = label

    def redistribute_inputs(
        self,
        args: Any,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Accept the emitted packed call and delegate native redistribution."""
        if kwargs is None:
            args, kwargs = args
        return self.compiled.redistribute_inputs(args, kwargs)

    def rewrap_outputs(self, outputs: Any) -> Any:
        """Rewrap local outputs with their live-plan source placements."""
        return _rewrap_local_outputs(
            outputs,
            self._spec,
            self._mesh,
            self._mesh_dim_names,
            self._label,
        )

    def redistribute_outputs(self, outputs: Any) -> Any:
        """Delegate output redistribution to the native compiled boundary."""
        return self.compiled.redistribute_outputs(outputs)


_MISSING = object()


def _restore_generated_forward(module: Any, saved_forward: Any) -> None:
    """Restore a class-owned generated forward after runtime state setup."""
    if saved_forward is _MISSING:
        module.__dict__.pop("forward", None)
    else:
        module.forward = saved_forward


def _has_generated_boundary_forward(module: Any) -> bool:
    """Detect an emitted boundary from behavior markers, not class names."""
    if getattr(type(module), "_hp_boundary_form", None) is not None:
        return True
    forward = getattr(type(module), "forward", None)
    code = getattr(forward, "__code__", None)
    return code is not None and bool({"_hp_boundary", "_hp_tp"} & set(code.co_names))


def _configure_inner_wrapper(
    module: Any,
    spec: Any,
    *,
    cp_mesh: Any,
    mesh: Any,
    mesh_dim_names: Any,
    tp_mesh: Any,
    expert_mesh: Any,
    module_fqn: str,
) -> None:
    """Install wrapper companion state while retaining an emitted forward."""
    if getattr(spec, "inner_wrapper", None) is None:
        return
    target = _resolve_inner_target(module, spec)
    generated_toggle = hasattr(target, "cp_enable")
    saved_forward = target.__dict__.get("forward", _MISSING)
    _wrap_inner_attention(
        module,
        cp_mesh,
        spec=spec,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
        tp_mesh=tp_mesh,
        ep_mesh=expert_mesh,
        validate_mode=False,
        module_fqn=module_fqn,
    )
    if generated_toggle:
        _restore_generated_forward(target, saved_forward)
        target.cp_enable = cp_mesh is not None


def _bind_compute_method(module: Any, spec: Any, compute_fn: Any) -> None:
    """Publish the native resolved compute callable under its emitted name."""
    factory = getattr(spec, "local_compute_fn", None)
    if hasattr(factory, "_target_"):
        factory = factory._target_  # pylint: disable=protected-access
    name = getattr(factory, "__name__", None)
    if not name:
        raise RuntimeError(
            f"generated local compute for {type(module).__name__} has no stable function name"
        )
    setattr(module, name, compute_fn)


def apply_generated_phase_c(
    model: Any,
    plan: Any,
    mesh: Any,
    validate_mode: bool,
    expert_mesh: Any = None,
) -> None:
    """Bind live-plan runtime state without rewriting generated forwards.

    Args:
        model: Generated model instance whose source already owns forward forms.
        plan: Plan freshly derived by the native planner.
        mesh: Native device mesh.
        validate_mode: Placement-validation mode requested by the caller.
        expert_mesh: Native expert mesh derived during plan application.

    Raises:
        ValueError: If validation mode is requested for a generated training entry.
    """
    if validate_mode:
        raise ValueError("generated model training does not support validate_placement mode")
    mesh_dim_names = plan.mesh_dim_names
    cp_mesh = _get_cp_submesh(mesh, mesh_dim_names)
    tp_mesh = _get_tp_submesh(mesh, mesh_dim_names)
    lowerer = create_tp_collective_lowerer(mesh, mesh_dim_names)
    tp_operators = _GeneratedTPOperators(lowerer)

    for module_fqn, spec in sorted(plan.modules.items(), key=_boundary_post_order_key):
        if not spec.is_boundary:
            continue
        module = _resolve_module(model, module_fqn)
        boundary = PrecompiledBoundary(spec, mesh, mesh_dim_names, op_lowerer=lowerer)
        _bind_input_indices(boundary, module)
        module._hp_boundary = _GeneratedBoundary(  # pylint: disable=protected-access
            boundary, spec, mesh, mesh_dim_names, module_fqn,
        )
        module._hp_tp = tp_operators  # pylint: disable=protected-access
        if spec._deferred_bias_params:  # pylint: disable=protected-access
            _install_bias_suppression(module, spec)
            module._hp_deferred_bias = functools.partial(  # pylint: disable=protected-access
                _maybe_add_deferred_biases, module, spec,
            )

        _configure_inner_wrapper(
            module,
            spec,
            cp_mesh=cp_mesh,
            mesh=mesh,
            mesh_dim_names=mesh_dim_names,
            tp_mesh=tp_mesh,
            expert_mesh=expert_mesh,
            module_fqn=module_fqn,
        )

        compute_fn = _resolve_local_compute_fn(
            module, spec, mesh, mesh_dim_names, expert_mesh,
        )
        generated_local_compute = (
            getattr(spec, "local_compute_fn", None) is not None
            and hasattr(module, "ep_enable")
        )
        if generated_local_compute:
            _bind_compute_method(module, spec, compute_fn)
            module.ep_enable = True
        elif compute_fn is not None:
            nested = _descendant_boundary_fqns(plan, module_fqn)
            _wrap_local_region_forward(
                module,
                boundary,
                spec,
                mesh,
                mesh_dim_names,
                validate_mode=False,
                compute_fn=compute_fn,
                exclude_subtrees=nested,
            )
        elif not _has_generated_boundary_forward(module):
            if _is_vocab_parallel_embed(module, spec, tp_mesh):
                _wrap_vocab_parallel_embedding(module, tp_mesh)
            _wrap_production_forward(
                module,
                boundary,
                spec,
                keep_output_dtensor=_keep_loss_parallel_output(plan, spec),
            )

        if hasattr(module, "tp_enable"):
            module.tp_enable = tp_mesh is not None


__all__ = ["apply_generated_phase_c"]
