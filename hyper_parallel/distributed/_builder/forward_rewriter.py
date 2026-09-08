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
"""forward_rewriter: forward-mutation machinery for the sharding applier.

All boundary forward generation, validation and installation lives here:
the injection signature validators (4b seed), the bias-suppression /
deferred-bias forwards, the production/validate boundary wrappers, the
MoE local-region wrapper, the CP inner-attention adapter, and the D-02
vocab-parallel embedding wrapper.

This module is the ONLY place in repo-owned code that assigns
``module.forward`` (AST gate, 05 §15.2.3): every other module returns a
replacement callable or a private :class:`_ForwardRewriteRequest`, and the
rewriter performs signature validation, installation and failure rollback.
The local-compute descendant adapters
(``install_local_compute_forward_adapters``) also live here;
``local_compute_context.py`` keeps only the region state. Qwen model-family
forward mutations remain in their transitional homes until changesets
M2/M3 (auto_models_adjust.md §5.3).
"""

import functools
import inspect
import logging
import types
import weakref
from typing import Any, Callable, List, Sequence

import torch
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.distributed.recipe_spec import (
    INNER_WRAPPER,
    PlacementMismatchError,
    require_injection_meta,
    resolve_placements,
)
from hyper_parallel.distributed._builder.local_compute_context import (
    _LOCAL_COMPUTE_ACTIVE,  # pylint: disable=protected-access
    local_compute_context,
)
from hyper_parallel.distributed._builder.parameter_sharding import (
    _temp_local_params,
)
from hyper_parallel.distributed._builder.rule_resolver import (
    _inner_target_name,
    _resolve_inner_output_placements,
    _resolve_inner_wrapper,
)
from hyper_parallel.distributed.tensor_parallel.head_count import (
    maybe_update_head_counts,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Runtime-layer signature validation (principle 1: the injected function's
# arguments must match the original function's)
# ────────────────────────────────────────────────────────────────────────────

def _is_subsequence(sub: List[str], full: List[str]) -> bool:
    """Return True if ``sub`` is an in-order subsequence of ``full``."""
    it = iter(full)
    return all(name in it for name in sub)


def _compute_parameters(compute_fn: Callable[..., Any], owner: str) -> List[inspect.Parameter]:
    """Return the declared compute parameters after its module argument."""
    try:
        params = list(inspect.signature(compute_fn).parameters.values())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{owner}: cannot introspect the signature of the injected compute fn") from exc
    if not params:
        raise TypeError(
            f"{owner}: a compute fn needs at least the module first "
            "parameter -- the contract is fn(module, *forward_args)"
        )
    return params[1:]


def _validate_explicit_compute_parameters(params: List[inspect.Parameter], owner: str) -> None:
    """Reject variadic compute parameters that hide signature mismatches."""
    for param in params:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"{owner}: a compute fn must not use *args/**kwargs "
                f"(parameter {param.name!r}) -- its arguments must explicitly "
                "match the original forward"
            )


def _validate_compute_parameter_names(
    fn_params: List[inspect.Parameter],
    fwd_params: List[inspect.Parameter],
    owner: str,
) -> None:
    """Validate compute parameter names and positional ordering."""
    fwd_names = {
        param.name
        for param in fwd_params
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    for param in fn_params:
        if param.name not in fwd_names:
            raise TypeError(
                f"{owner}: compute-fn parameter {param.name!r} has no "
                f"same-named entry among the original forward parameters {sorted(fwd_names)} -- "
                "the injected function's arguments must match the original function's "
                "(the skeleton forwards the actual forward arguments, so a name mismatch "
                "becomes a runtime TypeError)"
            )
    positional_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    fwd_pos = [param.name for param in fwd_params if param.kind in positional_kinds]
    fn_pos = [param.name for param in fn_params if param.kind in positional_kinds]
    if not _is_subsequence(fn_pos, fwd_pos):
        raise TypeError(
            f"{owner}: the compute fn's positional parameters {fn_pos} are not an in-order subsequence "
            f"of the original forward's positional parameters {fwd_pos} -- the skeleton forwards "
            "arguments positionally, so a reordering would bind them wrong"
        )


def _validate_required_compute_parameters(
    fn_params: List[inspect.Parameter],
    fwd_params: List[inspect.Parameter],
    owner: str,
) -> None:
    """Require the compute function to accept every required forward parameter."""
    fn_names = {param.name for param in fn_params}
    required = [
        param.name
        for param in fwd_params
        if param.default is inspect.Parameter.empty
        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    missing = [name for name in required if name not in fn_names]
    if missing:
        raise TypeError(
            f"{owner}: the original forward's required parameters {missing} are not accepted by the "
            "compute fn (the compute fn's declared arguments must cover the original function's required ones)"
        )


def validate_local_compute_signature(compute_fn: Callable[..., Any],
                                     forward: Callable[..., Any],
                                     *,
                                     owner: str) -> None:
    """Validate that a regional compute fn's arguments match the original
    forward (fail-fast at apply time).

    Rules (compute_fn's first parameter is module and forward's first
    parameter is self; both are skipped):
    1. The compute fn must not have *args/**kwargs (the skeleton forwards
       the actual forward arguments, and swallowing them would hide
       mismatches);
    2. Every compute-fn parameter must have a **same-named** parameter in
       forward (the skeleton forwards by kwarg, so a name mismatch is a
       runtime TypeError);
    3. The compute fn's positional-parameter sequence must be an in-order
       **subsequence** of forward's positional-parameter sequence
       (positional forwarding must not reorder);
    4. All required (defaultless) parameters of forward must be accepted by
       the compute fn.

    Args:
        compute_fn: The compute fn returned by the @local_compute factory.
        forward: The original module forward being replaced.
        owner: Call-site label prepended to error messages.

    Raises:
        TypeError: If any of the rules above is violated.
    """
    fn_params = _compute_parameters(compute_fn, owner)
    _validate_explicit_compute_parameters(fn_params, owner)
    fwd_params = list(inspect.signature(forward).parameters.values())
    _validate_compute_parameter_names(fn_params, fwd_params, owner)
    _validate_required_compute_parameters(fn_params, fwd_params, owner)


def validate_wrapped_forward(orig_forward: Callable[..., Any],
                             new_forward: Callable[..., Any],
                             *,
                             owner: str) -> None:
    """Validate that the forward replaced by inner_wrapper can receive all
    arguments of the original forward.

    A dummy-bind probe is constructed from the original signature
    (*args/**kwargs parameters cannot be forged and are skipped); the
    replacement side may pass through tolerantly with *args/**kwargs (it
    must forward arguments it does not care about to the original
    implementation), but every named parameter of the original forward must
    remain bindable by name.

    Args:
        orig_forward: The original module forward.
        new_forward: The replacement forward installed by the wrapper.
        owner: Call-site label prepended to error messages.

    Raises:
        TypeError: If the replacement signature cannot be introspected or
            cannot bind the original arguments.
    """
    orig_sig = inspect.signature(orig_forward)
    try:
        new_sig = inspect.signature(new_forward, follow_wrapped=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{owner}: cannot introspect the signature of the forward "
            "replaced by the injected wrapper") from exc
    args, kwargs = [], {}
    for p in orig_sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(None)
        else:
            # Probe POSITIONAL_OR_KEYWORD by name (kwarg) -- the
            # name-dimension contract is what the injection discipline
            # cares about; purely positional passing of optional positional
            # arguments is not required
            kwargs[p.name] = None
    try:
        new_sig.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{owner}: the forward replaced by the injected wrapper is "
            f"incompatible with the original forward's arguments ({exc}) -- "
            f"original signature {orig_sig}, replacement signature {new_sig}; "
            "the replacement forward must be able to receive all arguments "
            "of the original forward (it may pass through tolerantly with "
            "*args/**kwargs)") from exc


# ────────────────────────────────────────────────────────────────────────────
# Boundary forward wrapping: bias suppression / deferred bias (05 §4.4)
# ────────────────────────────────────────────────────────────────────────────

def _install_bias_suppression(module, spec):
    """D-22: make each defer-listed Linear run bias-free inside the region.

    The bias Parameter itself is never touched (same FQN / same object /
    state_dict / optimizer unchanged); the wrapper only hides it from
    ``F.linear`` for the duration of the region forward, so the boundary
    exit's Partial reduction sees a pure matmul contribution and the bias is
    added exactly once afterwards by :func:`_maybe_add_deferred_biases`.
    Both modes install the same wrapper (instruction-for-instruction
    identity, D-01''); nesting the suppression is idempotent.
    """
    for param_path in spec._deferred_bias_params:  # pylint: disable=protected-access
        owner_path = param_path.rpartition(".")[0]
        owner = module.get_submodule(owner_path) if owner_path else module
        original = owner.forward

        @functools.wraps(original)
        def bias_free_forward(
            *args: Any,
            __original: Callable[..., Any] = original,
            __owner: nn.Module = owner,
            **kwargs: Any,
        ) -> Any:
            """Run the owner's forward with its bias temporarily hidden.

            The bias Parameter object is restored on exit (even on error), so
            state_dict/optimizer visibility is unchanged; only ``F.linear``
            inside the region sees a bias-free Linear.
            """
            bias = __owner.bias
            try:
                __owner._parameters["bias"] = None  # pylint: disable=protected-access
                return __original(*args, **kwargs)
            finally:
                __owner._parameters["bias"] = bias  # pylint: disable=protected-access

        owner.forward = bias_free_forward


def _add_bias_to_primary_output(output, bias, module_name):
    """Add a deferred bias to the primary Tensor while preserving output structure."""
    if isinstance(output, torch.Tensor):
        primary_output = output
        rebuild = None
    elif isinstance(output, (tuple, list)):
        if not output or not isinstance(output[0], torch.Tensor):
            primary_type = type(output[0]).__name__ if output else "missing"
            raise TypeError(
                f"{module_name}: deferred bias requires output index 0 to be a Tensor, "
                f"got {primary_type}"
            )
        primary_output = output[0]
        rebuild = list(output)
    else:
        raise TypeError(
            f"{module_name}: deferred bias requires a Tensor, tuple, or list output, "
            f"got {type(output).__name__}"
        )

    if not isinstance(primary_output, DTensor) and isinstance(bias, DTensor):
        bias = bias.to_local()
    biased_output = primary_output + bias
    if rebuild is None:
        return biased_output
    rebuild[0] = biased_output
    return tuple(rebuild) if isinstance(output, tuple) else rebuild


def _maybe_add_deferred_biases(module, spec, output):
    """D-22: add each deferred bias exactly once AFTER the boundary exit.

    The bias is read at forward time (never captured in a closure), so
    production sees the unwrapped local tensor and validate sees the DTensor
    — whichever form the boundary output currently has: nested validate keeps
    the DTensor (dispatch add, Shard(1)+Replicate→Shard(1)); the outermost /
    production / local-region exits are local, and a DTensor bias is
    unwrapped to match. For structured attention outputs, index 0 is the
    primary hidden-state Tensor; metadata/cache entries are preserved.
    """
    if not spec._deferred_bias_params:  # pylint: disable=protected-access
        return output
    for param_path in spec._deferred_bias_params:  # pylint: disable=protected-access
        owner_path = param_path.rpartition(".")[0]
        owner = module.get_submodule(owner_path) if owner_path else module
        bias = owner.bias
        if bias is None:
            continue
        output = _add_bias_to_primary_output(output, bias, type(module).__name__)
    return output


# ────────────────────────────────────────────────────────────────────────────
# Production/validate boundary wrappers and output-contract validation
# ────────────────────────────────────────────────────────────────────────────

def _keep_loss_parallel_output(plan, spec):
    """Return whether a terminal vocab-sharded output must remain a DTensor."""
    if not plan.loss_parallel or not spec._is_terminal:  # pylint: disable=protected-access
        return False
    output_placements = (spec.out_dst or {}).get("output", {})
    return any(
        isinstance(placement, Shard) and placement.dim == -1
        for placement in output_placements.values()
    )


def _descendant_boundary_fqns(plan, module_fqn):
    """Relative FQNs of boundaries nested inside *module_fqn* (D-14).

    Returned relative to module_fqn (matching the name space of
    module.named_parameters(recurse=True)); the root spec (fqn "") treats
    every other boundary as a descendant.
    """
    if module_fqn == "":
        return [f for f in plan.modules if f]
    prefix = module_fqn + "."
    return [f[len(prefix):] for f in plan.modules if f.startswith(prefix)]


def _bind_input_indices(boundary, module):
    """Bind the arg_name of in_plan to the positional index of the forward signature.

    Inter-module calls are mostly positional (self.mlp(x) inside a layer), so
    RedistOp's kwargs lookup by name would miss -- the signature index is bound
    at compile time, and at runtime _get_arg checks kwargs first, then args.
    """
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        positional = [
            name for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        name_to_idx = {name: i for i, name in enumerate(positional)}
        for op in boundary.in_plan:
            if op.arg_index is None and op.arg_name in name_to_idx:
                op.arg_index = name_to_idx[op.arg_name]
    # Positional fallback: a single-input contract (in_plan has only 1 op) is
    # bound to the first positional parameter -- covers cases where the
    # template key (e.g. "hidden_states") differs from the leaf module's
    # signature (nn.Linear.forward(input)).
    if len(boundary.in_plan) == 1 and boundary.in_plan[0].arg_index is None:
        boundary.in_plan[0].arg_index = 0


def _wrap_production_forward(
        module, boundary, spec=None, keep_output_dtensor=False):
    """Production mode: pure local tensor computation + precompiled boundary communication (05 §4.4.1).

    _local_params_context was already invoked at the Phase C entry (parameters
    permanently unpacked).
    """
    original_forward = module.forward

    @functools.wraps(original_forward)
    def production_forward(*args: Any, **kwargs: Any) -> Any:
        """Production boundary forward: precompiled entry communication,
        local computation, exit redistribution, then D-22 deferred biases."""
        args, kwargs = boundary.redistribute_inputs(args, kwargs)
        outputs = original_forward(*args, **kwargs)
        outputs = boundary.redistribute_outputs(
            outputs,
            as_dtensor_input=keep_output_dtensor,
        )
        # D-22: deferred rowwise biases — added once after the exit reduction
        return _maybe_add_deferred_biases(module, spec, outputs) \
            if spec is not None else outputs

    module.forward = production_forward


def _wrap_validate_forward(
        module, boundary, spec, mesh_dim_names, keep_output_dtensor=False):
    """Validate mode: DTensor propagation end to end -> validate out_src (core) + out_dst (terminal modules only).

    The in-house DTensor is forward-only: validation covers only forward
    placement propagation; backward is local autograd in both modes (05 §1.0),
    and gradient equivalence is guaranteed by tests/ut/auto_models/distributed/grad_equiv.py.
    """
    original_forward = module.forward
    module_name = type(module).__name__

    @functools.wraps(original_forward)
    def validate_forward(*args: Any, **kwargs: Any) -> Any:
        """Validate boundary forward: wrap inputs as DTensors, propagate
        placements through the original forward, validate out_src, then
        redistribute to out_dst."""
        # D-14 nesting (05 §13.4): detect whether the call arrives from an
        # outer DTensor-propagating boundary BEFORE Step 1 wraps everything
        # into DTensors (Step 1 would make the check useless).
        nested = any(isinstance(a, DTensor) for a in args) or any(
            isinstance(v, DTensor) for v in kwargs.values())
        # Step 1: inputs -> DTensor
        args, kwargs = boundary.redistribute_inputs(args, kwargs, as_dtensor=True)

        # Step 2: parameters stay DTensors; placement propagates via __torch_function__ dispatch
        outputs = original_forward(*args, **kwargs)

        # Step 3: [core validation] out_src -- native DTensor-propagated output vs declaration
        if spec.out_src is not None:
            _validate_out_src(outputs, spec, mesh_dim_names, module_name)

        # Step 4: redistribute to out_dst
        outputs = boundary.redistribute_outputs(outputs, as_dtensor_input=True)

        # Step 5: [defensive validation] out_dst -- terminal modules only
        if spec._is_terminal and spec.out_dst is not None:  # pylint: disable=protected-access
            _validate_out_dst(outputs, spec, mesh_dim_names, module_name)

        # D-22: deferred biases are added after the exit reduction while the
        # output is still a DTensor (Shard/Replicate + Replicate bias dispatch
        # add — the Partial reduction is already done, so every rank adds the
        # bias exactly once); Step 6 then unwraps as usual.
        outputs = _maybe_add_deferred_biases(module, spec, outputs)

        # Step 6: return local (isomorphic to production boundary outputs) --
        # but under an outer DTensor-propagating boundary (D-14 nesting, 05
        # §13.4) keep the DTensor so the outer forward's dispatch chain is
        # unbroken; the outermost boundary exit unwraps.
        if nested or keep_output_dtensor:
            return outputs
        if isinstance(outputs, DTensor):
            outputs = outputs.to_local()
        elif isinstance(outputs, (tuple, list)):
            outputs = tuple(
                t.to_local() if isinstance(t, DTensor) else t for t in outputs
            )
        return outputs

    module.forward = validate_forward


def _validate_out_src(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_src")


def _validate_out_dst(outputs, spec, mesh_dim_names, module_name):
    _validate_outputs(outputs, spec, mesh_dim_names, module_name, "out_dst")


def _normalize_placements_ndim(placements, ndim):
    """Normalize negative dims like Shard(-1) against the tensor ndim (Shard(-1) == Shard(ndim-1))."""
    out = []
    for p in placements:
        if isinstance(p, Shard) and p.dim < 0:
            out.append(Shard(p.dim + ndim))
        else:
            out.append(p)
    return tuple(out)


def _validate_outputs(outputs, spec, mesh_dim_names, module_name, stage):
    """Placement validation for single/multi outputs (shared by out_src / out_dst).

    Multi outputs are mapped to tuple positions via spec.out_names (falling
    back to declaration key order); outputs that are not returned or are not
    DTensors are skipped. Declared and actual placements are
    negative-dim-normalized before comparison.
    """
    declared = getattr(spec, stage)
    if isinstance(outputs, (tuple, list)):
        out_names = getattr(spec, "out_names", None) or list(declared.keys())
        name_to_idx = {name: i for i, name in enumerate(out_names)}
        items = list(outputs)
    else:
        name_to_idx = {name: 0 for name in declared}
        items = [outputs]
    for out_name, expected_named in declared.items():
        idx = name_to_idx.get(out_name)
        if idx is None or idx >= len(items):
            continue
        tensor = items[idx]
        if not isinstance(tensor, DTensor):
            continue
        ndim = len(tensor.shape)
        expected = _normalize_placements_ndim(
            tuple(resolve_placements(expected_named, mesh_dim_names)), ndim)
        actual = _normalize_placements_ndim(tuple(tensor.placements), ndim)
        if expected != actual:
            suffix = f"[{out_name}]" if len(declared) > 1 else ""
            raise PlacementMismatchError(
                module_name, expected, actual, f"{stage}{suffix}"
            )


# ────────────────────────────────────────────────────────────────────────────
# MoE EP local region (05 §4.4.3 + D-03')
# ────────────────────────────────────────────────────────────────────────────

def _rewrap_local_outputs(output, spec, mesh, mesh_dim_names, module_name):
    """Wrap every declared local Tensor output with its ``out_src`` layout."""
    declared = spec.out_src or {}
    if not declared:
        return output

    is_sequence = isinstance(output, (tuple, list))
    items = list(output) if is_sequence else [output]
    out_names = list(getattr(spec, "out_names", None) or declared.keys())
    name_to_idx = {name: index for index, name in enumerate(out_names)}

    for out_name, named_placement in declared.items():
        index = name_to_idx.get(out_name)
        if index is None:
            raise ValueError(
                f"{module_name}: out_src declares output {out_name!r}, but "
                f"out_names={out_names!r} does not contain it"
            )
        if index >= len(items):
            raise ValueError(
                f"{module_name}: out_src maps output {out_name!r} to index "
                f"{index}, but forward returned only {len(items)} output(s)"
            )
        item = items[index]
        if item is None:
            continue
        if isinstance(item, DTensor):
            continue
        if not isinstance(item, torch.Tensor):
            raise TypeError(
                f"{module_name}: declared output {out_name!r} at index "
                f"{index} must be a Tensor or None, got {type(item).__name__}"
            )
        placements = tuple(resolve_placements(named_placement, mesh_dim_names))
        items[index] = DTensor.from_local(item, mesh, placements)

    if isinstance(output, tuple):
        return tuple(items)
    if isinstance(output, list):
        return items
    if len(items) != 1:
        raise ValueError(
            f"{module_name}: scalar forward output cannot satisfy "
            f"{len(declared)} declared out_src entries"
        )
    return items[0]


def _wrap_local_region_forward(module, boundary, spec, mesh, mesh_dim_names,
                               *, validate_mode=False, compute_fn=None,
                               exclude_subtrees=()):
    """Generic local-region forward wrapper (D-03', formerly the _wrap_moe_forward skeleton).

    Structure: boundary entry -> local region -> re-wrap per the declared
    out_src -> boundary exit. Applies to any module containing data-dependent
    logic that DTensor dispatch cannot express (e.g. MoE all-to-all) --
    injected by _apply_phase_c when _resolve_local_compute_fn resolves to
    non-None (derived gate, 05 §4.4.3).

    production: parameters were permanently unpacked at build time and inputs
    are local (boundary passthrough); validate: inputs are DTensors ->
    to_local -> temporarily unwrap parameters -> local computation -> re-wrap
    the output per the declared out_src via from_local (for data-dependent
    modules out_src is declarative validation -- the data dependence of
    all-to-all makes the placement underivable; this is an inherent
    limitation). Both modes share the same wrapper code (local_region
    tolerant passthrough semantics).

    compute_fn: the function actually executed inside the region; defaults to
    the module's own forward. Resolved uniformly by
    _resolve_local_compute_fn (user local_compute_fn / region_dispatch=False
    gate), independent of the original forward.

    spec.region_dispatch=True (dispatchable pure-ops injection): validate
    feeds the DTensors straight into compute_fn — strategy propagation runs
    THROUGH the injected fn and out_src is TRULY validated (propagated vs
    declared); production is unchanged (always local passthrough).
    """
    original_forward = module.forward
    if compute_fn is None:
        compute_fn = original_forward


    out_src_placements = None
    if spec.out_src:
        out_src_named = next(iter(spec.out_src.values()))
        out_src_placements = tuple(resolve_placements(out_src_named, mesh_dim_names))

    dispatch_through = bool(getattr(spec, "region_dispatch", None))
    if validate_mode and not dispatch_through:
        install_local_compute_forward_adapters(module, exclude=exclude_subtrees)

    @functools.wraps(original_forward)
    def local_region_forward(*args: Any, **kwargs: Any) -> Any:
        """Local-region forward: boundary entry → local region computation →
        re-wrap per the declared out_src → boundary exit (both modes share
        this wrapper)."""
        # Step 1: PrecompiledBoundary entry (e.g. TP all-gather; identity passthrough)
        args, kwargs = boundary.redistribute_inputs(
            args, kwargs, as_dtensor=validate_mode)

        # Step 2: local region -- the data-dependent computation (e.g. EP
        # dispatch/combine) executes on local tensors; region_dispatch=True
        # instead dispatches THROUGH the injected fn (pure standard ops) so
        # validate's strategy propagation covers it
        if validate_mode and dispatch_through:
            try:
                output = compute_fn(*args, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    f"{exc}\n[region_dispatch=True] dispatch failed while "
                    "validate was dispatching through the injected "
                    "function — does the injection contain "
                    "non-dispatchable communication primitives/custom "
                    "kernels? Declare region_dispatch=False instead "
                    "(skeleton black-box hosting)"
                ) from exc
            # True validation: propagated result vs the out_src declaration
            # (a wrong declaration fails fast)
            if spec.out_src is not None:
                _validate_out_src(output, spec, mesh_dim_names,
                                  type(module).__name__)
        elif validate_mode:
            local_args = tuple(
                a.to_local() if isinstance(a, DTensor) else a for a in args)
            local_kwargs = {
                k: (v.to_local() if isinstance(v, DTensor) else v)
                for k, v in kwargs.items()
            }
            with local_compute_context():
                with _temp_local_params(module, exclude=exclude_subtrees):
                    output = compute_fn(*local_args, **local_kwargs)
        else:
            output = compute_fn(*args, **kwargs)

        # Step 3: local -> DTensor (re-wrap per the declared out_src, restoring
        # the DTensor metadata broken by all-to-all; under production the
        # boundary exit needs the same contract)
        if not isinstance(output, DTensor):
            output = _rewrap_local_outputs(
                output, spec, mesh, mesh_dim_names, type(module).__name__)

        # Step 4: PrecompiledBoundary exit (e.g. TP reduce-scatter)
        output = boundary.redistribute_outputs(
            output, as_dtensor_input=validate_mode)
        # The final boundary exit is always local (when out_plan is empty, the
        # from_local wrap from Step 3 must also be unwrapped here)
        if isinstance(output, DTensor):
            output = output.to_local()
        # D-22: deferred rowwise biases — added once after the exit reduction
        return _maybe_add_deferred_biases(module, spec, output)

    module.forward = local_region_forward


# ────────────────────────────────────────────────────────────────────────────
# CP inner attention wrapper (05 §4.4.2 + D-01'' + D-04)
# ────────────────────────────────────────────────────────────────────────────

def _apply_custom_inner_wrapper(custom_fn, context):
    """Apply a user-defined inner_wrapper (@inner_wrapper callable).

    Contract: custom_fn must be decorated ``@inner_wrapper``; it is invoked
    with its declared context (anchor target_module + the mandatory mesh
    family, filled by name). It either replaces target.forward in place and
    returns None (the external contract) or RETURNS the replacement — a
    forward callable or _ForwardRewriteRequest(s) — for the rewriter to
    validate and install (the in-repo discipline, 05 §15.2.3). The
    replacement forward runs in a LOCAL-TENSOR world — the dual-mode
    adapter installed by _wrap_inner_attention owns all DTensor
    conversion (to_local / _temp_local_params / from_local rewrap per the
    declared placements), so custom wrappers never touch DTensor.
    """
    meta = require_injection_meta(
        custom_fn, INNER_WRAPPER, source="spec.inner_wrapper")
    return custom_fn(**{k: context[k] for k in sorted(meta.context)})


def _find_dtensor_argument(args, kwargs):
    """Return the first DTensor positional or keyword argument."""
    return next(
        (
            value
            for value in (*args, *kwargs.values())
            if isinstance(value, DTensor)
        ),
        None,
    )


def _validate_inner_dispatch_output(
    output,
    expected_placements,
    boundary_module,
    wrapper_name,
):
    """Validate DTensor outputs produced through a dispatchable inner wrapper."""
    if expected_placements is None:
        return
    outputs = list(output) if isinstance(output, (tuple, list)) else [output]
    tensor_outputs = [tensor for tensor in outputs if isinstance(tensor, torch.Tensor)]
    if len(tensor_outputs) != len(expected_placements):
        raise RuntimeError(
            f"inner_wrapper {wrapper_name!r}: tensor output count "
            f"{len(tensor_outputs)} does not match the "
            f"{len(expected_placements)} declared placements — a "
            "multi-output contract must be declared name by name with "
            "matching counts"
        )
    for tensor, placements in zip(tensor_outputs, expected_placements):
        if not isinstance(tensor, DTensor):
            raise RuntimeError(
                f"inner_wrapper {wrapper_name!r} [region_dispatch=True]: "
                f"the dispatch-propagated output is not a DTensor "
                f"({type(tensor).__name__}) — the injection appears to "
                "have left the dispatch chain, so true validation cannot "
                "complete"
            )
        if tuple(tensor.placements) != tuple(placements):
            raise PlacementMismatchError(
                f"{type(boundary_module).__name__} "
                f"(inner_wrapper {wrapper_name!r})",
                tuple(placements),
                tuple(tensor.placements),
                "inner_out_src",
            )


def _rewrap_inner_tensor(tensor, placements, mesh):
    """Wrap one local tensor with its declared layout and pass other values through."""
    if not isinstance(tensor, torch.Tensor) or isinstance(tensor, DTensor):
        return tensor
    return DTensor.from_local(tensor, mesh, placements)


def _rewrap_inner_outputs(output, out_placements, mesh, wrapper_name):
    """Rewrap tensor outputs while preserving auxiliary non-tensor outputs."""
    if not isinstance(output, (tuple, list)):
        return _rewrap_inner_tensor(output, out_placements[0], mesh)
    tensor_output_count = sum(isinstance(value, torch.Tensor) for value in output)
    if tensor_output_count != len(out_placements):
        raise RuntimeError(
            f"inner_wrapper {wrapper_name!r}: the replacement forward "
            f"returned {tensor_output_count} tensor outputs, which does "
            f"not match the {len(out_placements)} declared placements — "
            "a multi-output contract must be declared name by name with "
            "matching counts"
        )
    placement_iter = iter(out_placements)
    wrapped = [
        _rewrap_inner_tensor(value, next(placement_iter), mesh)
        if isinstance(value, torch.Tensor)
        else value
        for value in output
    ]
    return tuple(wrapped) if isinstance(output, tuple) else wrapped


def _install_inner_adapter(target, user_fwd, boundary_module, spec, mesh,
                           mesh_dim_names, wrapper_name, validate_mode=False):
    """Unified dual-mode adapter: rewrap rules are resolved at install time,
    with zero runtime decisions (05 §4.4.2 + D-01'').

    The replacement forward of a user wrapper only faces local tensors. The
    adapter handles:
    - validate (any input is a DTensor): every DTensor input is to_local'd
      (non-tensors pass through) + ``_temp_local_params(target)`` temporarily
      unwraps the parameters → call the user forward → rewrap the outputs
      back into DTensors per the declaration (the propagation chain is
      re-linked and boundary validation continues);
    - production (no DTensor inputs): straight passthrough, zero conversion
      overhead.

    Source of the rewrap placements (the framework infers and guesses
    nothing — everything is explicitly declared):
    - case A (target is the boundary module itself): the boundary
      ``spec.out_src`` declaration (multiple outputs positionally per
      ``out_names``/declaration key order);
    - case B (an inner submodule): the explicit ``spec.inner_out_src``
      declaration — the sentinel ``"first_input"`` (output layout == the
      runtime layout of the first DTensor input; for layout-preserving
      wrappers, single output only) or NamedPlacement /
      {name: NamedPlacement} (multiple outputs mapped to tuple positions per
      declaration key order); undeclared → fail-fast at install time.
    """
    first_input_rule, out_placements = _resolve_inner_output_placements(
        spec,
        boundary_module,
        target,
        mesh_dim_names,
        wrapper_name,
    )
    if validate_mode and not getattr(spec, "region_dispatch", None):
        install_local_compute_forward_adapters(target)

    @functools.wraps(user_fwd)
    def adapted(*args: Any, **kwargs: Any) -> Any:
        """Dual-mode adapter around the user's replacement forward.

        production (no DTensor input): straight passthrough; validate:
        unwrap DTensor inputs/parameters to local, run the user forward, and
        rewrap outputs per the declaration resolved at install time (or, with
        region_dispatch=True, dispatch through and truly validate instead).
        """
        source_dtensor = _find_dtensor_argument(args, kwargs)
        if source_dtensor is None:
            return user_fwd(*args, **kwargs)          # production: straight passthrough
        if getattr(spec, "region_dispatch", None):
            # Validate dispatch-through (region_dispatch=True): the injection
            # is pure standard ops — DTensors go straight in and dispatch
            # propagation runs through the inner region; the declared rewrap
            # rules are promoted to the true-validation baseline (propagated
            # result vs declaration, a mismatch fails fast).
            try:
                out = user_fwd(*args, **kwargs)
            except Exception as exc:
                raise type(exc)(
                    f"{exc}\n[region_dispatch=True] dispatch failed "
                    f"while validate was dispatching through "
                    f"inner_wrapper {wrapper_name!r} — does the injection "
                    "contain non-dispatchable communication "
                    "primitives/custom kernels? Declare "
                    "region_dispatch=False instead (adapter black-box "
                    "hosting)") from exc
            expected = (
                [tuple(source_dtensor.placements)]
                if first_input_rule
                else out_placements
            )
            _validate_inner_dispatch_output(
                out,
                expected,
                boundary_module,
                wrapper_name,
            )
            return out
        # validate: uniformly convert to local (parameters temporarily
        # unwrapped, restored on exit)
        local_args = tuple(
            a.to_local() if isinstance(a, DTensor) else a for a in args)
        local_kwargs = {k: (v.to_local() if isinstance(v, DTensor) else v)
                        for k, v in kwargs.items()}
        with local_compute_context():
            with _temp_local_params(target):
                out = user_fwd(*local_args, **local_kwargs)

        if first_input_rule:
            if isinstance(out, (tuple, list)):
                raise RuntimeError(
                    f"inner_wrapper {wrapper_name!r}: inner_out_src="
                    "'first_input' only supports a single output — for "
                    "multiple outputs declare inner_out_src explicitly in "
                    "the {name: {axis: placement}} form")
            return _rewrap_inner_tensor(
                out,
                tuple(source_dtensor.placements),
                source_dtensor.device_mesh,
            )
        if out_placements is None:
            return out                   # case A with no out_src declaration: no rewrap
        return _rewrap_inner_outputs(out, out_placements, mesh, wrapper_name)

    target.forward = adapted


def _wrap_inner_attention(module, cp_mesh, *, spec=None, mesh=None,
                          mesh_dim_names=(), tp_mesh=None, ep_mesh=None,
                          validate_mode=False, module_fqn=""):
    """Inject an inner forward wrapper (one-shot replacement at apply time, 05 §4.4.2).

    General "weave into / replace the inner forward" mechanism — **not gated
    on CP**: whenever ``spec.inner_wrapper`` is declared the wrapper is
    applied (declaration == application; the resolution chain is the derived
    gate). The shipped CP wrappers (INNER_WRAPPER_REGISTRY) are its first-class
    built-in use case and still require an active cp axis (fail-fast in the
    resolution chain otherwise); custom callables/Targets receive
    ``cp_mesh=None`` when no cp axis exists and own their semantics.

    Resolution (_resolve_inner_wrapper, a pure function of the chain) is
    separated from application; when resolution returns None (no explicit
    declaration), this returns None and injects nothing. Returns the resolved
    wrapper name (or None).

    D-01'': production and validate inject **the same** wrapper, so the
    in-region computation is instruction-for-instruction identical
    (kernel-level equivalence).
    D-04: when is_causal and CP is active, replace it with an offset-aware
    explicit mask.
    Nothing is located silently: after injection an INFO log records the
    target/wrapper/source, and spec._resolved_inner_wrapper +
    spec._resolved_inner_target are written back for plan introspection.
    """
    resolved = _resolve_inner_wrapper(
        module, spec, cp_mesh, mesh, tp_mesh=tp_mesh, ep_mesh=ep_mesh)
    if resolved is None:
        return None
    name, target, apply_fn = resolved
    if validate_mode and getattr(spec, "region_dispatch", None) is False:
        # A black-box inner wrapper receives local tensors and local parameter
        # shards in validate mode. Cached head counts must therefore match the
        # TP-local projection widths, just as they already do in production.
        maybe_update_head_counts(
            target,
            spec,
            module_fqn or type(module).__name__,
            mesh,
            mesh_dim_names,
        )
    orig_forward = target.forward
    # Record the pre-rewrite state for failure rollback: an in-place wrapper
    # writes an instance attribute, so __dict__ holds the full mutation.
    saved_state = {"forward": target.__dict__.get("forward", _MISSING)}
    try:
        primary, secondaries = _classify_rewrite_result(
            apply_fn(), target, name)
        for request in secondaries:
            _commit_forward_rewrite(request)
        if primary is None:
            # In-place contract (external @inner_wrapper wrappers). Detect
            # "a replacement really happened": attribute access on a bound
            # method creates a new object each time, so an `is` comparison is
            # always true — the underlying function objects (__func__) must
            # be compared, otherwise a pure-probe wrapper (which does not
            # replace forward) would also get an adapter installed by mistake
            # and be forced to declare inner_out_src
            new_forward = target.forward
            replaced = (getattr(new_forward, "__func__", new_forward)
                        is not getattr(orig_forward, "__func__", orig_forward))
        else:
            # In-repo discipline: the wrapper returned its replacement, the
            # rewriter commits the companion attributes and installs.
            saved_state.update({
                attr: getattr(target, attr, _MISSING)
                for attr in primary.companion_attrs
            })
            for attr, value in primary.companion_attrs.items():
                setattr(target, attr, value)
            new_forward = primary.forward
            replaced = True
        if replaced:
            # Principle 1: the replaced forward must accept all inputs of the
            # original forward
            validate_wrapped_forward(
                orig_forward, new_forward,
                owner=f"inner_wrapper {name!r} on {type(module).__name__}")
            # Uniformly install the dual-mode adapter: the user wrapper only
            # faces local tensors; DTensor conversion and declarative rewrap
            # are managed by the adapter (local_map semantics; validate skips
            # propagation checks for the inner region — the safety net is at
            # the boundary layer)
            _install_inner_adapter(
                target, new_forward, module, spec, mesh, mesh_dim_names, name,
                validate_mode=validate_mode)
    except Exception:
        # Failure rollback: restore every attribute written during the
        # rewrite so a half-installed wrapper never survives.
        for attr, saved in saved_state.items():
            _restore_attr(target, attr, saved)
        raise
    target_name = _inner_target_name(module, target)
    if spec is not None:
        spec._resolved_inner_wrapper = name  # pylint: disable=protected-access
        spec._resolved_inner_target = target_name  # pylint: disable=protected-access
    if name == "custom":
        source = "custom callable"
    elif spec is not None and isinstance(
            getattr(spec, "inner_wrapper", None), str):
        source = "explicitly specified (registry)"
    else:
        source = "explicitly specified (Target)"
    logger.info("inner-wrap: %s target=%s <- wrapper %r (%s)",
                type(module).__name__, target_name, name, source)
    return name


# ────────────────────────────────────────────────────────────────────────────
# D-02 vocab-parallel embedding wrapper
# ────────────────────────────────────────────────────────────────────────────

def _is_vocab_parallel_embed(module, spec, tp_mesh) -> bool:
    """production embed boundary check: nn.Embedding + weight Shard(0) on TP + TP>1."""
    if tp_mesh is None or tp_mesh.size() <= 1:
        return False
    if not isinstance(module, nn.Embedding):
        return False
    weight_named = spec.params.get("weight", {})
    return weight_named.get("tp") == Shard(0)


def _wrap_vocab_parallel_embedding(module, tp_mesh):
    """D-02: Megatron-style masked embedding (injected at the production embed boundary).

    The vocab-range mask logic of DTensor dispatch is lost after the
    parameter unwrap -- HF native F.embedding would index out of range when
    given global token ids. The wrapper: tokens outside the local vocab
    interval [lo, hi) are zeroed and indices are shifted by the offset, so the
    output is naturally a Partial contribution and the boundary exit's
    Partial->Shard(1) reduction is unchanged.
    """
    original_forward = module.forward
    v_local = module.weight.shape[0]
    lo = tp_mesh.get_local_rank() * v_local
    hi = lo + v_local

    @functools.wraps(original_forward)
    def masked_embedding_forward(
        input_ids: torch.Tensor, *args: Any, **kwargs: Any,
    ) -> torch.Tensor:
        """Masked vocab-parallel embedding forward: zero out-of-range token
        ids, shift indices by the local vocab offset, and mask the output so
        it is a pure Partial contribution."""
        mask = (input_ids >= lo) & (input_ids < hi)
        local_ids = torch.where(mask, input_ids - lo, torch.zeros_like(input_ids))
        out = original_forward(local_ids, *args, **kwargs)
        return out * mask.unsqueeze(-1).to(out.dtype)

    module.forward = masked_embedding_forward

# ────────────────────────────────────────────────────────────────────────────
# Install primitives — the ONLY forward-assignment sites in repo-owned code
# (AST gate, 05 §15.2.3). Everywhere else, a forward mutation is expressed as
# a returned callable or _ForwardRewriteRequest and committed here.
# ────────────────────────────────────────────────────────────────────────────

_MISSING = object()


class _ForwardRewriteRequest:
    """An atomic forward-rewrite request returned by wrapper factories.

    Carries the target module, the replacement forward and the companion
    attribute updates (e.g. ``attention_interface``/context flags) that must
    commit together with the forward swap; the rewriter validates, commits
    and rolls back as one unit (05 §15.2.3, P0 atomicity requirement).
    """

    __slots__ = ("target", "forward", "companion_attrs")

    def __init__(self, target, forward, companion_attrs=()):
        self.target = target
        self.forward = forward
        self.companion_attrs = dict(companion_attrs)


def _restore_attr(module, name, saved):
    """Restore one attribute to its pre-rewrite state (_MISSING deletes it)."""
    if saved is _MISSING:
        vars(module).pop(name, None)
    else:
        setattr(module, name, saved)


def _commit_forward_rewrite(request):
    """Commit one rewrite request atomically, verbatim (no dual-mode adapter).

    Used for secondary targets of a wrapper (e.g. the text-model input
    sharding of the MLA/DSA Ulysses scheme); the primary target of a plan
    injection goes through ``_wrap_inner_attention`` instead, which validates
    the signature and installs the dual-mode adapter.
    """
    target = request.target
    saved = {
        "forward": target.__dict__.get("forward", _MISSING),
        **{name: getattr(target, name, _MISSING)
           for name in request.companion_attrs},
    }
    try:
        for attr_name, value in request.companion_attrs.items():
            setattr(target, attr_name, value)
        target.forward = request.forward
    except Exception:
        for attr_name, value in saved.items():
            _restore_attr(target, attr_name, value)
        raise


def _classify_rewrite_result(returned, target, wrapper_name):
    """Normalize a wrapper's return into (primary request, secondary requests).

    Return contract for @inner_wrapper callables: None for in-place
    replacement (the external contract — target.forward was assigned by the
    wrapper itself); a replacement forward callable for the resolved target;
    or _ForwardRewriteRequest(s) — the entry targeting the resolved module is
    the primary (validated + dual-mode adapter installed), the rest are
    committed verbatim.
    """
    if returned is None:
        return None, []
    entries = (list(returned)
               if isinstance(returned, (list, tuple)) else [returned])
    requests = []
    for entry in entries:
        if isinstance(entry, _ForwardRewriteRequest):
            requests.append(entry)
        elif callable(entry):
            requests.append(_ForwardRewriteRequest(target, entry))
        else:
            raise TypeError(
                f"inner_wrapper {wrapper_name!r} returned "
                f"{type(entry).__name__} — expected None (in-place forward "
                "replacement), a replacement forward callable, or a "
                "_ForwardRewriteRequest")
    primaries = [request for request in requests if request.target is target]
    if len(primaries) > 1:
        raise TypeError(
            f"inner_wrapper {wrapper_name!r} returned multiple rewrite "
            "requests for the resolved target — exactly one primary "
            "replacement is allowed")
    secondaries = [request for request in requests
                   if request.target is not target]
    return (primaries[0] if primaries else None), secondaries


def _install_bound_forward(module, unbound_forward):
    """Install ``unbound_forward(module, *args, **kwargs)`` as a bound forward.

    The repo's only ``types.MethodType`` forward installation: for entry-point
    binders whose replacement is written as a plain function taking the module
    explicitly (e.g. the EP local-expert entry point); companion attributes
    (``local_expert_count`` and friends) are set by the binder beforehand.
    """
    module.forward = types.MethodType(unbound_forward, module)


# ────────────────────────────────────────────────────────────────────────────
# Local-compute descendant adapters (moved from local_compute_context.py,
# which keeps only the region state)
# ────────────────────────────────────────────────────────────────────────────

_ADAPTED_MODULES = weakref.WeakSet()


def install_local_compute_forward_adapters(
    module: Any,
    exclude: Sequence[str] = (),
) -> None:
    """Make descendant forwards expose their directly owned local parameters.

    A descendant FSDP pre-hook executes before its wrapped ``forward``, so the
    wrapper observes and temporarily unwraps the newly installed unsharded
    DTensor. Its restoration finishes before the FSDP post-hook. The same
    wrapper is a no-op when FSDP is disabled or execution is outside a local
    compute region.

    The root is intentionally omitted: the local-region skeleton already
    handles all parameters visible when its forward starts. Descendant
    installation is idempotent across overlapping local regions.

    Args:
        module: Root module of a trainer-managed local compute region.
        exclude: Relative descendant FQNs that remain DTensor dispatch islands.
    """
    excluded = tuple(name.rstrip(".") for name in exclude)
    descendants = tuple(module.named_modules())[1:]
    for relative_fqn, target in descendants:
        if any(
            relative_fqn == name or relative_fqn.startswith(name + ".")
            for name in excluded
        ):
            continue
        if target in _ADAPTED_MODULES:
            continue
        _wrap_module_forward(target)
        _ADAPTED_MODULES.add(target)


def _wrap_module_forward(module: Any) -> None:
    """Wrap one descendant module without changing its public forward contract."""
    original_forward = module.forward

    @functools.wraps(original_forward)
    def local_param_forward(*args: Any, **kwargs: Any) -> Any:
        """Run forward with directly owned DTensor parameters unwrapped locally."""
        if not _LOCAL_COMPUTE_ACTIVE.get():
            return original_forward(*args, **kwargs)
        saved = []
        for name, parameter in module.named_parameters(recurse=False):
            if isinstance(parameter, DTensor):
                saved.append((name, parameter))
                module._parameters[name] = parameter.to_local()  # pylint: disable=protected-access
        try:
            return original_forward(*args, **kwargs)
        finally:
            for name, parameter in saved:
                module._parameters[name] = parameter  # pylint: disable=protected-access

    module.forward = local_param_forward
