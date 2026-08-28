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
"""injection: template decorators and signature validation for injected
functions (the entry-point discipline of the explicit injection mechanism).

Design principle: **an injected function is called exactly once at apply
time, must declare the mesh family (filled by the framework by name; axes
that are not active are filled with None), and may use them or not**. Two
decorators cover all injection channels, each with a single canonical form
(since 2026-08-10 the direct compute-fn form of local_compute_fn is
retired; everything is unified into the factory form -- the same
specification as inner_wrapper: the mesh family is always passed in, and
whether to use it is the user's own choice):

- ``@local_compute``: a regional-compute **factory**,
  ``fn(mesh, tp_mesh, cp_mesh, ep_mesh, [module], <config keys...>)
  -> compute_fn`` -- built once at apply time (typically building
  communication-group closures from ep_mesh so the runtime incurs zero
  mesh overhead). The returned compute fn ``fn(module,
  *original_forward_args)`` needs no further decoration (see signature
  validation below);
- ``@inner_wrapper``: an inner-forward wrapper, ``fn(target_module, mesh,
  tp_mesh, cp_mesh, ep_mesh) -> None`` -- replaces ``target.forward`` in
  place.

The full set of framework context (reserved names whose semantics are
defined by the framework and cannot be configured by users):
- Anchors: ``target_module`` (the module wrapped by inner_wrapper,
  required) / ``module`` (the boundary module of @local_compute, optional
  declaration) -- the object the injection acts upon;
- Mesh family (**required**: all four must be declared and all are filled
  by the framework; the corresponding entry is None when the axis is not
  active): ``mesh`` (the active DTensor mesh of the current plan's
  coordinate system, with dp stripped -- consistent with the coordinate
  system of PrecompiledBoundary / resolve_placements), ``tp_mesh`` /
  ``cp_mesh`` / ``ep_mesh`` (the (edp, ep) expert mesh derived by D-10
  TP-extend-EP -- it is also the sharding domain of expert parameters,
  uniformly derived by the framework so that the a2a communication domain
  and the sharding domain are strictly identical);

Hard rules enforced by the decorators (fail-fast at import time):
- Required context must be declared in full: both decorators must declare
  ``mesh``/``tp_mesh``/``cp_mesh``/``ep_mesh``; ``@inner_wrapper`` must
  additionally declare ``target_module``;
- Context parameters **must not have default values** (the framework
  always fills them, so a default would be meaningless); a user
  configuring a reserved key of the same name in Target/YAML triggers
  fail-fast (reserved names are not configurable);
- Injected functions forbid ``*args`` / ``**kwargs`` -- the signature must
  be an explicit parameter list. This is the prerequisite for the policy
  "config keys are bound by name and typos must not be silently
  swallowed";
- All other named parameters are **user config keys**, coming entirely
  from explicit YAML/Target configuration with no automatic filling by
  the framework; config keys only accept data values -- **passing
  functions into an injected function is not allowed** (functions
  wrapping functions never ends; if you need custom behavior, write your
  own injected function and hard-code routing/layout logic in its body).

Runtime-layer validation (fail-fast at apply time):
- ``validate_local_compute_signature``: every parameter of the compute fn
  returned by the factory must have a same-named parameter in the
  original forward, appear in the same positional order, and catch all
  required parameters of forward -- "the injected function's arguments
  must match the original function's";
- ``validate_wrapped_forward``: the forward replaced by inner_wrapper must
  be able to bind all arguments of the original forward (a dummy-bind
  probe; the replacement side may pass through tolerantly with
  *args/**kwargs).
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Injection kinds (meta.kind)
LOCAL_COMPUTE = "local_compute"        # regional-compute factory (the only form of local_compute_fn)
INNER_WRAPPER = "inner_wrapper"        # inner-forward wrapper

# Reserved framework-context names per kind (declaring one means the
# framework fills it by name; configuring a key with the same name errors)
_MESH_FAMILY = frozenset({"mesh", "tp_mesh", "cp_mesh", "ep_mesh"})
FACTORY_CONTEXT = frozenset({"module"}) | _MESH_FAMILY
WRAPPER_CONTEXT = frozenset({"target_module"}) | _MESH_FAMILY

# Required context per kind (the decorators enforce full declaration --
# the mesh family is always passed by the framework and the user merely
# uses it; the framework fills None for axes that are not active)
FACTORY_REQUIRED = _MESH_FAMILY
WRAPPER_REQUIRED = frozenset({"target_module"}) | _MESH_FAMILY

_ALLOWED_CONTEXT = {
    LOCAL_COMPUTE: FACTORY_CONTEXT,    # optional module anchor + mesh family
    INNER_WRAPPER: WRAPPER_CONTEXT,
}
_REQUIRED_CONTEXT = {
    LOCAL_COMPUTE: FACTORY_REQUIRED,   # mesh family is required
    INNER_WRAPPER: WRAPPER_REQUIRED,
}

_DECORATOR_NAMES = {
    LOCAL_COMPUTE: "@local_compute",
    INNER_WRAPPER: "@inner_wrapper",
}


@dataclass(frozen=True)
class InjectionMeta:
    """Metadata written by the decorators onto the injected function object
    (``fn._injection_meta``).

    Attributes:
        kind: The injection kind (LOCAL_COMPUTE / INNER_WRAPPER).
        context: The declared framework-context keys.
    """
    kind: str                 # LOCAL_COMPUTE / INNER_WRAPPER
    context: frozenset        # declared framework-context keys


def _make_decorator(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Build the template decorator for one injection kind.

    Args:
        kind: The injection kind (LOCAL_COMPUTE or INNER_WRAPPER).

    Returns:
        A decorator that validates the injected function's signature at
        import time and stamps it with :class:`InjectionMeta`.
    """
    allowed = _ALLOWED_CONTEXT[kind]
    required = _REQUIRED_CONTEXT[kind]

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Validate ``fn`` against the injection discipline and stamp its meta.

        Args:
            fn: The injected function to decorate.

        Returns:
            The same function, with ``_injection_meta`` attached.

        Raises:
            TypeError: If the signature cannot be introspected, uses
                *args/**kwargs, gives a context parameter a default value,
                or omits required context parameters.
        """
        fname = getattr(fn, "__name__", fn)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{_DECORATOR_NAMES[kind]}: cannot introspect the signature of "
                f"{fn!r} (an injected function must be an introspectable "
                "plain callable)") from exc
        context = []
        for name, p in sig.parameters.items():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
                raise TypeError(
                    f"{_DECORATOR_NAMES[kind]} injected function {fname}"
                    f" must not use *args/**kwargs (parameter {name!r}) -- "
                    "the signature of an injected function must be an explicit "
                    "parameter list: context is declared by name and filled by "
                    "the framework, config keys are bound by name, and "
                    "**kwargs would silently swallow typos (conflicting with "
                    "the _check_target_config_keys policy)")
            if name in allowed:
                if p.default is not inspect.Parameter.empty:
                    raise TypeError(
                        f"{_DECORATOR_NAMES[kind]} injected function {fname}: "
                        f"context parameter {name!r} must not have a default "
                        "value -- context names are reserved by the framework "
                        "and are always filled by name, so a default would "
                        "never take effect")
                context.append(name)
        missing = sorted(required - set(context))
        if missing:
            raise TypeError(
                f"{_DECORATOR_NAMES[kind]} injected function {fname} is "
                f"missing required context parameters {missing} -- the "
                "injection discipline requires explicitly receiving "
                f"{sorted(required)} (all filled by the framework by name at "
                "apply time; the user merely uses them)")
        fn._injection_meta = InjectionMeta(kind=kind, context=frozenset(context))
        return fn

    return decorator


local_compute = _make_decorator(LOCAL_COMPUTE)
inner_wrapper = _make_decorator(INNER_WRAPPER)


def require_injection_meta(fn: Callable[..., Any], kind: str, *, source: str) -> InjectionMeta:
    """Fetch an injected function's metadata; fail-fast (with an instructive
    error) if it is undecorated or of the wrong kind.

    Args:
        fn: The function expected to carry injection metadata.
        kind: The expected injection kind.
        source: Call-site label prepended to error messages.

    Returns:
        The :class:`InjectionMeta` attached to ``fn``.

    Raises:
        TypeError: If ``fn`` lacks the template decorator or its decorator
            kind does not match ``kind``.
    """
    meta = getattr(fn, "_injection_meta", None)
    name = getattr(fn, "__name__", fn)
    if meta is None:
        raise TypeError(
            f"{source}: injected function {name} is missing the "
            f"{_DECORATOR_NAMES[kind]} decorator -- the explicit-injection "
            "discipline requires every injected function to carry a template "
            "decorator (the decorator declares the framework context it needs "
            "and guarantees the signature is validatable): use @local_compute "
            "for the runtime fn of local_compute_fn (a regional-compute "
            "factory) and @inner_wrapper for inner_wrapper (exported by "
            "hyper_parallel.auto_models.components.distributed)")
    if meta.kind != kind:
        raise TypeError(
            f"{source}: injected function {name} has the wrong decorator "
            f"kind -- got {_DECORATOR_NAMES[meta.kind]}, expected "
            f"{_DECORATOR_NAMES[kind]}")
    return meta


def fill_context_kwargs(meta: InjectionMeta,
                        context: Dict[str, Any],
                        configured: Optional[Dict[str, Any]] = None,
                        *,
                        source: str = "") -> Dict[str, Any]:
    """Collect the framework-filled kwargs for the context keys declared in
    ``meta`` (minimal, no hidden behavior).

    - The filled set equals the declared set, no more and no less (each fill
      is logged at INFO);
    - Context keys are framework-reserved names: a user configuring a
      same-named key in Target/YAML triggers fail-fast (context parameters
      have no defaults and the YAML resolver never back-fills them, so any
      reserved key appearing in ``configured`` was definitely written
      explicitly by the user).

    Args:
        meta: The injection metadata whose declared context keys are filled.
        context: The framework-provided context values keyed by name.
        configured: User-configured keys, checked against reserved names.
        source: Call-site label prepended to log and error messages.

    Returns:
        The kwargs dict mapping each declared context key to its
        framework-provided value.

    Raises:
        ValueError: If ``configured`` contains framework-reserved context
            keys.
    """
    configured = configured or {}
    reserved = sorted(set(configured) & meta.context)
    if reserved:
        raise ValueError(
            f"{source}: framework-reserved context keys {reserved} were "
            "configured -- context is filled by the framework according to "
            "the declaration and is not configurable; your config keys must "
            "not collide with reserved names")
    kwargs = {}
    for key in sorted(meta.context):
        kwargs[key] = context[key]
        logger.info("%s: context key %s filled by framework (%s)",
                    source, key, type(context[key]).__name__)
    return kwargs


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
