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
"""One model-build context and effective spec shared by direct and source paths."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Callable

import torch
from torch import nn

from hyper_parallel.components.modules.module_family import (
    FamilyBoundary, FamilyBuildInfo, _build_family_member, _context_from_mesh, collect_family_modules, resolve_family,
    validate_family_context,
)
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import _normalize_family_key, get_model_adapter
from hyper_parallel.models.replacement import (
    ModuleReplacementPlan, _apply_module_replacement_actions, _install_module_candidates,
    _validate_replacement, compile_module_replacements,
)
from hyper_parallel.components.checkpoint.weight_conversion import get_model_conversion_mapping


def resolve_effective_model_spec(config: Any, explicit: ModelAdapterSpec | None = None) -> ModelAdapterSpec | None:
    """Resolve once without changing process-wide registration.

    Args:
        config: Model identity, independent of a constructed source object.
        explicit: Complete per-build declaration replacing the builtin spec.

    Returns:
        The effective immutable spec or None for a fully self-described model.
    """
    identities = tuple(value for value in (
        getattr(config, "model_type", None), *(getattr(config, "architectures", None) or ()),
    ) if value)
    if explicit is not None:
        if not isinstance(explicit, ModelAdapterSpec):
            raise TypeError("model_spec must be ModelAdapterSpec")
        accepted = {_normalize_family_key(explicit.model_type), _normalize_family_key(explicit.architecture)}
        if identities and not accepted.intersection(_normalize_family_key(value) for value in identities):
            raise ValueError("explicit model_spec does not match the model configuration identity")
        return explicit
    for identity in identities:
        spec = get_model_adapter(identity)
        if spec is not None:
            return spec
    return None


def build_module_context(
    distributed_setup: Any = None,
    *,
    low_precision_config: Any = None,
    device: Any = None,
    dtype: Any = None,
    model_spec: ModelAdapterSpec | None = None,
) -> Mapping[str, Any]:
    """Create the read-only logical context for family selection and model preparation.

    Args:
        distributed_setup: Existing job setup; no new topology configuration.
        low_precision_config: Optional normalized numerical scheme.
        device: Target device, if already known.
        dtype: Target model dtype, if already known.
        model_spec: Already resolved per-build model facts.

    Returns:
        Logical axes and target constraints, without creating groups or workspaces.
    """
    mesh = getattr(distributed_setup, "mesh_context", None)
    result = {}
    for axis in ("tp", "cp", "ep", "pp"):
        size = getattr(mesh, f"{axis}_size", 1)
        actual_type = type(size)
        if actual_type is not int or size < 1:
            raise ValueError(f"{axis}_size must be a positive integer")
        result[f"{axis}_size"] = size
        result[axis] = size > 1
    policy = low_precision_config
    if policy is None:
        policy = getattr(distributed_setup, "low_precision_config", None)
    result["low_precision"] = policy if getattr(policy, "enabled", False) else None
    result["legacy_low_precision_targets"] = bool(getattr(distributed_setup,
                                                          "_hp_legacy_low_precision_targets", False))
    result["sequence_parallel"] = bool(getattr(mesh, "sequence_parallel", False))
    result["loss_parallel"] = bool(getattr(mesh, "loss_parallel", False))
    result["mesh_dim_names"] = tuple(getattr(getattr(mesh, "device_mesh", None), "mesh_dim_names", ()) or ())
    result["mesh"] = mesh
    result["model_spec"] = model_spec
    result["module_replacement_requests"] = getattr(distributed_setup, "module_replacement_requests", None)
    result["module_replacements"] = getattr(distributed_setup, "module_replacements", None)
    if device is not None:
        result["device_type"] = torch.device(device).type
    if dtype is not None and dtype != "auto":
        result["dtype"] = str(dtype).removeprefix("torch.")
    return MappingProxyType(result)


def attach_build_context(model: nn.Module, context: Mapping[str, Any]) -> None:
    """Attach this build's explicit spec and context, without registering either globally.

    Args:
        model: Final or source model tree for this build.
        context: Read-only job context and effective model facts.
    """
    if (getattr(model, "_hp_components_prepared", False)
            and getattr(model, "_hp_module_build_context", None) is not context):
        raise ValueError("prepared model components cannot be reused with another build context")
    model._hp_module_build_context = context  # pylint: disable=protected-access
    model._hp_model_adapter_spec = context.get("model_spec")  # pylint: disable=protected-access


def _adopt_family_state(source: nn.Module, candidate: nn.Module, fqn: str) -> None:
    """Retain logical tensors when selecting another member of the same family."""
    source_contract = getattr(source, "state_contract", None)
    candidate_contract = getattr(candidate, "state_contract", None)
    original = source_contract() if source_contract is not None else None
    selected = candidate_contract() if candidate_contract is not None else None
    if original != selected:
        raise ValueError(f"family at {fqn!r} changed its logical state contract during configuration")
    for kind in ("parameter", "buffer"):
        source_values = dict(getattr(source, f"named_{kind}s")(remove_duplicate=False))
        candidate_values = dict(getattr(candidate, f"named_{kind}s")(remove_duplicate=False))
        if source_values.keys() != candidate_values.keys():
            raise ValueError(f"family at {fqn!r} changed logical {kind} names during configuration")
        for name, value in source_values.items():
            if value.shape != candidate_values[name].shape:
                raise ValueError(f"family at {fqn!r} changed logical {kind} shape for {name!r}")
            parent, _, leaf = name.rpartition(".")
            owner = candidate.get_submodule(parent)
            if kind == "parameter":
                owner.register_parameter(leaf, value)
            else:
                persistent = leaf not in owner._non_persistent_buffers_set  # pylint: disable=protected-access
                owner.register_buffer(leaf, value, persistent=persistent)
    for name, original in source.named_modules():
        target = candidate.get_submodule(name)
        target.training = original.training
        if type(original).get_extra_state is not nn.Module.get_extra_state:
            target.set_extra_state(deepcopy(original.get_extra_state()))
    _validate_replacement(source, candidate, fqn)


def _validate_family_lifetime(source: nn.Module) -> None:
    """Reject changes after a member has entered distributed execution."""
    if any(getattr(child, "_hp_parallel_binding", None) is not None
           or getattr(child, "hsdp_scheduler", None) is not None for child in source.modules()):
        raise ValueError("family configuration must finish before communication binding and FSDP")
    if any(parameter.grad is not None for parameter in source.parameters()):
        raise ValueError("family configuration requires cleared gradients and must finish before optimizer creation")


def _prepare_family_member(
    boundary: FamilyBoundary, context: Mapping[str, Any],
) -> tuple[nn.Module, FamilyBuildInfo]:
    """Resolve a saved request, preparing a candidate without changing the model."""
    source, info = boundary.module, boundary.info
    if info.layout is not None:
        validate_family_context(source, context)
        return source, info
    context = _context_from_mesh(None, context)
    member = resolve_family(info.request, context=context)
    previous = info.resolved
    axis = member.family.parallel_axis
    same_topology = axis is None or member.axis_sizes[axis] == previous.axis_sizes[axis]
    unchanged = (member.constructor is previous.constructor
                 and member.selectors == previous.selectors and same_topology)
    if not unchanged or member.axis_sizes != previous.axis_sizes:
        _validate_family_lifetime(source)
    if unchanged:
        if member.family.validate is not None:
            member.family.validate(config=info.config_snapshot, module=None if info.origin == "direct" else source,
                                   context=context, member=member)
        return source, replace(info, resolved=member)
    if info.origin == "direct":
        # A meta skeleton avoids reinitializing or copying the caller's existing weights.
        with torch.device("meta"):
            candidate = _build_family_member(info.request, member, config=info.config_snapshot,
                                             module_fqn=boundary.fqns[0], context=context)
    else:
        candidate = _build_family_member(info.request, member, module=source,
                                         module_fqn=boundary.fqns[0], context=context)
    _adopt_family_state(source, candidate, boundary.fqns[0])
    return candidate, replace(info, resolved=member)


def _finalize_family_modules(model: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    """Finalize family requests atomically before layout, keeping all module aliases."""
    boundaries = collect_family_modules(model)
    if not boundaries:
        return model
    prepared = [(boundary, *_prepare_family_member(boundary, context)) for boundary in boundaries]
    model = _install_module_candidates(
        model, ((boundary.fqns, boundary.module, candidate) for boundary, candidate, _ in prepared),
    )
    for _, candidate, info in prepared:
        candidate._hp_family_build_info = info  # pylint: disable=protected-access
    return model


def construct_family_model(
    factory: Callable, config: Any, *, context: Mapping[str, Any],
) -> nn.Module:
    """Call a model factory once and attach the shared preparation context.

    Args:
        factory: Model constructor accepting config and optionally keyword mesh/context.
        config: Complete model configuration.
        context: The already resolved job context, shared by nested family calls.

    Returns:
        A constructed model that must pass through prepare_model_components before layout or codegen.
    """
    if not callable(factory):
        raise TypeError("model_factory must be callable")
    parameters = inspect.signature(factory).parameters
    runtime = {"context": context}
    mesh = context.get("mesh")
    runtime["mesh"] = mesh if getattr(mesh, "device_mesh", None) is not None else None
    arguments = {name: value for name, value in runtime.items() if name in parameters}
    inspect.signature(factory).bind(config, **arguments)
    model = factory(config, **arguments)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return nn.Module")
    attach_build_context(model, context)
    return model


def _provider_rules(provider: Callable, model: nn.Module, context: Mapping[str, Any]) -> tuple:
    """Load default module replacement requests from the effective model specification."""
    signature = inspect.signature(provider)
    try:
        signature.bind(model=model, context=context)
    except TypeError:
        signature.bind()
        value = provider()
    else:
        value = provider(model=model, context=context)
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(value)
    rules = getattr(value, "get_replacement_rules", None)
    return tuple(rules(model=model, context=context)) if rules is not None else ()


def prepare_model_components(
    model: nn.Module, context: Mapping[str, Any], *, capture_checkpoint_metadata: bool = True,
) -> tuple[nn.Module, Any]:
    """Finalize family choices and source components once before parameter layout.

    Args:
        model: Final or source model tree for this build.
        context: Read-only job context and effective model facts.
        capture_checkpoint_metadata: Save source shapes when loading transformed checkpoint tensors.
    """
    if getattr(model, "_hp_components_prepared", False):
        if getattr(model, "_hp_module_build_context", None) is not context:
            raise ValueError("prepared model components cannot be reused with another build context")
        return model, model._hp_family_weight_mapping  # pylint: disable=protected-access
    model = _finalize_family_modules(model, context)
    attach_build_context(model, context)
    spec = context.get("model_spec")
    defaults = () if spec is None or spec.replacements is None else _provider_rules(spec.replacements, model, context)
    overrides = context.get("module_replacement_requests")
    legacy = () if overrides is not None else (context.get("module_replacements") or ())
    plan = compile_module_replacements(model, legacy, defaults=defaults, overrides=overrides or ())
    weights = model_weight_mapping(model)
    _validate_low_precision_targets(model, plan, context)
    if plan.targets:
        model, weights = _apply_module_replacement_actions(
            model, plan, weights_mapping=weights, context=context,
            capture_checkpoint_metadata=capture_checkpoint_metadata,
        )
    model._hp_components_prepared = True  # pylint: disable=protected-access
    model._hp_family_weight_mapping = weights  # pylint: disable=protected-access
    return model, weights


def model_weight_mapping(model: nn.Module) -> list:
    """Read declared conversion rules without importing HF modeling for a plain Module.

    Args:
        model: Concrete source or directly composed model.

    Returns:
        Model-owned or HF-normalized checkpoint conversion rules.
    """
    if getattr(model, "config_class", None) is not None:
        return get_model_conversion_mapping(model)
    declared = getattr(model, "_weight_conversions", None)
    if declared is not None:
        return list(declared)
    return []


def _validate_low_precision_targets(
    model: nn.Module, plan: ModuleReplacementPlan, context: Mapping[str, Any],
) -> None:
    """Validate a policy after direct members and model-owned defaults are known."""
    if context.get("low_precision") is None:
        return
    if context.get("legacy_low_precision_targets"):
        return
    if context.get("module_replacement_requests") is None and context.get("module_replacements"):
        return
    if any(boundary.info.resolved.selectors.get("precision", "inherit") != "inherit"
           for boundary in collect_family_modules(model)):
        return
    for target in plan.targets:
        request = target.spec.request
        if request is not None and (request.kwargs.get("precision", "inherit") != "inherit"
                                    or request.kwargs.get("use_mxfp8") is True):
            return
    raise ValueError("low_precision is enabled but no low-precision family or module replacement is configured")
