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
"""Install and verify source replacements carried by a prepared model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import replace
from typing import Any
import weakref

from torch import nn

from hyper_parallel.components.modules.module_family import FamilyRequest, resolve_family
from hyper_parallel.models.preparation_manifest import _encode_construction_value, _replacement_replay
from hyper_parallel.models.prepared_family import (
    PREPARED_FAMILY_RECORD_VERSION,
    _decode_construction_value,
    _load_construction_symbol,
    build_prepared_family_replacement,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementPlan,
    ModuleReplacementRequest,
    ModuleReplacementSpec,
    ModuleReplacementTarget,
    _factory_for_request,
    _module_aliases,
    _replacement_rule_id,
    apply_module_replacements,
    module_replacement,
)


_TOPOLOGY_FIELDS = ("tp_size", "cp_size", "ep_size", "pp_size", "sequence_parallel", "loss_parallel")


def _mismatch(detail: str) -> ValueError:
    """Report a stale artifact in terms of its recorded boundary."""
    return ValueError(f"Prepared bundle mismatch: {detail}. Regenerate the bundle from the current recipe.")


def _source_constructor(record: Mapping[str, Any]):
    """Restore the original shared request and the recorded candidate constructor."""
    replay = record["replay"]
    target = _load_construction_symbol(replay["target"])
    if replay["request"]:
        request = ModuleReplacementRequest(
            tuple(record["module_fqns"]), target, kwargs=_decode_construction_value(replay["arguments"]),
        )
        factory = _factory_for_request(request)
    else:
        request, factory = None, target
    return request, factory


def _prepare_factory(record, factory, source_type, expected_type, generated_type):
    """Retain the recorded concrete member and optional emitted forward subclass."""
    selection = record.get("family_request", {}).get("selection")

    @module_replacement
    def build(*, module, module_fqn, context):
        if type(module) is not source_type:  # pylint: disable=unidiomatic-typecheck
            # A copied source class can have a generated module identity.
            # A shallow view preserves its tensors and the native type checks.
            module = copy(module)
            module.__class__ = source_type
        if selection is None:
            candidate = factory(module=module, module_fqn=module_fqn, context=context)
        else:
            candidate = build_prepared_family_replacement(module=module, module_fqn=module_fqn, prepared=selection)
        if type(candidate) is not expected_type:  # pylint: disable=unidiomatic-typecheck
            raise _mismatch(f"{module_fqn!r} installed {type(candidate)!r}, expected {expected_type!r}")
        if generated_type is not expected_type:
            # The emitted subclass supplies forwards only. The recorded native
            # constructor owns parameters, state and checkpoint conversion.
            candidate.__class__ = generated_type
        return candidate

    return build


def _recorded_target(model, record, aliases, namespace):
    paths = tuple(record["module_fqns"])
    if not paths or len(set(paths)) != len(paths):
        raise _mismatch("invalid replacement aliases")
    try:
        source = model.get_submodule(paths[0])
    except AttributeError as error:
        raise _mismatch(f"missing source {paths[0]!r}") from error
    if tuple(sorted(paths)) != tuple(sorted(aliases[id(source)][1])):
        raise _mismatch(f"source aliases changed at {paths[0]!r}")
    source_type = _load_construction_symbol(record["source_type"])
    actual_type = type(source)
    if actual_type is not source_type and (
        actual_type.__module__ != namespace["__name__"] or actual_type.__qualname__ != source_type.__qualname__
    ):
        raise _mismatch(f"source type changed at {paths[0]!r}")
    if record["replay"]["version"] != PREPARED_FAMILY_RECORD_VERSION:
        raise _mismatch(f"unsupported replacement record version at {paths[0]!r}")
    expected = _load_construction_symbol(record["replacement_type"])
    generated = namespace.get(expected.__name__, expected)
    if not isinstance(generated, type) or not issubclass(generated, expected):
        generated = expected
    request, factory = _source_constructor(record)
    spec = ModuleReplacementSpec(
        paths, _prepare_factory(record, factory, source_type, expected, generated),
        declaration_id=record["rule_id"], request=request,
    )
    return ModuleReplacementTarget(paths, weakref.ref(source), spec), source_type, generated


def install_prepared_components(
    model: nn.Module, records: Sequence[Mapping[str, Any]], topology: Mapping[str, Any], namespace: Mapping[str, Any],
) -> None:
    """Replay a bundle against its original tree using the shared atomic installer."""
    inventory = _module_aliases(model)
    staged = [_recorded_target(model, record, inventory, namespace) for record in records]
    targets = tuple(item[0] for item in staged)
    if len({id(target.source) for target in targets}) != len(targets):
        raise _mismatch("duplicate prepared replacement boundary")
    weights = []
    if targets:
        plan = ModuleReplacementPlan(targets, tuple(target.spec for target in targets), atomic=True)
        context = _decode_construction_value(records[0]["replay"]["context"])
        apply_module_replacements(model, plan, weights_mapping=weights, context=context)
    for record, (target, source_type, installed_type) in zip(records, staged):
        candidate = model.get_submodule(target.module_fqns[0])
        candidate._hp_preinstalled_source_type = source_type  # pylint: disable=protected-access
        candidate._hp_preinstalled_record = record  # pylint: disable=protected-access
        candidate._hp_preinstalled_type = installed_type  # pylint: disable=protected-access
    model._hp_preinstalled_boundaries = tuple(
        (target.module_fqns, weakref.ref(model.get_submodule(target.module_fqns[0])))
        for target in targets
    )
    model._hp_preinstalled_topology = {name: topology[name] for name in _TOPOLOGY_FIELDS if name in topology}
    model._hp_preinstalled_conversions = tuple(weights)


def validate_preinstalled_topology(model: nn.Module, context: Mapping[str, Any]) -> None:
    """Reject topology changes before preparing another family member."""
    for name, expected in getattr(model, "_hp_preinstalled_topology", {}).items():
        if context.get(name) != expected:
            raise _mismatch(f"{name} is {context.get(name)!r}, recorded {expected!r}")


def _validate_family_selection(record, request, context, path):
    selection = record.get("family_request", {}).get("selection")
    if selection is None:
        return
    if request is None or selection["version"] != PREPARED_FAMILY_RECORD_VERSION:
        raise _mismatch(f"family request or record version changed at {path!r}")
    # The raw configured call already matched. Resolve the normalized native
    # request, which excludes construction-only arguments such as config.
    arguments = _decode_construction_value(selection["request_arguments"])
    current = resolve_family(FamilyRequest(request.target, arguments), context=context)
    if (current.variant != selection["variant"]
            or current.constructor is not _load_construction_symbol(selection["member"])
            or dict(current.selectors) != selection["selectors"]
            or _encode_construction_value(current.arguments) != selection["resolved_arguments"]):
        raise _mismatch(f"family member or selectors changed at {path!r}")


def _validate_installed_target(target, module, record, rule_id, context):
    path = target.module_fqns[0]
    if record is None or type(module) is not getattr(module, "_hp_preinstalled_type", None):  # pylint: disable=C0123
        raise _mismatch(f"missing record or installed member at {path!r}")
    if tuple(record["module_fqns"]) != target.module_fqns or record["rule_id"] != rule_id:
        raise _mismatch(f"declaration or aliases changed at {path!r}")
    replay = _replacement_replay(target.spec, context)
    if any(replay.get(name) != record["replay"][name] for name in ("target", "arguments", "request")):
        raise _mismatch(f"replacement target or arguments changed at {path!r}")
    _validate_family_selection(record, target.spec.request, context, path)


def reconcile_preinstalled_components(
    model: nn.Module, plan: ModuleReplacementPlan, context: Mapping[str, Any],
) -> ModuleReplacementPlan:
    """Remove already satisfied targets from the shared compiled plan."""
    if not hasattr(model, "_hp_preinstalled_topology"):
        return plan
    actual = {
        identity: (module, getattr(module, "_hp_preinstalled_record", None))
        for identity, (module, _) in _module_aliases(model).items()
        if getattr(module, "_hp_preinstalled_source_type", None) is not None
    }
    for paths, installed in model._hp_preinstalled_boundaries:  # pylint: disable=protected-access
        if id(installed()) not in actual:
            raise _mismatch(f"missing installed replacement at {paths[0]!r}")
    rule_ids = {id(spec): _replacement_rule_id(spec, index) for index, spec in enumerate(plan.specs)}
    pending = []
    for target in plan.targets:
        match = actual.pop(id(target.source), None)
        if match is None:
            pending.append(target)
            continue
        _validate_installed_target(target, *match, rule_ids[id(target.spec)], context)
    if actual:
        raise _mismatch("the model contains a replacement with no matching current declaration")
    return replace(plan, targets=tuple(pending))
