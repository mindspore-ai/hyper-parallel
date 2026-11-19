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
"""Lower recorded construction and forward strategies to source edits."""

from __future__ import annotations

import importlib
import inspect

from hyper_parallel.codegen.inline.ir import ImportPatch, ReplacementSpec, SourceObservation, StrategySpec
from hyper_parallel.codegen.inline.recognition import class_name_of, inner_wrapper_target


_CP_DISPATCH_SHELL = '''if self.cp_enable:
    return {helper_name}(self, __hp_forward_args__)

return self._forward_impl(__hp_forward_args__)
'''


def cp_dispatch_shell_body(helper_name: str) -> str:
    """Render CP dispatch around the extracted original forward.

    Args:
        helper_name: Generated helper that implements the selected CP forward.
    """
    return _CP_DISPATCH_SHELL.format(helper_name=helper_name)

#: The framework CP attention strategy kind (structure-keyed, not family-named).
CP_ATTENTION_KIND = "cp_attention"

_STRATEGY_CACHE: dict[tuple[str, bool], StrategySpec | None] = {}


#: Generic components are valid direct targets without a decorator. Model-owned
#: classes are accepted only when their native replacement decorator marks them.
_COMPONENTS_PREFIX = "hyper_parallel.components."


def _direct_component_class(target: str) -> type | None:
    """Import a generic or natively decorated replacement class."""
    module_name, _, attr = target.rpartition(".")
    if not module_name or not attr:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    component = getattr(module, attr, None)
    if not isinstance(component, type):
        return None
    if (
        target.startswith(_COMPONENTS_PREFIX)
        or getattr(component, "_hp_module_replacement", False)
    ):
        return component
    return None


def replacement_spec_for(
    target: str | None,
    model_type: str | None = None,
    *,
    module_type: str | None = None,
) -> ReplacementSpec | None:
    """Lower an explicitly observed component class from an older snapshot."""
    del model_type
    if target is None:
        return None
    component = _direct_component_class(target.replace(":", ".", 1))
    old_ctor = class_name_of(module_type)
    if component is None or not old_ctor:
        return None
    params = inspect.signature(component).parameters
    if "module" not in params:
        return None
    name = component.__name__
    return ReplacementSpec(
        old_ctor=old_ctor, new_ctor=name, mode="wrap_source", remove_class=False,
        imports=(ImportPatch(component.__module__, (name,)),),
        keyword_args=tuple(
            f"{key}={value}" for key, value in (("module_fqn", "''"), ("context", "{}")) if key in params
        ),
    )


def _family_wrap_spec(target: str, module_type: str | None) -> ReplacementSpec:
    """Lower an authoritative family request to the existing source wrapper form."""
    module_name, _, name = target.replace(":", ".", 1).rpartition(".")
    old_ctor = class_name_of(module_type)
    if not module_name or not name or not old_ctor:
        raise ValueError(f"Invalid prepared family replacement {target!r} for {module_type!r}")
    return ReplacementSpec(
        old_ctor=old_ctor, new_ctor=name,
        imports=(ImportPatch(module_name, (name,)),), mode="wrap_source",
        keyword_args=("module_fqn=''",), remove_class=False, defer_to_fqns=True,
    )


def replacement_spec_for_observation(
    rule: SourceObservation, model_type: str | None,
) -> ReplacementSpec | None:
    """Render a prepared family directly or lower a supported legacy replacement."""
    if rule.prepared_family:
        if rule.replace_target is None:
            raise ValueError("Prepared family replacement has no construction target")
        return _family_wrap_spec(rule.replace_target, rule.module_type)
    spec = replacement_spec_for(rule.replace_target, model_type, module_type=rule.module_type)
    if spec is None and rule.opaque_legacy:
        raise RuntimeError(
            f"Opaque legacy replacement {rule.replace_target!r} at {rule.match!r} "
            "has no replay record or inline declaration"
        )
    return spec


def strategy_spec_for(
    target: str | None,
    model_type: str | None = None,
    *,
    inner_wrapper: bool = False,
) -> StrategySpec | None:
    """Resolve a strategy target from structure.

    A recorded CP ``inner_wrapper`` target resolves to the
    framework CP attention strategy (imports only). EP mutations come from the
    builder recorder instead of secondary structural inference. Anything else
    is unresolvable and yields ``None`` — recognition never guesses a strategy.

    Args:
        target: Import path of the selected strategy.
        model_type: Optional source model type.
        inner_wrapper: Whether the strategy wraps the existing forward boundary.
    """
    if target is None:
        return None
    del model_type
    key = (target, inner_wrapper)
    if key in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[key]
    spec = _resolve_strategy_spec(target, inner_wrapper)
    _STRATEGY_CACHE[key] = spec
    return spec


def _resolve_strategy_spec(target: str, inner_wrapper: bool) -> StrategySpec | None:
    if inner_wrapper:
        return _cp_strategy_spec(target)
    return None


def _cp_strategy_spec(target: str) -> StrategySpec:
    """Build the CP strategy spec: an in-place dispatch shell plus imports.

    A recognizable inner forward rewrite participates in inline CP: the
    materialization pass derives an
    in-artifact rewrite of the attention host whose forward falls back to
    ``_forward_impl`` when CP is disabled and delegates to the helper when
    active. The helper is the same kernel the runtime CP wrapper delegates
    to (trainer and codegen share one CP implementation), and the import is
    derived from the wrapper factory source, so this module never hardcodes a
    family helper and
    stays free of torch imports at resolution time.  ``subclass_state_attr``
    carries the family kernel's companion attribute so the runtime binding
    satisfies the identical contract the native wrapper installs.

    Families without an inline helper keep the original framework CP
    strategy (imports only), preserving the runtime wrapper path.
    """
    resolved = inner_wrapper_target(target)
    if resolved is None:
        return StrategySpec(kind=CP_ATTENTION_KIND, imports=())
    if resolved.inline_body is not None:
        return StrategySpec(
            kind=CP_ATTENTION_KIND,
            subclass_shell=resolved.inline_body,
            imports=(),
        )
    return StrategySpec(
        kind=CP_ATTENTION_KIND,
        subclass_shell=cp_dispatch_shell_body(resolved.helper_name),
        subclass_state_attr=resolved.state_attr,
        imports=(ImportPatch(resolved.helper_module, (resolved.helper_name,)),),
    )


__all__ = [
    "CP_ATTENTION_KIND",
    "cp_dispatch_shell_body",
    "replacement_spec_for",
    "replacement_spec_for_observation",
    "strategy_spec_for",
]
