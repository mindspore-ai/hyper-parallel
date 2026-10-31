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
"""Parallel topology, activation-checkpoint and plan-override configuration.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5).
``PlanOverride`` stays the YAML DTO that desugars to
``auto_models.distributed.recipe_spec.ModuleShardingSpec``.
"""
# pylint: disable=unused-import

import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, List, Literal, Optional, Union

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.config import (
    LowPrecisionConfig,
    LowPrecisionDtypeScheme,
)
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.models.replacement import (
    MethodReplacementSpec,
    ModuleReplacementFactory,
    ModuleReplacementSpec,
    method_replacement,
    module_replacement,
)

from hyper_parallel.components.modules.family import get_module_family
from hyper_parallel.trainer.config.replacement import ReplacementTarget
from hyper_parallel.trainer.config.target import Target, _serialize_config_value

logger = logging.getLogger(__name__)


@dataclass
class AcceleratorConfig:
    """Parallel topology and target-selected strategy configurations."""

    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    sequence_parallel: bool = False
    loss_parallel: bool = False
    encoder_parallel: EncoderParallelConfig = field(default_factory=EncoderParallelConfig)


@dataclass
class ActivationCheckpointSelection:
    """Select adapter-declared recompute regions and exact layer coverage.

    ``layer_count`` selects the contiguous prefix ``[0, layer_count)`` from
    the flattened layer-container order. ``layer_indices`` selects explicit
    zero-based entries from that same order.

    ``model_adapter_safe_regions`` wraps the outermost module matching the
    adapter's ``region_patterns`` inside every selected layer and keeps every
    ``exclude_patterns`` descendant outside the recompute region. Which shape that
    produces is the model's declaration, not the Trainer's: leaf patterns give
    one region per submodule, a layer pattern gives one region per layer.
    """

    source: Literal["default", "model_adapter_safe_regions"] = "default"
    layer_count: Optional[int] = None
    layer_indices: Optional[List[int]] = None

    def __post_init__(self) -> None:
        """Validate the mutually exclusive layer-count and layer-index selectors."""
        self._validate_selector_choice()
        self._validate_selector_values()

    def _validate_selector_choice(self) -> None:
        """Require one supported source and its matching selector form."""
        if self.source not in ("default", "model_adapter_safe_regions"):
            raise ValueError(
                "activation_checkpoint.selection.source must be default or "
                "model_adapter_safe_regions"
            )
        selectors = (self.layer_count is not None, self.layer_indices is not None)
        if self.source == "default" and any(selectors):
            raise ValueError(
                "activation_checkpoint.selection.layer_count and layer_indices "
                "are only valid when "
                "selection.source=model_adapter_safe_regions"
            )
        if self.source == "model_adapter_safe_regions" and sum(selectors) != 1:
            raise ValueError(
                "activation_checkpoint.selection requires exactly one of "
                "layer_count or layer_indices when "
                "selection.source=model_adapter_safe_regions"
            )

    def _validate_selector_values(self) -> None:
        """Require non-negative, distinct layer numbers."""
        if self.layer_count is not None and (
            isinstance(self.layer_count, bool)
            or not isinstance(self.layer_count, int)
            or self.layer_count < 0
        ):
            raise ValueError(
                "activation_checkpoint.selection.layer_count must be a "
                "non-negative integer"
            )
        if self.layer_indices is None:
            return
        if not isinstance(self.layer_indices, list) or not self.layer_indices:
            raise ValueError(
                "activation_checkpoint.selection.layer_indices must be a "
                "non-empty list of non-negative integers"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.layer_indices
        ):
            raise ValueError(
                "activation_checkpoint.selection.layer_indices must contain "
                "only non-negative integers"
            )
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError(
                "activation_checkpoint.selection.layer_indices must not "
                "contain duplicates"
            )


@dataclass
class ActivationCheckpointConfig:
    """Activation-checkpoint options exposed by the initial YAML schema.

    ``swap_inputs`` is consumed only when ``mode`` is ``"full"`` or
    ``"selective"``.
    """

    mode: Optional[Literal["off", "full", "selective"]] = "off"
    swap_inputs: bool = False
    selection: ActivationCheckpointSelection = field(
        default_factory=ActivationCheckpointSelection
    )
    # Escape hatch for SAC's cached-tensor mutation check: safe when
    # every in-place rewrite of a cached operator output is
    # deterministic, so the recompute replays the same values (the
    # CSA indexer's masked_fill_ family).
    allow_cache_entry_mutation: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous values for activation input swapping."""
        if not isinstance(self.swap_inputs, bool):
            raise TypeError("activation_checkpoint.swap_inputs must be a bool")
        if not isinstance(self.allow_cache_entry_mutation, bool):
            raise TypeError(
                "activation_checkpoint.allow_cache_entry_mutation must be a bool")


def _resolve_replace_implementation(value: Union[str, Target[Any]]) -> Target[Any]:
    """Resolve a replacement implementation without invoking its callable."""
    if isinstance(value, Target):
        return value
    if not isinstance(value, str) or not value.strip():
        raise TypeError("replace.implementation must be a non-empty import path or Target")
    parts = value.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            symbol = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise ValueError(f"replace.implementation {value!r} failed while importing {exc.name!r}") from exc
        except ImportError as exc:
            raise ValueError(f"replace.implementation {value!r} failed while importing: {exc}") from exc
        for attribute in parts[split_at:]:
            if not hasattr(symbol, attribute):
                raise ValueError(f"replace.implementation {value!r} has no attribute {attribute!r}")
            symbol = getattr(symbol, attribute)
        if not callable(symbol):
            raise TypeError(f"replace.implementation {value!r} must name a callable")
        return Target(symbol, target_path=value)
    raise ValueError(f"replace.implementation {value!r} could not be imported")


@dataclass(frozen=True)
class ReplaceAction:
    """One public module or method replacement declaration."""

    kind: Literal["module", "method"]
    implementation: Union[str, Target[Any]]
    source_type: Optional[str] = None
    exact_type: bool = False

    def __post_init__(self) -> None:
        """Validate kind-specific fields and normalize the delayed target."""
        if self.kind not in ("module", "method"):
            raise ValueError("replace.kind must be 'module' or 'method'")
        if not isinstance(self.exact_type, bool):
            raise TypeError("replace.exact_type must be a bool")
        if self.kind == "module":
            if self.source_type is not None and (
                    not isinstance(self.source_type, str) or not self.source_type.strip()):
                raise ValueError("replace.source_type must be a non-empty import path when provided")
            if self.exact_type and self.source_type is None:
                raise ValueError("replace.exact_type requires replace.source_type")
        elif self.source_type is not None or self.exact_type:
            raise ValueError("replace.source_type and replace.exact_type are only valid for kind='module'")
        object.__setattr__(
            self, "implementation", _resolve_replace_implementation(self.implementation)
        )


@dataclass
class PlanOverride:
    """One plan_overrides entry (YAML ``plan_overrides`` list item).

    This is the **YAML transport form** of ``ShardingPlanner(plan_overrides=
    {...})`` — the planner's single override interface. Placement objects are
    not YAML-serializable, so contract fields use the string DSL
    (``"replicate"`` / ``"partial"`` / ``"shard(N)"``) plus the sentinels
    ``"auto"`` (explicit inherit) / ``"none"`` (explicit clear); they are
    desugared to real objects in ``to_override()`` — the planner never sees
    strings. The desugar happens trainer-side in
    ``instantiate_infrastructure`` BEFORE planner construction.

    Fields:
        replace: public replacement action. ``kind='module'`` requires a
            decorated module factory at ``implementation``; optional
            ``source_type`` checks the expected source module type;
            ``kind='method'`` selects a literal method suffix in ``match``;
            ``.forward`` uses the existing boundary executor and defaults to
            ``region_dispatch=False``, while other methods use a decorated
            ``@method_replacement`` factory before sharding.
        match: module FQN/glob, or module glob plus a literal method suffix for
            ``kind='method'``. Boundary globs first merge matching planner boundaries;
            a glob with a concrete params/I/O contract also expands against
            the final model's ``named_modules`` and creates missing boundaries.
            A merge-only glob fails if it matches a real module that has no
            planner boundary, with a suggested contract skeleton.
            ``*`` spans dots, so ``"*.self_attn"`` hits
            ``model.layers.0.self_attn``.
        when: optional activation condition. Sharding actions accept ``"cp"``
            (active when cp_size>1), ``"ep"`` (ep_size>1),
            ``"sequence_parallel"``, or ``"low_precision"`` (active when
            online low precision is enabled). Module replacement accepts only
            ``"low_precision"`` and cannot depend on the parallel topology.
        low_precision_dtype_scheme: a named ``training.low_precision.
            dtype_schemes`` entry bound to this match. Only valid on
            ``when="low_precision"`` replacement entries; the resolved scheme
            is injected into the replacement factory context so the strategy
            factory never resolves a catalog at replacement time.
        Forward replacements use ``@forward_replacement`` or ``@local_compute``
            factories through ``replace.implementation``.
        region_dispatch: validate execution-mode declaration; forward method
            replacement defaults to ``false``. ``false`` = the region
            computation (the module's own forward or the injected function)
            cannot be dispatched: it contains communication primitives /
            custom kernels (e.g. an in-house EP-aware MoE all-to-all inside
            forward, or the in-repo CP/EP reference implementations), so it
            runs as a local-region skeleton black box; ``true`` = the
            injected code is pure standard operators (fused kernels /
            scripted-style optimizations), so under validate the dispatch
            passes through and out_src is genuinely checked. This field is
            not needed without an injection (ordinary boundaries pass through
            naturally).
        params / in_src / in_dst / out_src / out_dst: contract fields in the
            YAML form ``{name: {axis: placement_str}}`` (out_* also accept
            the scalar shorthand ``{axis: placement_str}``), or the sentinels
            ``"auto"`` / ``"none"``. Merge mode (match hits a derived
            boundary): an omitted field inherits and an explicit ``{}``
            clears. Insert mode (misses every boundary): nothing can inherit;
            at least one concrete parameter/I/O field must declare the new
            boundary, and all fields required by that module's contract must
            be written explicitly.
        tp_divide_attrs: optional module-instance integer attributes divided
            exactly by the active TP size when the module forward runs on
            local tensors. Omit for no user adjustment; an explicit empty
            list clears an inherited glob declaration.
    """

    match: Union[str, List[str]]
    replace: Optional[ReplaceAction] = None
    when: Optional[Literal[
        "cp", "ep", "sequence_parallel", "low_precision"
    ]] = None
    low_precision_dtype_scheme: Optional[str] = None
    region_dispatch: Optional[bool] = None
    params: Optional[Any] = None
    in_src: Optional[Any] = None
    in_dst: Optional[Any] = None
    out_src: Optional[Any] = None
    out_dst: Optional[Any] = None
    tp_divide_attrs: Optional[List[str]] = None

    # Contract fields (string sentinels are resolved at planner merge time;
    # insert mode rejects them)
    _CONTRACT_FIELDS = ("params", "in_src", "in_dst", "out_src", "out_dst")
    _BOUNDARY_FIELDS = _CONTRACT_FIELDS + ("tp_divide_attrs",)

    def __post_init__(self) -> None:
        """Validate the public replacement action and its boundary contract."""
        if self.replace is None:
            return
        if not isinstance(self.replace, ReplaceAction):
            raise TypeError("plan_overrides.replace must be a ReplaceAction")
        if self.replace.kind == "module":
            self._validate_module_replacement()
            return
        self._validate_method_replacement()

    def _validate_module_replacement(self) -> None:
        """Validate fields exclusive to a module replacement rule."""
        if self.when not in (None, "low_precision"):
            raise ValueError("kind='module' only supports when='low_precision'")
        if self.region_dispatch is not None:
            raise ValueError("region_dispatch cannot appear with kind='module'")
        if any(getattr(self, name) is not None for name in self._BOUNDARY_FIELDS):
            raise ValueError("kind='module' must use its own plan_overrides rule")

    def _validate_method_replacement(self) -> None:
        """Validate a literal method match and dispatch to its protocol checks."""
        if isinstance(self.match, list):
            raise ValueError(f"kind={self.replace.kind!r} requires one string match per rule")
        if self.low_precision_dtype_scheme is not None:
            raise ValueError("low_precision_dtype_scheme is only valid for kind='module'")
        if not isinstance(self.match, str):
            raise TypeError("kind='method' match must be a string")
        owner_pattern, separator, method_name = self.match.rpartition(".")
        if (not separator or not owner_pattern or not method_name.isidentifier()
                or (method_name.startswith("__") and method_name.endswith("__"))):
            raise ValueError("kind='method' match must end in a literal method name")
        if method_name != "forward":
            self._validate_non_forward_method()
            return
        self._validate_forward_method()

    def _validate_non_forward_method(self) -> None:
        """Validate a decorated method factory without a planner contract."""
        if self.region_dispatch is not None or any(
                getattr(self, name) is not None for name in self._BOUNDARY_FIELDS):
            raise ValueError("kind='method' on a non-forward method cannot declare a boundary contract")
        if not getattr(self.replace.implementation.callable, "_hp_method_replacement", False):
            raise TypeError("kind='method' implementation must use @method_replacement")

    def _validate_forward_method(self) -> None:
        """Validate a forward factory and its boundary dispatch declaration."""
        self.region_dispatch = False if self.region_dispatch is None else self.region_dispatch
        if not isinstance(self.region_dispatch, bool):
            raise TypeError("region_dispatch must be a bool")
        implementation = self.replace.implementation
        meta = getattr(implementation.callable, "_injection_meta", None)
        if meta is None:
            raise TypeError(
                "kind='method' on forward implementation must use @forward_replacement or @local_compute"
            )
        if meta.kind not in ("local_compute", "inner_wrapper"):
            raise TypeError(f"unsupported forward target metadata kind {meta.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the public declaration."""
        result: dict[str, Any] = {"match": self.match}
        if self.when is not None:
            result["when"] = self.when
        if self.low_precision_dtype_scheme is not None:
            result["low_precision_dtype_scheme"] = self.low_precision_dtype_scheme
        if self.replace is not None:
            implementation_config = _serialize_config_value(self.replace.implementation)
            implementation_value = (
                implementation_config["_target_"]
                if isinstance(implementation_config, dict) and set(implementation_config) == {"_target_"}
                else implementation_config
            )
            action = {"kind": self.replace.kind, "implementation": implementation_value}
            if self.replace.kind == "module":
                if self.replace.source_type is not None:
                    action["source_type"] = self.replace.source_type
                if self.replace.exact_type:
                    action["exact_type"] = True
            result["replace"] = action
        for name in ("region_dispatch", "params", "in_src", "in_dst", "out_src", "out_dst", "tp_divide_attrs"):
            value = getattr(self, name)
            default_forward_dispatch = (
                name == "region_dispatch" and value is False
                and self.replace is not None and self.replace.kind == "method"
                and self.match.rpartition(".")[2] == "forward"
            )
            if value is not None and not default_forward_dispatch:
                result[name] = _serialize_config_value(value)
        return result

    def to_override(self) -> "tuple[str, Any]":
        """Desugar to a ``(match, ModuleShardingSpec)`` plan_overrides entry.

        Placement DSL strings become real Placement objects here; the
        ``"auto"``/``"none"`` sentinels pass through AS STRINGS into the spec
        (the planner's merge resolves them; insert rejects them). Imports are
        lazy: trainer.config must not become an import-time dependency of
        components.distributed (the zero-dependency boundary is guarded by
        test_s5_zero_dep_lint).
        """
        # Keep trainer.config outside the distributed component's import-time dependency graph.
        # pylint: disable-next=import-outside-toplevel
        from hyper_parallel.distributed.recipe_spec import (
            ModuleShardingSpec,
        )

        if self.replace is not None and self.replace.kind == "method" and (
                self.match.rpartition(".")[2] != "forward"):
            raise ValueError("non-forward method replacements do not create planner boundaries")

        if not self.match:
            raise ValueError(
                "plan_overrides entry is missing 'match' (an fqn or "
                "fqn glob such as '*.self_attn' or '*.mlp')")
        implementation = self.replace.implementation if self.replace is not None else None
        meta = getattr(implementation.callable, "_injection_meta", None) if implementation is not None else None
        spec = ModuleShardingSpec(
            local_compute_fn=implementation if meta is not None and meta.kind == "local_compute" else None,
            inner_target="self" if meta is not None and meta.kind == "inner_wrapper" else None,
            inner_wrapper=implementation if meta is not None and meta.kind == "inner_wrapper" else None,
            region_dispatch=self.region_dispatch,
            tp_divide_attrs=self._validate_tp_divide_attrs(),
        )
        for attr in self._CONTRACT_FIELDS:
            raw = getattr(self, attr)
            if raw is None:
                continue
            setattr(spec, attr, self._parse_contract_field(attr, raw))
        match = (
            self.match.rpartition(".")[0]
            if self.replace is not None and self.replace.kind == "method"
            else self.match
        )
        return match, spec

    def _validate_tp_divide_attrs(self) -> Optional[List[str]]:
        """Validate the YAML transport shape for TP-local attributes."""
        attrs = self.tp_divide_attrs
        if attrs is None:
            return None
        if not isinstance(attrs, list):
            raise ValueError(
                f"plan_overrides match={self.match!r}: tp_divide_attrs "
                f"must be a list of attribute names, got {type(attrs).__name__}")
        seen = set()
        for attr in attrs:
            if not isinstance(attr, str) or not attr or not attr.isidentifier():
                raise ValueError(
                    f"plan_overrides match={self.match!r}: "
                    f"tp_divide_attrs may only contain valid attribute names, got {attr!r}")
            if attr in seen:
                raise ValueError(
                    f"plan_overrides match={self.match!r}: "
                    f"tp_divide_attrs contains duplicate attribute {attr!r}")
            seen.add(attr)
        return list(attrs)

    def _parse_contract_field(self, attr, raw):
        """YAML form → spec field value (DSL parse / sentinel pass-through)."""
        # Keep trainer.config outside the distributed component's import-time dependency graph.
        # pylint: disable-next=import-outside-toplevel
        from hyper_parallel.distributed.recipe_spec import (
            parse_named_placement,
        )

        if isinstance(raw, str):
            if raw in ("auto", "none"):
                return raw
            raise ValueError(
                f"plan_overrides match={self.match!r}: string value of contract "
                f"field {attr} only accepts the sentinels 'auto' (explicit "
                f"inherit) / 'none' (explicit clear), got {raw!r}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"plan_overrides match={self.match!r}: contract field {attr} "
                f"expects a mapping or a sentinel string, got {raw!r}")
        if not raw:
            return {}  # explicit empty ("as written": clear / no sharding), unlike omitting the field
        if attr in ("out_src", "out_dst") and all(
                isinstance(v, str) for v in raw.values()):
            # Scalar shorthand {axis: placement} — already a NamedPlacement
            # (the output name is filled in uniformly by the planner's
            # _normalize_out_fields)
            return parse_named_placement(
                raw, path=f"plan_overrides[{self.match!r}].{attr}")
        return {
            name: parse_named_placement(
                named, path=f"plan_overrides[{self.match!r}].{attr}.{name}")
            for name, named in raw.items()
        }


_WHEN_CONDITIONS = ("cp", "ep", "sequence_parallel", "low_precision")


def _import_module_type(path: str) -> type[nn.Module]:
    """Import the source module type named by a replacement YAML entry."""

    if not isinstance(path, str) or not path.strip():
        raise ValueError("plan_overrides.replace.source_type must be a non-empty symbol path")
    parts = path.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            symbol = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise ValueError(
                f"plan_overrides.replace.source_type {path!r} failed while importing {exc.name!r}"
            ) from exc
        except ImportError as exc:
            raise ValueError(
                f"plan_overrides.replace.source_type {path!r} failed while importing: {exc}"
            ) from exc
        for attribute in parts[split_at:]:
            if not hasattr(symbol, attribute):
                raise ValueError(
                    f"plan_overrides.replace.source_type {path!r} has no attribute {attribute!r}"
                )
            symbol = getattr(symbol, attribute)
        if not isinstance(symbol, type) or not issubclass(symbol, nn.Module):
            raise TypeError(f"plan_overrides.replace.source_type {path!r} must name an nn.Module type")
        return symbol
    raise ValueError(f"plan_overrides.replace.source_type {path!r} could not be imported")


def _target_replacement_factory(
    target: Target[Any],
    low_precision_policy: Optional[LowPrecisionDtypeScheme] = None,
) -> ModuleReplacementFactory:
    """Bind a YAML Target's static args to the replacement factory protocol."""

    if not isinstance(target, ReplacementTarget) and get_module_family(target.callable) is not None:
        # Preserve configured Python objects; only Family baselines need selection.
        # Target has no public accessors for its raw constructor values.
        target = ReplacementTarget(
            target.callable, target_path=target._target_path, **target._kwargs,
        )

    if not getattr(target.callable, "_hp_module_replacement", False):
        raise TypeError(
            "plan_overrides replace.implementation for kind='module' must use @module_replacement"
        )

    @module_replacement
    def factory(
        *,
        module: nn.Module,
        module_fqn: str,
        context: Mapping[str, Any],
    ) -> nn.Module:
        """Build one replacement module by delegating to the bound YAML Target."""
        if low_precision_policy is not None:
            factory_context = dict(context)
            factory_context["low_precision"] = low_precision_policy
            context = MappingProxyType(factory_context)
        return target.build(module=module, module_fqn=module_fqn, context=context)

    return factory


def _target_method_factory(target: Target[Any]):
    """Bind YAML static arguments to the method replacement factory protocol."""
    if not getattr(target.callable, "_hp_method_replacement", False):
        raise TypeError("plan_overrides replace.implementation for kind='method' must use @method_replacement")

    @method_replacement
    def factory(
        *, module: nn.Module, method_fqn: str, original_method: Any, context: Mapping[str, Any],
    ) -> Any:
        """Build one replacement callable from the configured target."""
        return target.build(
            module=module, method_fqn=method_fqn, original_method=original_method, context=context,
        )

    return factory


def entries_to_method_replacements(
    entries: List[PlanOverride], *, cp_size: int = 1, ep_size: int = 1,
    sequence_parallel: bool = False, low_precision_enabled: bool = False,
) -> tuple[MethodReplacementSpec, ...]:
    """Resolve active non-forward method actions into executable rules.

    Args:
        entries: Trainer plan override declarations.
        cp_size: Active context parallel degree.
        ep_size: Active expert parallel degree.
        sequence_parallel: Whether sequence parallelism is active.
        low_precision_enabled: Whether online low precision is enabled.

    Returns:
        Rules applied after module replacement and before sharding.
    """
    rules = []
    for entry in entries:
        if entry.replace is None or entry.replace.kind != "method":
            continue
        if entry.match.rpartition(".")[2] == "forward":
            continue
        _validate_when(entry)
        active = {
            None: True,
            "cp": cp_size > 1,
            "ep": ep_size > 1,
            "sequence_parallel": sequence_parallel,
            "low_precision": low_precision_enabled,
        }[entry.when]
        if active:
            rules.append(MethodReplacementSpec(
                match=entry.match,
                factory=_target_method_factory(entry.replace.implementation),
            ))
        else:
            logger.info("plan_overrides: match=%r skipped because condition %r is inactive",
                        entry.match, entry.when)
    return tuple(rules)


def entries_to_module_replacements(
    entries: List[PlanOverride],
    *,
    low_precision_enabled: bool = False,
    low_precision_config: Optional[LowPrecisionConfig] = None,
) -> tuple[ModuleReplacementSpec, ...]:
    """Desugar replacement actions and bind their named dtype schemes.

    Args:
        entries: Raw planner override entries.
        low_precision_enabled: Whether low-precision replacement is active.
        low_precision_config: Named dtype schemes for active replacements.
    """

    effective_config = low_precision_config if low_precision_enabled else None
    rules = []
    for entry in entries:
        if entry.replace is None or entry.replace.kind != "module":
            continue
        if entry.when == "low_precision" and not low_precision_enabled:
            continue
        _validate_replacement_entry(entry, effective_config)
        patterns = (entry.match,) if isinstance(entry.match, str) else tuple(entry.match)
        policy = None
        if entry.when == "low_precision" and effective_config is not None:
            policy = effective_config.resolve_dtype_scheme(entry.low_precision_dtype_scheme)
        rules.append(
            ModuleReplacementSpec(
                match=patterns,
                factory=_target_replacement_factory(entry.replace.implementation, policy),
                module_type=(
                    _import_module_type(entry.replace.source_type)
                    if entry.replace.source_type is not None else None
                ),
                exact_type=entry.replace.exact_type,
            )
        )
    return tuple(rules)


def _validate_replacement_entry(entry: PlanOverride, effective_config: Optional[LowPrecisionConfig]) -> None:
    """Validate one replacement before constructing its planner rule."""
    if entry.when not in (None, "low_precision"):
        raise ValueError(
            f"plan_overrides match={entry.match!r} uses kind='module' and "
            "does not support 'when' values other than 'low_precision' "
            "(module replacements are not conditioned on parallel topology)"
        )
    if entry.low_precision_dtype_scheme is not None and (
        not isinstance(entry.low_precision_dtype_scheme, str)
        or not entry.low_precision_dtype_scheme.strip()
    ):
        raise ValueError(
            f"plan_overrides match={entry.match!r}: "
            "low_precision_dtype_scheme must be a non-empty string"
        )
    if entry.low_precision_dtype_scheme is not None and entry.when != "low_precision":
        raise ValueError(
            f"plan_overrides match={entry.match!r} sets "
            "low_precision_dtype_scheme but is not a low-precision entry"
        )
    if entry.when == "low_precision" and effective_config is None:
        raise ValueError(
            f"plan_overrides match={entry.match!r} enables a low-precision "
            "replacement but training.low_precision is missing"
        )
    if not isinstance(entry.replace.implementation, Target):
        raise TypeError(
            f"plan_overrides match={entry.match!r} replace.implementation must be a YAML Target"
        )


def _has_sharding_action(entry: PlanOverride) -> bool:
    action_names = (
        "region_dispatch", "params", "in_src", "in_dst", "out_src", "out_dst",
    )
    return any(getattr(entry, name) is not None for name in action_names)


def _validate_when(entry: PlanOverride) -> None:
    """Validate programmatic when values that bypass the YAML Literal check."""

    if entry.when is not None and entry.when not in _WHEN_CONDITIONS:
        raise ValueError(
            f"plan_overrides match={entry.match!r} has invalid when={entry.when!r}; "
            f"expected one of {list(_WHEN_CONDITIONS)}"
        )


def _merge_plan_override(overrides: dict[str, Any], match: str, spec: Any) -> None:
    """Merge one desugared spec into an existing match, with later fields winning."""
    if match not in overrides:
        overrides[match] = spec
        return
    previous = overrides[match]
    for name in ("local_compute_fn", "inner_target", "inner_wrapper",
                 "inner_out_src", "params", "in_src", "in_dst",
                 "out_src", "out_dst", "tp_divide_attrs"):
        value = getattr(spec, name)
        if value is not None:
            setattr(previous, name, value)
    if spec.region_dispatch is not None:
        previous.region_dispatch = spec.region_dispatch


def entries_to_plan_overrides(
        entries: "List[PlanOverride]", *, cp_size: int = 1, ep_size: int = 1,
        sequence_parallel: bool = False,
        low_precision_enabled: bool = False,
) -> "dict[str, Any]":
    """Desugar PlanOverride entries into a ``plan_overrides`` dict.

    Args:
        entries: Raw YAML planner override entries.
        cp_size: Context-parallel degree.
        ep_size: Expert-parallel degree.
        sequence_parallel: Whether decoder sequence parallelism is active.
        low_precision_enabled: Whether low-precision rules are active.

    - ``when`` filter: an entry whose declared condition is inactive
      (``when="cp"`` with cp_size==1 etc.) is SKIPPED with an INFO log —
      declared gating, never silent application. An unknown ``when`` value
      fails fast listing the valid conditions (programmatic construction
      bypasses the resolver's Literal check);
    - two entries sharing the same ``match`` are merged field-wise
      (non-None fields, later entry wins) — the planner's interface is
      exactly this dict.
    """
    overrides: dict[str, Any] = {}
    for entry in entries:
        if entry.replace is not None and entry.replace.kind == "method" and (
                entry.match.rpartition(".")[2] != "forward"):
            _validate_when(entry)
            continue
        if isinstance(entry.match, list) and (
            entry.replace is None or entry.replace.kind != "module" or _has_sharding_action(entry)
        ):
            raise ValueError(
                "plan_overrides match lists are supported only for replace.kind='module' "
                "entries; use separate string matches for sharding actions"
            )
        if entry.replace is not None and entry.replace.kind == "module" and entry.when not in (
            None,
            "low_precision",
        ):
            raise ValueError(
                f"plan_overrides match={entry.match!r} uses replace.kind='module' and "
                "does not support 'when' values other than 'low_precision' "
                "(module replacements are not conditioned on parallel topology)"
            )
        _validate_when(entry)
        if entry.when is not None:
            active = {
                "cp": cp_size > 1,
                "ep": ep_size > 1,
                "sequence_parallel": sequence_parallel,
                "low_precision": low_precision_enabled,
            }[entry.when]
            if not active:
                logger.info(
                    "plan_overrides: match=%r skipped because condition %r "
                    "is inactive",
                    entry.match, entry.when)
                continue
        if entry.replace is not None and entry.replace.kind == "module":
            continue
        match, spec = entry.to_override()
        _merge_plan_override(overrides, match, spec)
    return overrides


def normalize_distributed_setup_overrides(
    distributed_setup: Any,
    config: "TrainerConfig",
) -> None:
    """Fill the normalized plan fields of a ``DistributedSetup`` (05 §15.2.6).

    Args:
        distributed_setup: Mutable setup receiving normalized planner rules.
        config: Trainer configuration containing raw override entries.

    This is the single Trainer-side desugar point: raw YAML ``PlanOverride``
    entries (``TrainerConfig.plan_overrides``) are converted HERE —
    when-filtered against the built mesh and compiled into replacement
    rules — so the AutoModels build pipeline consumes only normalized
    objects and never imports trainer config.
    """
    entries = list(getattr(config, "plan_overrides", None) or [])
    mesh_context = getattr(distributed_setup, "mesh_context", None)
    low_precision_enabled = bool(
        getattr(
            getattr(distributed_setup, "low_precision_config", None),
            "enabled",
            False,
        )
    )
    if low_precision_enabled and not any(
        entry.replace is not None and entry.replace.kind == "module" and entry.when == "low_precision"
        for entry in entries
    ):
        raise ValueError(
            "low_precision is enabled but no low-precision module replacement "
            "is configured"
        )
    distributed_setup.module_replacements = entries_to_module_replacements(
        entries,
        low_precision_enabled=low_precision_enabled,
        low_precision_config=getattr(distributed_setup, "low_precision_config", None),
    )
    distributed_setup.method_replacements = entries_to_method_replacements(
        entries,
        cp_size=getattr(mesh_context, "cp_size", 1),
        ep_size=getattr(mesh_context, "ep_size", 1),
        sequence_parallel=getattr(mesh_context, "sequence_parallel", False),
        low_precision_enabled=low_precision_enabled,
    )
    distributed_setup.plan_overrides = (
        entries_to_plan_overrides(
            entries,
            cp_size=getattr(mesh_context, "cp_size", 1),
            ep_size=getattr(mesh_context, "ep_size", 1),
            sequence_parallel=getattr(mesh_context, "sequence_parallel", False),
            low_precision_enabled=low_precision_enabled,
        )
        if entries
        else None
    )
