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
``FSDP2Config`` / ``FSDP2MixedPrecisionConfig`` / ``CompileConfig`` are
defined in ``auto_models/build_options.py`` and only re-exported here;
``PlanOverride`` stays the YAML DTO that desugars to
``auto_models.distributed.recipe_spec.ModuleShardingSpec``.
"""

import importlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Union

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.models.build_options import (
    CompileConfig,
    FSDP2Config,
    FSDP2MixedPrecisionConfig,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementFactory,
    ModuleReplacementSpec,
    module_replacement,
)

from hyper_parallel.trainer.config.target import Target

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


@dataclass
class ActivationCheckpointConfig:
    """Activation-checkpoint options exposed by the initial YAML schema.

    ``swap_inputs`` is consumed only when ``mode`` is ``"full"`` or
    ``"selective"``.
    """

    mode: Optional[Literal["off", "full", "selective"]] = "off"
    swap_inputs: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous values for activation input swapping."""
        if not isinstance(self.swap_inputs, bool):
            raise TypeError("activation_checkpoint.swap_inputs must be a bool")


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
        module_type: importable source ``nn.Module`` type required by
            ``replace_module``. The symbol is resolved but never constructed.
        exact_type: replacement type check mode. Defaults to subclass matching.
        replace_module: Target factory decorated with ``@module_replacement``.
            It receives ``module``, ``module_fqn``, and a read-only context,
            and must return a structure-preserving replacement. A list
            ``match`` is supported only for replacement-only entries.
        match: fqn or fqn glob matched (fnmatchcase) against the plan's
            boundary FQNs — ``*`` spans dots, so ``"*.self_attn"`` hits
            ``model.layers.0.self_attn``.
        when: optional activation condition. Sharding actions accept ``"cp"``
            (active when cp_size>1), ``"ep"`` (ep_size>1), or
            ``"low_precision"`` (active when online low precision is
            enabled). A replacement action accepts only ``"low_precision"``;
            module replacement must not depend on the parallel topology.
        local_compute_fn: a Target whose callable is a **factory** — must be
            decorated ``@local_compute`` (injection discipline); the mesh
            family ``mesh``/``tp_mesh``/``cp_mesh``/``ep_mesh`` is mandatory
            context, all filled by the framework at apply time (``module``/
            ``spec`` optional); the factory must return the region compute
            fn ``fn(module, *local_args)`` (params validated against the
            module's forward at apply time).
            This is also the **performance-replacement** channel: point it at
            any factory returning a faster compute fn (see
            examples/distributed/perf_replacement.py). Shipped EP archetype
            factories (complete semantics, explicitly chosen — see the
            archetype table in recipes.py):
            ``hyper_parallel.distributed.expert_parallel.recipes.qwen2moe_ep_compute_fn``
            / ``mixtral_ep_compute_fn`` /
            ``routed_only_ep_compute_fn`` / ``deepseekv3_ep_compute_fn``; the
            Qwen3-MoE archetype lives at its model-adapter home
            ``hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn``
            (M3; the legacy ``recipes.qwen3moe_ep_compute_fn`` delegate was
            removed in M5). A
            MoE whose behavior matches no archetype writes its own factory,
            referencing ``MOE_ROUTER_ADAPTERS`` entries by name (reference:
            examples/distributed/ep_factories.py).
        inner_target: attribute name of the inner submodule whose forward is
            wrapped (``"self"`` = the boundary module itself; default:
            auto-location).
        inner_wrapper: a ``INNER_WRAPPER_REGISTRY`` name (``"sdpa_qkv"`` /
            ``"sdpa_hf"`` / ``"flex_qkv"`` / ``"flex_hf"``) or a Target
            pointing at an ``@inner_wrapper``-decorated wrapper fn
            ``fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh)``
            — shipped defaults live in
            ``hyper_parallel.distributed.context_parallel.wrappers``. The
            mechanism is not CP-gated (declaration == application), but the
            four shipped names are CP schemes and require an active cp axis
            (fail-fast otherwise); custom wrappers receive ``cp_mesh=None``
            when no cp axis exists.
        inner_out_src: required when inner_target points at a submodule
            (not ``self``) — the inner output placement declaration:
            the sentinel ``"first_input"`` (output layout == first DTensor
            input's layout, for attention-style layout-preserving
            wrappers), ``{axis: placement_str}`` (single output), or
            ``{name: {axis: placement_str}}`` (multi-output). The framework
            never derives inner output layouts — undeclared fails fast at
            apply time.
        region_dispatch: validate execution-mode declaration (no default —
            required when declaring an injection). ``false`` = the region
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
            boundary): usually omitted — empty inherits the derived contract;
            insert mode (misses every boundary): all must be fully declared.
        tp_divide_attrs: optional module-instance integer attributes divided
            exactly by the active TP size when the module forward runs on
            local tensors. Omit for no user adjustment; an explicit empty
            list clears an inherited glob declaration.
    """

    match: Union[str, List[str]]
    when: Optional[Literal["cp", "ep", "low_precision"]] = None
    module_type: Optional[str] = None
    exact_type: bool = False
    replace_module: Optional[Target[Any]] = None
    local_compute_fn: Optional[Target[Any]] = None
    inner_target: Optional[str] = None
    inner_wrapper: Optional[Union[str, Target[Any]]] = None
    inner_out_src: Optional[Any] = None
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

        if not self.match:
            raise ValueError(
                "plan_overrides entry is missing 'match' (an fqn or "
                "fqn glob such as '*.self_attn' or '*.mlp')")
        spec = ModuleShardingSpec(
            local_compute_fn=self.local_compute_fn,
            inner_target=self.inner_target,
            inner_wrapper=self.inner_wrapper,
            inner_out_src=self._parse_inner_out_src(),
            region_dispatch=self.region_dispatch,
            tp_divide_attrs=self._validate_tp_divide_attrs(),
        )
        for attr in self._CONTRACT_FIELDS:
            raw = getattr(self, attr)
            if raw is None:
                continue
            setattr(spec, attr, self._parse_contract_field(attr, raw))
        return self.match, spec

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

    def _parse_inner_out_src(self):
        """Desugar the YAML form of inner_out_src: sentinel / single-output DSL / multi-output DSL."""
        # Keep trainer.config outside the distributed component's import-time dependency graph.
        # pylint: disable-next=import-outside-toplevel
        from hyper_parallel.distributed.recipe_spec import (
            parse_named_placement,
        )

        raw = self.inner_out_src
        if raw is None:
            return None
        path = f"plan_overrides[{self.match!r}].inner_out_src"
        if isinstance(raw, str):
            if raw == "first_input":
                return raw
            raise ValueError(
                f"{path}: string value only accepts the sentinel 'first_input' "
                f"(output layout == layout of the first DTensor input), got {raw!r}")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(
                f"{path}: expected 'first_input', {{axis: placement}} or "
                f"{{name: {{axis: placement}}}}, got {raw!r}")
        if all(isinstance(v, str) for v in raw.values()):
            return parse_named_placement(raw, path=path)  # single output
        return {name: parse_named_placement(named, path=f"{path}.{name}")
                for name, named in raw.items()}  # multi-output

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


_WHEN_CONDITIONS = ("cp", "ep", "low_precision")


def _import_module_type(path: str) -> type:
    """Import the source module type named by a replacement YAML entry."""

    if not isinstance(path, str) or not path:
        raise ValueError("plan_overrides.module_type must be a non-empty symbol path")
    parts = path.split(".")
    for split_at in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_at])
        try:
            symbol = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                continue
            raise ValueError(
                f"plan_overrides.module_type {path!r} failed while importing {exc.name!r}"
            ) from exc
        except ImportError as exc:
            raise ValueError(
                f"plan_overrides.module_type {path!r} failed while importing: {exc}"
            ) from exc
        for attribute in parts[split_at:]:
            if not hasattr(symbol, attribute):
                raise ValueError(
                    f"plan_overrides.module_type {path!r} has no attribute {attribute!r}"
                )
            symbol = getattr(symbol, attribute)
        if not isinstance(symbol, type):
            raise TypeError(f"plan_overrides.module_type {path!r} must name a type")
        return symbol
    raise ValueError(f"plan_overrides.module_type {path!r} could not be imported")


def _target_replacement_factory(target: Target[Any]) -> ModuleReplacementFactory:
    """Bind a YAML Target's static args to the replacement factory protocol."""

    if not getattr(target._target_, "_hp_module_replacement", False):  # pylint: disable=protected-access
        raise TypeError(
            "plan_overrides replace_module target must be decorated with "
            "@module_replacement"
        )

    @module_replacement
    def factory(
        *,
        module: nn.Module,
        module_fqn: str,
        context: Mapping[str, Any],
    ) -> nn.Module:
        """Build one replacement module by delegating to the bound YAML Target."""
        return target.build(module=module, module_fqn=module_fqn, context=context)

    return factory


def entries_to_module_replacements(
    entries: List[PlanOverride],
    *,
    low_precision_enabled: bool = False,
) -> tuple[ModuleReplacementSpec, ...]:
    """Desugar active YAML replacement actions without involving sharding."""

    rules = []
    for entry in entries:
        if entry.replace_module is None:
            continue
        if entry.when not in (None, "low_precision"):
            raise ValueError(
                f"plan_overrides match={entry.match!r} uses replace_module and "
                "does not support 'when' values other than 'low_precision' "
                "(module replacements are not conditioned on parallel topology)"
            )
        if entry.when == "low_precision" and not low_precision_enabled:
            continue
        if entry.module_type is None:
            raise ValueError(
                f"plan_overrides match={entry.match!r} uses replace_module but omits module_type"
            )
        if not isinstance(entry.replace_module, Target):
            raise TypeError(
                f"plan_overrides match={entry.match!r} replace_module must be a YAML Target"
            )
        patterns = (entry.match,) if isinstance(entry.match, str) else tuple(entry.match)
        rules.append(
            ModuleReplacementSpec(
                match=patterns,
                factory=_target_replacement_factory(entry.replace_module),
                module_type=_import_module_type(entry.module_type),
                exact_type=entry.exact_type,
            )
        )
    return tuple(rules)


def _has_sharding_action(entry: PlanOverride) -> bool:
    return any(
        getattr(entry, name) is not None
        for name in (
            "local_compute_fn", "inner_target", "inner_wrapper", "inner_out_src",
            "region_dispatch", "params", "in_src", "in_dst", "out_src", "out_dst",
        )
    )


def _validate_when(entry: PlanOverride) -> None:
    """Validate programmatic when values that bypass the YAML Literal check."""

    if entry.when is not None and entry.when not in _WHEN_CONDITIONS:
        raise ValueError(
            f"plan_overrides match={entry.match!r} has invalid when={entry.when!r}; "
            f"expected one of {list(_WHEN_CONDITIONS)}"
        )


def entries_to_plan_overrides(
        entries: "List[PlanOverride]", *, cp_size: int = 1, ep_size: int = 1,
        low_precision_enabled: bool = False,
) -> "dict[str, Any]":
    """Desugar PlanOverride entries into a ``plan_overrides`` dict.

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
        if isinstance(entry.match, list) and (
            entry.replace_module is None or _has_sharding_action(entry)
        ):
            raise ValueError(
                "plan_overrides match lists are supported only for replace_module "
                "entries; use separate string matches for sharding actions"
            )
        if entry.replace_module is not None and entry.when not in (
            None,
            "low_precision",
        ):
            raise ValueError(
                f"plan_overrides match={entry.match!r} uses replace_module and "
                "does not support 'when' values other than 'low_precision' "
                "(module replacements are not conditioned on parallel topology)"
            )
        _validate_when(entry)
        if entry.when is not None:
            active = {
                "cp": cp_size > 1,
                "ep": ep_size > 1,
                "low_precision": low_precision_enabled,
            }[entry.when]
            if not active:
                logger.info(
                    "plan_overrides: match=%r skipped because condition %r "
                    "is inactive",
                    entry.match, entry.when)
                continue
        if entry.replace_module is not None and not _has_sharding_action(entry):
            continue
        match, spec = entry.to_override()
        if match in overrides:
            prev = overrides[match]
            for name in ("local_compute_fn", "inner_target", "inner_wrapper",
                         "inner_out_src", "params", "in_src", "in_dst",
                         "out_src", "out_dst", "tp_divide_attrs"):
                value = getattr(spec, name)
                if value is not None:
                    setattr(prev, name, value)
            if spec.region_dispatch is not None:
                prev.region_dispatch = spec.region_dispatch
        else:
            overrides[match] = spec
    return overrides


def normalize_distributed_setup_overrides(
    distributed_setup: Any,
    config: "TrainerConfig",
) -> None:
    """Fill the normalized plan fields of a ``DistributedSetup`` (05 §15.2.6).

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
        entry.replace_module is not None and entry.when == "low_precision"
        for entry in entries
    ):
        raise ValueError(
            "low_precision is enabled but no low-precision module replacement "
            "is configured"
        )
    distributed_setup.module_replacements = entries_to_module_replacements(
        entries,
        low_precision_enabled=low_precision_enabled,
    )
    distributed_setup.plan_overrides = (
        entries_to_plan_overrides(
            entries,
            cp_size=getattr(mesh_context, "cp_size", 1),
            ep_size=getattr(mesh_context, "ep_size", 1),
            low_precision_enabled=low_precision_enabled,
        )
        if entries
        else None
    )
