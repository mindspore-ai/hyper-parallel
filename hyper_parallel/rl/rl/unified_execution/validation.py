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
"""Audit realized model computation and HP boundaries on both runtimes."""

from __future__ import annotations

from fnmatch import fnmatchcase
from functools import lru_cache, partial
import inspect
from types import CodeType
from typing import Any

from torch import nn

from rl.unified_execution.models import get_model_family
from rl.unified_execution.contract import fingerprint, parse_unified_payload
from rl.unified_execution.execution_plan import fixed_capacity_ep_capacity, pik_row_matches, pik_row_options
from rl.workers.adapters.log_probs import bind_token_log_probs
from hyper_parallel.components.modules.gqa_attention import GQAAttention
from hyper_parallel.core.activation_memory.wrapper import CheckpointExcludeWrapper, CheckpointWrapper, SwapWrapper
from hyper_parallel.distributed._builder import forward_rewriter


def _module_type(module: Any) -> str:
    # FSDP synthesizes a subclass but leaves the computation class in its MRO.
    for cls in type(module).__mro__:
        if cls.__module__.startswith(("hyper_parallel.core.fully_shard", "torch.distributed.fsdp")):
            continue
        return f"{cls.__module__}.{cls.__qualname__}"
    raise ValueError("Cannot identify computation module class")


def model_structure(model: Any) -> dict[str, Any]:
    """Describe actual computation modules and logical parameters before cache adaptation."""
    modules = {name: _module_type(module) for name, module in model.named_modules()}
    parameters, first_names, aliases = {}, {}, {}
    logical_shapes = getattr(model, "_rl_logical_parameter_shapes", {})
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameters[name] = list(logical_shapes.get(name, parameter.shape))
        identity = id(parameter)
        if identity in first_names:
            aliases[name] = first_names[identity]
        else:
            first_names[identity] = name
    return {"modules": modules, "parameters": parameters, "aliases": aliases}


@lru_cache(maxsize=1)
def _forward_wrapper_targets() -> dict[CodeType, str]:
    """Identify HP wrapper implementations and the closure callable they execute."""
    targets = {}
    # HP has no public wrapper-introspection API. Code identities preserve its
    # TP/SP boundaries while preventing __wrapped__ from hiding custom compute.
    factories = {
        "_wrap_production_forward": ("production_forward", "original_forward"),
        "_wrap_validate_forward": ("validate_forward", "original_forward"),
        "_wrap_local_region_forward": ("local_region_forward", "compute_fn"),
        "_wrap_module_forward": ("local_param_forward", "original_forward"),
    }
    for factory_name, (wrapper_name, target_name) in factories.items():
        factory = getattr(forward_rewriter, factory_name)
        for constant in factory.__code__.co_consts:
            if isinstance(constant, CodeType) and constant.co_name == wrapper_name:
                targets[constant] = target_name
    for constant in bind_token_log_probs.__code__.co_consts:
        if isinstance(constant, CodeType) and constant.co_name == "token_log_probs_forward":
            targets[constant] = "original_forward"
    return targets


def _unwrap_activation_module(module: nn.Module) -> nn.Module:
    """Retain known HP activation wrappers while selecting their real computation owner."""
    wrappers = (CheckpointWrapper, CheckpointExcludeWrapper, SwapWrapper)
    while isinstance(module, wrappers):
        function = getattr(module.forward, "__func__", module.forward)
        if function not in {wrapper.forward for wrapper in wrappers}:
            raise ValueError("Unified training does not support a custom activation-wrapper forward")
        module = module._wrapped_module  # pylint: disable=protected-access
    return module


def _require_forward_path(module: nn.Module, expected: tuple, path: str) -> None:
    """Reject any executable forward that can bypass the registered numerical core."""
    if _forward_leaf(module, path) not in expected:
        raise ValueError(f"Unified training cannot verify the attention execution path at {path}")


def _forward_leaf(module: nn.Module, path: str) -> Any:
    """Follow only known HP wrappers to the function they actually execute."""
    function = module.forward
    visited = set()
    while True:
        function = getattr(function, "__func__", function)
        if id(function) in visited:
            raise ValueError(f"Unified training found a cyclic forward wrapper at {path}")
        visited.add(id(function))
        target = _forward_wrapper_targets().get(getattr(function, "__code__", None))
        if target is None:
            return function
        function = inspect.getclosurevars(function).nonlocals[target]


def validate_attention_execution_path(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
    """Audit the shared Qwen3 forward chain without loading a provider or installing adapters.

    Args:
        model: Built Qwen3 causal language model or decoder, including supported HP wrappers.

    Returns:
        The decoder and attention owners beneath transparent activation wrappers.
        Existing production TP/SP boundary functions remain installed unchanged.

    Raises:
        ValueError: A custom forward can bypass the shared attention interface.
    """
    model = _unwrap_activation_module(model)
    family = get_model_family(getattr(model.config, "model_type", None))
    owners = family.execution_types()
    if hasattr(model, "layers"):
        decoder = model
    else:
        _require_forward_path(model, (owners["causal_lm"].forward,), "model.forward")
        decoder = _unwrap_activation_module(getattr(model, "model", None))
    if decoder is None or getattr(decoder.config, "model_type", None) != family.model_type:
        raise ValueError("Unified training attention requires a Qwen3 decoder model")
    _require_forward_path(decoder, (owners["decoder"].forward,), "model.model.forward")
    attentions = []
    for index, wrapped_layer in enumerate(decoder.layers):
        layer = _unwrap_activation_module(wrapped_layer)
        _require_forward_path(layer, (owners["layer"].forward,), f"layers.{index}.forward")
        attention = _unwrap_activation_module(getattr(layer, family.attention_attribute))
        _require_forward_path(attention, (owners["attention"].forward, GQAAttention.forward),
                              f"layers.{index}.self_attn.forward")
        attentions.append(attention)
    return decoder, attentions


def capture_training_structure(model: Any, payload: dict, *, inference: bool = False) -> dict[str, Any]:
    """Check actual replacement coverage and fingerprint shared computation.

    The same audit runs on inference before adding its attention cache bridge.
    The audit also records the required input sequence multiple for RL roles;
    it never replaces parameters or invalidates optimizer references.
    """
    payload = parse_unified_payload(payload)
    family = get_model_family(getattr(model.config, "model_type", None))
    if family.model_type != payload.get("model_family", "qwen3"):
        raise ValueError("Unified model family differs from the Actor contract")
    if getattr(model.config, "attention_dropout", 0.0) != 0.0:
        raise ValueError("Unified training requires attention_dropout=0")
    if family.moe:
        ep_size = payload.get("expert_parallelism", {}).get("inference" if inference else "training", 1)
        _validate_moe_components(model, family.execution_types(), family.router_adapter, ep_size,
                                 fixed_capacity_ep_capacity(payload["plan_overrides"]))
    _validate_pik_components(model, payload, inference=inference)
    structure = model_structure(model)
    replaced = set(getattr(model, "_hp_replaced_module_fqns", ()))
    parallelism = payload["parallelism" if inference else "training_parallelism"]
    for entry in payload["compute_plan_overrides"]:
        condition = entry.get("when")
        enabled = {"cp": parallelism["cp"] > 1, "ep": False, "low_precision": False,
                   "sequence_parallel": parallelism["sequence_parallel"]}
        if condition is not None and not enabled[condition]:
            continue
        patterns = entry["match"] if isinstance(entry["match"], list) else [entry["match"]]
        for pattern in patterns:
            matches = {name for name in structure["modules"] if fnmatchcase(name, pattern)}
            if not matches or not matches <= replaced:
                raise ValueError(f"Unified model did not apply module replacement {pattern!r}")
    return {"structure": structure, "fingerprint": fingerprint(structure)}


def _validate_moe_components(model: nn.Module, owners: dict[str, type], router_adapter: str, ep_size: int = 1,
                             graph_capacity: int | None = None) -> None:
    """Require the shared HP expert implementation and authoritative HF routing."""
    # Only a MoE model needs the optional native grouped-expert implementation.
    from hyper_parallel.components.modules.grouped_experts import GroupedExperts  # pylint: disable=C0415

    for index, wrapped_layer in enumerate(model.model.layers):
        layer = _unwrap_activation_module(wrapped_layer)
        mlp = _unwrap_activation_module(layer.mlp)
        if not hasattr(mlp, "experts"):
            continue
        if ep_size > 1:
            _validate_ep_compute(mlp, ep_size, f"layers.{index}.mlp", router_adapter, graph_capacity)
        else:
            _require_forward_path(mlp, (owners["moe_block"].forward,), f"layers.{index}.mlp.forward")
            _require_forward_path(mlp.experts, (GroupedExperts.forward,), f"layers.{index}.mlp.experts.forward")
        _require_forward_path(mlp.gate, (owners["router"].forward,), f"layers.{index}.mlp.gate.forward")


def _validate_ep_compute(module: nn.Module, ep_size: int, path: str, router_adapter: str,
                         graph_capacity: int | None = None) -> None:
    """Audit HP's installed EP factory and real router, retaining its communication."""
    # EP dependencies are optional for Dense model execution.
    from hyper_parallel.distributed.expert_parallel import experts, recipes, routing  # pylint: disable=C0415

    expert_forward = experts._local_swiglu_expert_forward  # pylint: disable=protected-access
    function = _forward_leaf(module, path)
    if graph_capacity is not None:
        _validate_graph_ep_compute(module, function, ep_size, graph_capacity, path,
                                  routing.MOE_ROUTER_ADAPTERS[router_adapter])
        _require_forward_path(module.experts, (expert_forward,), path + ".experts")
        return
    if not isinstance(function, partial) or function.args != (module,) or function.keywords:
        raise ValueError(f"Unified EP requires the HP-bound local compute function at {path}")
    function = function.func
    expected_codes = {item for item in recipes.build_ep_compute.__code__.co_consts
                      if isinstance(item, CodeType) and item.co_name == "compute_fn"
                      and "overlap_fn" not in item.co_freevars}
    if getattr(function, "__code__", None) not in expected_codes:
        raise ValueError(f"Unified EP requires HP build_ep_compute at {path}")
    closure = inspect.getclosurevars(function).nonlocals
    if (closure.get("router_fn") is not routing.MOE_ROUTER_ADAPTERS[router_adapter]
            or closure.get("combine") is not experts.SHARED_MERGE_MODES["none"]
            or closure["ep_group"].size() != ep_size):
        raise ValueError(f"Unified EP router, combine or communication group differs at {path}")
    _require_forward_path(module.experts, (expert_forward,), path + ".experts")
    if not getattr(module.experts, "_ep_use_grouped_gemm", False):
        raise ValueError("Unified EP requires the HP grouped GEMM expert recipe")


def _validate_graph_ep_compute(module: nn.Module, function: Any, ep_size: int, capacity: int, path: str,
                               router_fn: Any) -> None:
    """Audit the opt-in transport without weakening the original HP recipe check."""
    # EP computation is optional for Dense models and configuration-only processes.
    from rl.unified_execution.components.moe import fixed_capacity_ep_compute_fn  # pylint: disable=C0415

    if not isinstance(function, partial) or function.args != (module,) or function.keywords:
        raise ValueError(f"Unified graph EP requires the shared bound computation at {path}")
    expected_codes = {item for item in fixed_capacity_ep_compute_fn.__code__.co_consts
                      if isinstance(item, CodeType) and item.co_name == "compute_fn"}
    if getattr(function.func, "__code__", None) not in expected_codes:
        raise ValueError(f"Unified graph EP requires the shared recipe at {path}")
    options = inspect.getclosurevars(function.func).nonlocals["options"]
    if set(options) != {"group", "rank", "world_size", "local_experts", "capacity", "router_fn"}:
        raise ValueError(f"Unified graph EP received unknown execution options at {path}")
    if (options["router_fn"] is not router_fn or options["capacity"] != capacity or options["world_size"] != ep_size
            or options["group"].size() != ep_size or options["group"].rank() != options["rank"]
            or options["local_experts"] != module.experts.local_expert_count
            or not getattr(module.experts, "_ep_use_grouped_gemm", False)):
        raise ValueError(f"Unified graph EP computation differs from its shared plan at {path}")


def _validate_pik_components(model: nn.Module, payload: dict, *, inference: bool) -> None:
    """Check every row replacement after HP build, before model execution."""
    family = payload.get("model_family", "qwen3")
    options = pik_row_options(payload["plan_overrides"], family)
    if options is None:
        return
    # The NPU computation adapter is optional for non-PIK configurations.
    from rl.unified_execution.components.row_parallel import PikLinear, pik_row_compute_fn  # pylint: disable=C0415

    codes = {code for code in pik_row_compute_fn.__code__.co_consts
             if isinstance(code, CodeType) and code.co_name == "compute_fn"}
    topology = payload["parallelism" if inference else "training_parallelism"]
    model.unified_sequence_multiple = topology["tp"] if topology["sequence_parallel"] else 1
    for index, wrapped_layer in enumerate(model.model.layers):
        layer = _unwrap_activation_module(wrapped_layer)
        for pattern in pik_row_matches(family):
            parent_name, child = pattern.split(".")[-2:]
            parent = getattr(layer, parent_name)
            parent = _unwrap_activation_module(parent)
            sp = topology["sequence_parallel"] and topology["tp"] > 1
            if topology["tp"] > 1:
                _validate_pik_boundary(parent, f"layers.{index}", sp=sp)
            module = getattr(parent, child)
            if topology["tp"] > 1:
                _validate_pik_boundary(module, f"layers.{index}.{child}", sp=sp)
            function = _forward_leaf(module, f"layers.{index}.{child}")
            installed = (isinstance(function, partial) and function.args == (module,)
                         and getattr(function.func, "__code__", None) in codes)
            if not isinstance(module, PikLinear) or not (installed or function is PikLinear.forward):
                raise ValueError(f"PIK row computation was not installed at layer {index} {child}")
            _validate_pik_context(module, options, topology, sp, index, child)


def _validate_pik_context(module: nn.Module, options: dict, topology: dict, sp: bool,
                          index: int, child: str) -> None:
    """Match prepared leaf arithmetic and topology against the shared model contract."""
    context = module.pik_context
    if context is None:
        raise ValueError(f"PIK context is missing at layer {index} {child}")
    if (context.plan.num_leaves != options["num_leaves"]
            or context.plan.leaf_layout != options["leaf_layout"]
            or context.world_size != topology["tp"]
            or context.global_k != module.in_features
            or context.output_features != module.out_features
            or context.sequence_parallel != sp):
        raise ValueError(f"PIK row arithmetic or topology differs at layer {index} {child}")


def _validate_pik_boundary(module: nn.Module, path: str, *, sp: bool) -> None:
    """Audit the resolved HP output contract, including per-layer overrides."""
    function = getattr(module.forward, "__func__", module.forward)
    if getattr(function, "__code__", None) not in _forward_wrapper_targets():
        raise ValueError(f"PIK requires an HP-owned boundary at {path}")
    boundary = inspect.getclosurevars(function).nonlocals.get("boundary")
    if boundary is None:
        raise ValueError(f"PIK cannot audit the HP output boundary at {path}")
    for field in ("out_src", "out_dst"):
        placement = (getattr(boundary.spec, field) or {}).get("output", {}).get("tp")
        valid = placement is not None and (placement.is_shard() and placement.dim == 1
                                           if sp else placement.is_replicate())
        if not valid:
            raise ValueError(f"PIK requires identity reduced/SP output at {path}: {field}")
