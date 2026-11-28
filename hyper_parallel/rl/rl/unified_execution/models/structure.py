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

import inspect
from fnmatch import fnmatchcase
from functools import lru_cache
from types import CodeType
from typing import Any

from torch import nn

from hyper_parallel.components.modules.attention.grouped_query_attention.grouped_query_attention import (
    GroupedQueryAttentionModule,
)
from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsModule
from hyper_parallel.core.activation_memory.wrapper import CheckpointExcludeWrapper, CheckpointWrapper, SwapWrapper
from hyper_parallel.distributed._builder import forward_rewriter
from rl.unified_execution.contract import fingerprint, parse_unified_payload
from rl.unified_execution.execution_plan import (
    pik_row_matches,
    pik_row_options,
)
from rl.unified_execution.models import get_model_family


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
    for path, module in model.named_modules():
        if isinstance(module, GroupedExpertsModule):
            modules[path] = "qwen3_moe.packed_experts"
        if isinstance(module, GroupedQueryAttentionModule):
            modules[path] = f"{GroupedQueryAttentionModule.__module__}.{GroupedQueryAttentionModule.__qualname__}"
    return {"modules": modules, "parameters": parameters, "aliases": aliases}


@lru_cache(maxsize=1)
def _forward_wrapper_targets() -> dict[CodeType, str]:
    """Identify HP boundaries. Local regions may execute a replacement instead of __wrapped__."""
    targets = {}
    # HP has no public wrapper-introspection API. Code identities preserve its
    # TP/SP boundaries while preventing __wrapped__ from hiding custom compute.
    factories = {
        "_wrap_production_forward": ("production_forward", "__wrapped__"),
        "_wrap_validate_forward": ("validate_forward", "__wrapped__"),
        "_wrap_local_region_forward": ("local_region_forward", "compute_fn"),
        "_wrap_module_forward": ("local_param_forward", "__wrapped__"),
        "_install_inner_adapter": ("adapted", "__wrapped__"),
    }
    for factory_name, (wrapper_name, target_name) in factories.items():
        factory = getattr(forward_rewriter, factory_name)
        for constant in factory.__code__.co_consts:
            if isinstance(constant, CodeType) and constant.co_name == wrapper_name:
                targets[constant] = target_name
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
        raise ValueError(f"Cannot verify the unified execution path at {path}")


def _forward_leaf(module: nn.Module, path: str) -> Any:
    """Follow only known HP wrappers to the function they actually execute."""
    function = module.forward
    visited = set()
    while True:
        function = getattr(function, "__func__", function)
        if id(function) in visited:
            raise ValueError(f"Unified training found a cyclic forward wrapper at {path}")
        visited.add(id(function))
        if function is getattr(module, "_rl_token_log_probs_forward", None):
            function = function.__wrapped__
            continue
        target = _forward_wrapper_targets().get(getattr(function, "__code__", None))
        if target is None:
            return function
        function = (function.__wrapped__ if target == "__wrapped__"
                    else inspect.getclosurevars(function).nonlocals[target])


def validate_gqa_execution_path(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
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
        allowed = (owners["attention"].forward, GroupedQueryAttentionModule.forward)
        _require_forward_path(attention, allowed, f"layers.{index}.self_attn.forward")
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
        _validate_experts(model, family.execution_types(), ep_size, inference=inference)
    _validate_pik_components(model, payload, inference=inference)
    structure = model_structure(model)
    replaced = set(getattr(model, "_hp_replaced_module_fqns", ()))
    parallelism = payload["parallelism" if inference else "training_parallelism"]
    for entry in payload["compute_plan_overrides"]:
        condition = entry.get("when")
        enabled = {"cp": not inference and parallelism["cp"] > 1, "ep": False, "low_precision": False,
                   "sequence_parallel": parallelism["sequence_parallel"]}
        if condition is not None and not enabled[condition]:
            continue
        patterns = entry["match"] if isinstance(entry["match"], list) else [entry["match"]]
        for pattern in patterns:
            matches = {name for name in structure["modules"] if fnmatchcase(name, pattern)}
            if not matches or not matches <= replaced:
                raise ValueError(f"Unified model did not apply module replacement {pattern!r}")
    return {"structure": structure, "fingerprint": fingerprint(structure)}


def _validate_experts(model: nn.Module, owners: dict[str, type], ep_size: int, *, inference: bool) -> None:
    """Verify native HP module ownership while retaining the model's router."""
    # Optional runtime classes are imported only for an MoE structure audit.
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.components.modules.moe.grouped_experts.ep.ep_grouped_experts import EPGroupedExperts

    for index, layer in enumerate(model.model.layers):
        mlp = _unwrap_activation_module(_unwrap_activation_module(layer).mlp)
        if not hasattr(mlp, "experts"):
            continue
        experts = _unwrap_activation_module(mlp.experts)
        path = f"layers.{index}.mlp"
        if experts.config.routing_weight_mode != "output":
            raise ValueError(f"Unified MoE requires source-side router weighting at {path}")
        _require_forward_path(mlp, (owners["moe_block"].forward,), path)
        _require_forward_path(mlp.gate, (owners["router"].forward,), path + ".gate")
        if inference and ep_size > 1:
            from rl.unified_execution.vllm.moe.grouped_experts import RolloutEPGroupedExperts
            _require_forward_path(experts, (RolloutEPGroupedExperts.forward,), path + ".experts")
        else:
            _require_forward_path(experts, (GroupedExpertsModule.forward,), path + ".experts")
            if ep_size > 1 and (not isinstance(experts, EPGroupedExperts) or experts.ep_context.mesh.size() != ep_size):
                raise ValueError(f"Training expert ownership differs at {path}")


def training_sequence_multiple(model: nn.Module, *, tp_size: int, cp_size: int,
                               sequence_parallel: bool) -> int:
    """Align complete input rows to every native training sequence boundary.

    HP also splits the routed MLP input across TP when EP is active, including
    non-SP models. Its TP chunks must have equal length before gathering them
    back into the residual stream. CP partitions the already aligned row.
    """
    expert_tp = tp_size > 1 and any(
        isinstance(module, GroupedExpertsModule)
        and getattr(module, "ep_context", None) is not None
        and module.ep_context.mesh.size() > 1
        for module in model.modules()
    )
    return cp_size * (tp_size if sequence_parallel or expert_tp else 1)


def _validate_pik_components(model: nn.Module, payload: dict, *, inference: bool) -> None:
    """Check every row replacement after HP build, before model execution."""
    family = payload.get("model_family", "qwen3")
    options = pik_row_options(payload["plan_overrides"], family)
    if options is None:
        return
    # The NPU computation adapter is optional for non-PIK configurations.
    from rl.unified_execution.modules.row_parallel_linear import PikLinear  # pylint: disable=C0415

    topology = payload["parallelism" if inference else "training_parallelism"]
    if inference:
        model.unified_sequence_multiple = topology["tp"] if topology["sequence_parallel"] else 1
    else:
        model.unified_sequence_multiple = training_sequence_multiple(
            model, tp_size=topology["tp"], cp_size=topology["cp"],
            sequence_parallel=topology["sequence_parallel"],
        )
    for index, wrapped_layer in enumerate(model.model.layers):
        layer = _unwrap_activation_module(wrapped_layer)
        for pattern in pik_row_matches(family):
            parent_name, child = pattern.removeprefix("*.").rsplit(".", 1)
            parent = layer.get_submodule(parent_name)
            parent = _unwrap_activation_module(parent)
            if not inference and topology["cp"] > 1 and parent_name == get_model_family(family).attention_attribute:
                cp_mesh = getattr(getattr(parent, "cp_context", None), "mesh", None)
                if cp_mesh is None or cp_mesh.size() != topology["cp"]:
                    raise ValueError("Training CP attention was not bound to the requested HP mesh")
            sp = topology["sequence_parallel"] and topology["tp"] > 1
            if topology["tp"] > 1 and (parent_name != get_model_family(family).attention_attribute or sp):
                # The native planner elides an identity GQA boundary for non-SP TP.
                _validate_pik_boundary(parent, f"layers.{index}", sp=sp)
            module = getattr(parent, child)
            function = _forward_leaf(module, f"layers.{index}.{child}")
            if not isinstance(module, PikLinear) or function is not PikLinear.forward:
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
