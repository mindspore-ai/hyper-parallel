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
"""HyperParallel optimizer module."""
from importlib import import_module as _import_module  # pylint: disable=invalid-name

import inspect
import logging
import re
from typing import Any, Dict, List, Optional

from hyper_parallel.core.optimizer.swap_optimizer import (
    SwapOptimizer,
    SwapOptimizerConfig,
    swap_optimizer,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_HEAD_PATTERN = re.compile(r"(?:^|\.)(?:lm_head|output|output_layer)\.weight$")

# Optimizer implementations import torch at module load. Keep them off the
# eager path so importing the package does not pull in torch.
_LAZY_EXPORTS = {
    "AdamW": ".adamw",
    "Muon": ".muon",
    "Sinkhorn": ".sinkhorn",
    "ChainedOptimizer": ".optimizer",
    "detect_dtensor_backend": ".dtensor_compat",
}


def __getattr__(name):  # pylint: disable=invalid-name
    """Lazily import torch-only optimizer symbols."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pylint: disable=invalid-name
    """Include lazy torch-only optimizer exports in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


def _load_torch_optimizer_runtime():
    """Import torch-only optimizer helpers used by the factory APIs."""
    # pylint: disable=import-outside-toplevel,unused-import
    import hyper_parallel.core.optimizer.utils  # noqa: F401 - install rank0 logging helpers

    from hyper_parallel.core.optimizer.adamw import AdamW
    from hyper_parallel.core.optimizer.dtensor_compat import detect_dtensor_backend
    from hyper_parallel.core.optimizer.muon import Muon
    from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer

    return AdamW, Muon, ChainedOptimizer, detect_dtensor_backend


def _effective_optimizer_config(
    optimizer_class: Any,
    configured_values: Dict[str, Any],
    runtime_optimizer: Any,
) -> Dict[str, Any]:
    """Merge constructor defaults, user values, and resolved runtime defaults."""
    signature = inspect.signature(optimizer_class.__init__)
    effective_config = {}
    for name, parameter in signature.parameters.items():
        if name not in {"self", "params"} and parameter.default is not inspect.Parameter.empty:
            effective_config[name] = parameter.default
    effective_config.update(configured_values)
    effective_config.update(runtime_optimizer.defaults)
    if hasattr(runtime_optimizer, "hsdp_replica_count"):
        effective_config["hsdp_replica_count"] = runtime_optimizer.hsdp_replica_count
    return effective_config


def _filter_optimizer_config(
        optimizer_name: str,
        optimizer_class: Any,
        configured_values: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize prefixed YAML keys and remove unsupported constructor args."""
    prefix = f"{optimizer_name}_"
    normalized_config = {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in (configured_values or {}).items()
    }
    allowed_keys = (
        inspect.signature(optimizer_class.__init__).parameters.keys()
        - {"self", "params"}
    )
    allowed_keys = set(allowed_keys) | set(getattr(optimizer_class, "ADDITIONAL_CONFIG_KEYS", ()))
    filtered_config = {
        key: value
        for key, value in normalized_config.items()
        if key in allowed_keys
    }
    excluded_keys = normalized_config.keys() - allowed_keys
    if excluded_keys:
        logger.info_rank0(
            "Excluded %s config: %s",
            optimizer_name,
            list(excluded_keys),
        )
    return filtered_config


def _build_configured_optimizer(
        optimizer_name: str,
        optimizer_class: Any,
        param_groups: Any,
        configured_values: Dict[str, Any],
) -> Any:
    """Construct one leaf optimizer and log its effective configuration."""
    optimizer = optimizer_class(param_groups, **configured_values)
    logger.info_rank0(
        f"Effective {optimizer_name} config: %s",
        _effective_optimizer_config(
            optimizer_class,
            configured_values,
            optimizer,
        ),
    )
    return optimizer


def _build_optimizer_groups(
        model: Any,
        configs: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Route each trainable parameter exactly once, honoring explicit regex selectors.

    Args:
        model: Model containing the parameters to optimize.
        configs: Enabled optimizer names mapped to options, optionally including
            ``param_patterns`` (a list of regular expressions matching any alias).

    Returns:
        Named lists of parameter groups, including AdamW decay exclusions.
    """
    aliases, embedding_ids, norm_ids = _parameter_roles(model)
    selectors = _compile_selectors(configs)
    selected = {name: [] for name in configs}
    for param, names in aliases.items():
        family = _explicit_family(names, selectors)
        if family is None:
            family = _default_family(param, names, embedding_ids, norm_ids, configs)
        if family not in selected:
            raise ValueError(f"No enabled optimizer owns trainable parameter {names[0]}")
        if family in ("muon", "sinkhorn") and param.ndim < 2:
            raise ValueError(f"{family} requires matrix parameters, got {names[0]}")
        selected[family].append(param)
    return {family: _make_groups(family, params, aliases, norm_ids) for family, params in selected.items()}


def _parameter_roles(model):
    """Collect aliases once and recognize embedding and normalization modules."""
    # Preserve the package's lazy Torch import contract.
    embedding_class = _import_module("torch.nn").Embedding
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if param.requires_grad:
            aliases.setdefault(param, []).append(name)
    embedding_ids = {id(module.weight) for module in model.modules() if isinstance(module, embedding_class)}
    norm_ids = {id(param) for module in model.modules()
                if re.search(r"norm(?:[123]d|gated)?$", type(module).__name__, re.IGNORECASE)
                for param in module.parameters(recurse=False)}
    return aliases, embedding_ids, norm_ids


def _compile_selectors(configs):
    """Compile explicit selectors without modifying the supplied configs."""
    selectors = {}
    for family, config in configs.items():
        patterns = config.get("param_patterns", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        selectors[family] = [re.compile(pattern) for pattern in patterns]
    return selectors


def _explicit_family(names, selectors):
    """Resolve at most one explicitly requested optimizer across all tied aliases."""
    explicit = [family for family, patterns in selectors.items()
                if any(pattern.search(name) for pattern in patterns for name in names)]
    if len(explicit) > 1:
        raise ValueError(f"Overlapping optimizer selectors for {names}: {explicit}")
    return explicit[0] if explicit else None


def _make_groups(family, params, aliases, norm_ids):
    """Separate AdamW decay exclusions; matrix optimizers need only one group."""
    if family != "adamw":
        return [{"params": params}] if params else []
    no_decay = [param for param in params if (param.ndim < 2 and id(param) not in norm_ids) or any(
        name.rsplit(".", 1)[-1] in ("bias", "scale", "scales", "scaling_factor") for name in aliases[param]
    )]
    excluded_ids = {id(param) for param in no_decay}
    decay = [param for param in params if id(param) not in excluded_ids]
    groups = [{"params": decay}] if decay else []
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _default_family(param, names, embedding_ids, norm_ids, configs):
    """Apply semantic embedding routing before general matrix/vector routing."""
    if id(param) in norm_ids:
        return "adamw"
    if id(param) in embedding_ids or any(_HEAD_PATTERN.search(name) for name in names):
        return "sinkhorn" if "sinkhorn" in configs else "adamw"
    if param.ndim >= 2 and "muon" in configs:
        return "muon"
    return "adamw"


def _infer_head_dim(model: Any) -> Any:
    """Infer default attention head width from model configuration.

    Args:
        model: Module exposing an optional attention configuration.

    Returns:
        Head width, or None when it cannot be inferred.
    """
    config = getattr(model, "config", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    hidden = getattr(config, "hidden_size", None)
    heads = getattr(config, "num_attention_heads", None)
    if isinstance(hidden, int) and isinstance(heads, int) and heads > 0 and hidden % heads == 0:
        return hidden // heads
    return None


def _resolve_optimizer_groups(model, classes, automatic, explicit):
    """Resolve exactly one configuration mode without changing caller dictionaries."""
    if automatic:
        if any(value is not None for values in explicit.values() for value in values):
            raise ValueError("Automatic optimizer configs cannot be mixed with explicit groups/kwargs")
        groups = _build_optimizer_groups(model, automatic)
        configs = {name: {key: value for key, value in config.items() if key != "param_patterns"}
                   for name, config in automatic.items()}
    else:
        groups = {name: params or [] for name, (params, _) in explicit.items()}
        configs = {name: _filter_optimizer_config(name, classes[name], config)
                   for name, (_, config) in explicit.items()}
    return groups, configs


def get_hyper_optimizer(
        model: Any,
        muon_params: Optional[List[Dict[str, Any]]] = None,
        adamw_params: Optional[List[Dict[str, Any]]] = None,
        muon_kwargs: Optional[Dict[str, Any]] = None,
        adamw_kwargs: Optional[Dict[str, Any]] = None,
        *,
        sinkhorn_params: Optional[List[Dict[str, Any]]] = None,
        sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
        muon: Optional[Dict[str, Any]] = None,
        sinkhorn: Optional[Dict[str, Any]] = None,
        adamw: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a named optimizer composition with explicit or automatically routed groups.

    Args:
        model: Model whose parameter names are used for routing and checkpoints.
        muon_params: Explicit Muon groups (legacy API).
        adamw_params: Explicit AdamW groups (legacy API).
        muon_kwargs: Muon options for explicit groups, including legacy prefixed keys.
        adamw_kwargs: AdamW options for explicit groups.
        sinkhorn_params: Explicit Sinkhorn groups.
        sinkhorn_kwargs: Sinkhorn options for explicit groups.
        muon: Enable automatically routed Muon with these options; None disables it.
        sinkhorn: Enable Sinkhorn for embeddings and prediction heads.
        adamw: Enable AdamW for remaining parameters. Each automatic config may
            include ``param_patterns`` for explicit regex selection.

    Returns:
        ChainedOptimizer containing the non-empty named optimizers.

    Raises:
        ValueError: Configuration modes are mixed, ownership overlaps, or automatic
            routing leaves a trainable parameter unassigned.
    """
    AdamW, Muon, ChainedOptimizer, detect_dtensor_backend = _load_torch_optimizer_runtime()
    # These imports stay lazy for package-level optimizer discovery without torch.
    from hyper_parallel.core.optimizer.sinkhorn import Sinkhorn  # pylint: disable=import-outside-toplevel

    classes = {"adamw": AdamW, "muon": Muon, "sinkhorn": Sinkhorn}
    automatic = {name: dict(config) for name, config in
                 (("adamw", adamw), ("muon", muon), ("sinkhorn", sinkhorn)) if config is not None}
    explicit = {"adamw": (adamw_params, adamw_kwargs), "muon": (muon_params, muon_kwargs),
                "sinkhorn": (sinkhorn_params, sinkhorn_kwargs)}
    groups, configs = _resolve_optimizer_groups(model, classes, automatic, explicit)
    for name, param in model.named_parameters():
        param.model_name = name
    muon_config = configs.get("muon", {})
    if muon_config.get("head_wise") and muon_config.get("head_dim") is None:
        muon_config["head_dim"] = _infer_head_dim(model)
    all_groups = [group for family_groups in groups.values() for group in family_groups]
    detect_dtensor_backend(all_groups, [])
    optimizers = {
        name: _build_configured_optimizer(name, classes[name], family_groups, configs[name])
        for name, family_groups in groups.items() if family_groups
    }
    if not optimizers:
        raise ValueError("At least one non-empty optimizer parameter group is required")
    return ChainedOptimizer(model, optimizers=optimizers)


__all__ = [
    'SwapOptimizer',
    'SwapOptimizerConfig',
    'get_hyper_optimizer',
    'swap_optimizer',
]
