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
from typing import Any, Dict, List, Optional

from hyper_parallel.core.optimizer.swap_optimizer import (
    SwapOptimizer,
    SwapOptimizerConfig,
    swap_optimizer,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Torch-only optimizer implementations import torch at module load. Keep them
# off the eager path so MindSpore-only environments can import SwapOptimizer.
_LAZY_EXPORTS = {
    "AdamW": ".adamw",
    "Muon": ".muon",
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
    effective_config = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name not in {"self", "params"}
        and parameter.default is not inspect.Parameter.empty
    }
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


def get_hyper_optimizer(
        model: Any,
        muon_params: List[Dict[str, Any]],
        adamw_params: List[Dict[str, Any]],
        muon_kwargs: Optional[Dict[str, Any]] = None,
        adamw_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create a chained Muon + AdamW optimizer.

    Args:
        model: The neural network model.
        muon_params: Param groups for Muon. Empty list disables Muon.
        adamw_params: Param groups for AdamW. Empty list disables AdamW.
        muon_kwargs: Dedicated configurations dict for Muon.
        adamw_kwargs: Dedicated configurations dict for AdamW.

    Example:
        from hyper_parallel.core.optimizer import get_hyper_optimizer

        _adamw_legacy = {
            'adamw_lr': 1e-3,
            'adamw_weight_decay': 1e-2,
            'adamw_betas': (0.9, 0.95),
            'adamw_eps': 1e-8,
            'fused': True
        }
        _muon_legacy = {
            'muon_lr': 2e-2,
            'muon_weight_decay': 0.1,
            'muon_momentum': 0.95,
            'muon_ns_steps': 5,
            'muon_ns_variant': 'asym5',
            'muon_nesterov': True,
            'muon_hsdp_replica_count': 2
        }

        optimizer = get_hyper_optimizer(
            model=model,
            muon_params=muon_groups,
            adamw_params=adamw_groups,
            adamw_kwargs=_adamw_legacy,
            muon_kwargs=_muon_legacy,
        )

        optimizer.step()
    """
    AdamW, Muon, ChainedOptimizer, detect_dtensor_backend = _load_torch_optimizer_runtime()

    filtered_adamw_config = _filter_optimizer_config(
        "adamw",
        AdamW,
        adamw_kwargs,
    )
    filtered_muon_config = _filter_optimizer_config(
        "muon",
        Muon,
        muon_kwargs,
    )
    optimizers = {}
    detect_dtensor_backend(adamw_params, muon_params)

    if adamw_params:
        optimizers["adamw"] = _build_configured_optimizer(
            "adamw",
            AdamW,
            adamw_params,
            filtered_adamw_config,
        )

    if muon_params:
        optimizers["muon"] = _build_configured_optimizer(
            "muon",
            Muon,
            muon_params,
            filtered_muon_config,
        )

    flatten = bool(adamw_params and muon_params)

    return ChainedOptimizer(
        model,
        optimizers=optimizers,
        flatten=flatten,
    )


__all__ = [
    'SwapOptimizer',
    'SwapOptimizerConfig',
    'get_hyper_optimizer',
    'swap_optimizer',
]
