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


def get_hyper_lr_scheduler(*args: Any, **kwargs: Any) -> Any:
    """Create the HyperParallel LR scheduler."""
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.optimizer.lr_scheduler import (
        get_hyper_lr_scheduler as _get_hyper_lr_scheduler,
    )
    return _get_hyper_lr_scheduler(*args, **kwargs)


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

    # 1. Arguments Preparation
    # 1.1 adamw
    adamw_raw = adamw_kwargs or {}
    adamw_config = {
        k[6:] if k.startswith("adamw_") else k: v
        for k, v in adamw_raw.items()
    }
    allowed_keys_adamw = inspect.signature(AdamW.__init__).parameters.keys() - {'self', 'params'}
    filtered_adamw_config = {k: v for k, v in adamw_config.items() if k in allowed_keys_adamw}
    if excluded_adamw_keys := adamw_config.keys() - allowed_keys_adamw:
        logger.info_rank0("Excluded adamw config: %s", list(excluded_adamw_keys))

    # 1.2 muon
    muon_raw = muon_kwargs or {}
    muon_config = {
        k[5:] if k.startswith("muon_") else k: v
        for k, v in muon_raw.items()
    }
    allowed_keys_muon = inspect.signature(Muon.__init__).parameters.keys() - {'self', 'params'}
    filtered_muon_config = {k: v for k, v in muon_config.items() if k in allowed_keys_muon}
    if excluded_muon_keys := muon_config.keys() - allowed_keys_muon:
        logger.info_rank0("Excluded muon config: %s", list(excluded_muon_keys))

    # 2. Optimizer Creation
    optimizers = {}
    detect_dtensor_backend(adamw_params, muon_params)

    # build optimizer
    if adamw_params:
        optimizers["adamw"] = AdamW(adamw_params, **filtered_adamw_config)
        logger.info_rank0("Using adamw config: %s", filtered_adamw_config)

    if muon_params:
        optimizers["muon"] = Muon(muon_params, **filtered_muon_config)
        logger.info_rank0("Using muon config: %s", filtered_muon_config)

    flatten = bool(adamw_params and muon_params)

    return ChainedOptimizer(model, optimizers=optimizers, flatten=flatten)


__all__ = [
    'SwapOptimizer',
    'SwapOptimizerConfig',
    'get_hyper_optimizer',
    'get_hyper_lr_scheduler',
    'swap_optimizer',
]
