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
"""Optimizer component configuration types — following design doc §9.2-§9.5."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from hyper_models.config.configurable import Configurable


class Optimizer(Configurable):
    """Base category for optimizer components."""

    @dataclass
    class Config(Configurable.Config):
        """Base configuration accepted by the optimizer slot."""

        max_grad_norm: float = 1.0

        def build(
            self,
            model: nn.Module,
            *,
            optimizer_init: Optional["OptimizerInit"] = None,
            device_mesh: Optional[Any] = None,
            is_peft: bool = False,
        ) -> list[torch.optim.Optimizer]:
            """Build optimizer(s). Subclasses override this."""
            raise NotImplementedError


class AdamW(Optimizer):
    """AdamW optimizer configuration owner."""

    @dataclass
    class Config(Optimizer.Config):
        """AdamW optimizer parameters."""

        lr: float = 1e-4
        weight_decay: float = 0.01
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        foreach: Optional[bool] = None

        def build(
            self,
            model: nn.Module,
            *,
            optimizer_init: Optional["OptimizerInit"] = None,
            device_mesh: Optional[Any] = None,
            is_peft: bool = False,
        ) -> list[torch.optim.Optimizer]:
            """Build AdamW optimizer(s) with param grouping.

            Returns list[Optimizer] (nemo_automodel convention).
            """
            parts = getattr(model, "parts", [model])
            optimizers = []
            for part in parts:
                if optimizer_init is not None and getattr(optimizer_init, "param_groups", None):
                    param_groups = optimizer_init.param_groups
                else:
                    param_groups = _build_param_groups(part, self.weight_decay)
                optimizers.append(torch.optim.AdamW(
                    param_groups,
                    lr=self.lr, betas=self.betas, eps=self.eps,
                    foreach=self.foreach if self.foreach is not None else True,
                ))
            return optimizers

    def __init__(self, config: "AdamW.Config") -> None:
        self.config = config


class OptimizerFromFactoryConfig(Optimizer.Config):
    """External optimizer (e.g., dion.Muon) escape hatch."""

    factory: Optional[Callable] = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, factory=None, kwargs=None):
        self.factory = factory
        self.kwargs = kwargs or {}

    def build(
        self,
        model: nn.Module,
        *,
        optimizer_init: Optional["OptimizerInit"] = None,
        device_mesh: Optional[Any] = None,
        is_peft: bool = False,
    ) -> list[torch.optim.Optimizer]:
        """Build external optimizer."""
        if optimizer_init is not None and getattr(optimizer_init, "param_groups", None):
            param_groups = optimizer_init.param_groups
        else:
            param_groups = _build_param_groups(model, self.kwargs.get("weight_decay", 0.1))
        return [self.factory(param_groups, **{k: v for k, v in self.kwargs.items() if k != "weight_decay"})]


# ── Registry ──

OPTIMIZER_CONFIG_REGISTRY: dict[str, type] = {
    "adamw": AdamW.Config,
    "torch.optim.adamw": AdamW.Config,
}


def build_optimizer_config(
    target: Any,
    kwargs: Optional[dict] = None,
) -> Optimizer.Config:
    """Normalize _target_ factory + kwargs into OptimizerConfig instance.

    Following design doc §9.4.
    """
    if isinstance(target, Optimizer.Config):
        return target
    if isinstance(target, str):
        resolved = OPTIMIZER_CONFIG_REGISTRY.get(target.lower())
        if resolved is None:
            resolved = _import_from_path(target)
        target = resolved
    kwargs = dict(kwargs or {})
    if isinstance(target, type) and issubclass(target, Optimizer.Config):
        return target(**kwargs)
    if callable(target):
        return OptimizerFromFactoryConfig(factory=target, kwargs=kwargs)
    raise TypeError(f"Unsupported optimizer target: {target!r}")


def _import_from_path(path: str) -> type:
    """Import a class from a dotted path string."""
    import importlib
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay/no_decay parameter grouping.

    Following design doc §9.5.
    """
    decay_params, no_decay_params = [], []
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_ids:
            continue
        seen_ids.add(param_id)

        if _is_no_decay(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _is_no_decay(name: str) -> bool:
    """Check if a parameter name should be excluded from weight decay.

    Following design doc §9.5.
    """
    no_decay_patterns = ("bias", "norm", "rmsnorm", "layernorm", "ln_")
    return any(pattern in name.lower() for pattern in no_decay_patterns)


# OptimizerInit dataclass (used by build_model to export param groups)
from dataclasses import dataclass as _dataclass


@_dataclass
class OptimizerInit:
    """Optimizer initialization description exported by build_model.

    Following design doc §6.7.
    """
    param_groups: list[dict]
    device_mesh: Optional[Any] = None
    is_peft: bool = False
    tp_grad_info: Any = None

    @classmethod
    def from_distributed_setup(
        cls,
        *,
        distributed_setup=None,
        model: nn.Module = None,
        peft_config=None,
        weight_decay: float = 0.0,
    ) -> "OptimizerInit":
        """Derive param groups from distributed setup + model."""
        mesh_ctx = getattr(distributed_setup, "mesh_context", None) if distributed_setup else None
        device_mesh = mesh_ctx.device_mesh if mesh_ctx is not None else None
        is_peft = peft_config is not None

        decay_p, no_decay_p = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay_p if _is_no_decay(name) else decay_p).append(param)
        param_groups = [
            {"params": decay_p, "weight_decay": weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ]
        return cls(
            param_groups=param_groups,
            device_mesh=device_mesh,
            is_peft=is_peft,
        )


__all__ = [
    "Optimizer", "AdamW",
    "OptimizerFromFactoryConfig",
    "build_optimizer_config",
    "OPTIMIZER_CONFIG_REGISTRY",
    "OptimizerInit",
]