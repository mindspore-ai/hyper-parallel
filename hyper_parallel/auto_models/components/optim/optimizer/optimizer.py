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
"""YAML-targeted optimizer implementations."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn

from hyper_parallel.auto_models.components.distributed.init_utils import (
    get_global_rank_safe,
)
from hyper_parallel.core.optimizer import get_hyper_optimizer

logger = logging.getLogger(__name__)


def _info_rank0(message: str, *args: Any) -> None:
    """Log an informational message only on global rank zero."""
    if get_global_rank_safe() == 0:
        logger.info(message, *args)


# adapted from https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer_pt_utils.py#L1123
def get_parameter_names(
        model: nn.Module,
        forbidden_param_names: Optional[Sequence[str]],
) -> List[str]:
    """Return parameter names that do not match no-decay name keywords."""
    forbidden_param_names = [] if forbidden_param_names is None else forbidden_param_names
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_param_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_param_names)
        ]

    result += [
        name
        for name, _ in model.named_parameters(recurse=False)
        if not any(forbidden in name.lower() for forbidden in forbidden_param_names)
    ]
    return result


class AdamW:
    """Build a core AdamW optimizer from YAML configuration."""

    def __init__(
            self,
            adamw_config: dict,
            model: nn.Module,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Initialize AdamW optimizer configuration.

        Args:
            lr: Learning rate.
            weight_decay: Weight decay for decay parameter groups.
            betas: AdamW coefficient pair.
            eps: AdamW numerical-stability term.
            foreach: Whether to use the foreach implementation.
        """
        self.config = adamw_config
        self.model = model

        adamw_groups, _ = self.get_adamw_param_groups(
            self.model,
            weight_decay=adamw_config.get("adamw_weight_decay", 1e-2),
            no_decay_params=no_decay_params,
        )
        if not adamw_groups:
            raise ValueError("AdamW requires at least one trainable parameter")

        self.optimizer = get_hyper_optimizer(
            model=self.model,
            muon_params=[],
            adamw_params=adamw_groups,
            adamw_kwargs=adamw_config,
        )

    @staticmethod
    def get_adamw_param_groups(
            model: "nn.Module",
            weight_decay: float = 1e-2,
            no_decay_params: Optional[Sequence[str]] = None,
            param_groups: Optional[Sequence[Dict[str, Any]]] = None,
            allowed_param_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Extracts model parameters and splits them into decaying and non-decaying groups.
        It also injects metadata attributes ('model_name' and 'is_muon') into each parameter.

        Args:
            model: The PyTorch module to extract parameters from.
            weight_decay: The default weight decay coefficient for trainable parameters.
            no_decay_params: Specific parameter name keywords to exclude from weight decay (e.g., ['bias']).

        Returns:
            param_groups: A list of parameter groups compatible with PyTorch optimizers.
            adamw_names: A list of all parameter names targeted for AdamW.
        """
        adamw_names = []
        no_decay_keywords = ("bias", "norm", "ln_") if no_decay_params is None else no_decay_params
        no_decay_params = tuple(keyword.lower() for keyword in no_decay_keywords)

        if param_groups is None:
            decay_param_names = set(get_parameter_names(model, no_decay_params))
            allowed_ids = set(allowed_param_ids) if allowed_param_ids is not None else None
            decay_parameters = []
            no_decay_parameters = []
            no_decay_parameter_names = []
            for n, p in model.named_parameters():
                if not p.requires_grad or (allowed_ids is not None and id(p) not in allowed_ids):
                    continue
                setattr(p, "model_name", n)
                adamw_names.append(n)
                if n in decay_param_names:
                    decay_parameters.append(p)
                else:
                    no_decay_parameter_names.append(n)
                    no_decay_parameters.append(p)

            param_groups = []
            if decay_parameters:
                param_groups.append({"params": decay_parameters, "weight_decay": weight_decay})
            if no_decay_parameters:
                _info_rank0(
                    "Parameters without weight decay: %s",
                    no_decay_parameter_names,
                )
                param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

        return param_groups, adamw_names

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer


class Muon:
    """Build a mixed Muon and AdamW optimizer from YAML configuration."""

    _DEFAULT_ADAMW_NAME_KEYWORDS: Tuple[str, ...] = (
        "embed",
        "lm_head",
        "out_head",
        "out_proj.bias",
    )

    def __init__(
            self,
            muon_config: dict,
            adamw_config: dict,
            model: nn.Module,
            extra_adamw_name_keywords: Optional[List[str]] = None,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Build a mixed Muon and fallback AdamW runtime for ``model``."""
        self.muon_config = muon_config
        self.adamw_config = adamw_config
        self.model = model

        muon_params, adamw_params, muon_names, adamw_names = self.split_muon_adamw_params(
            model,
            extra_adamw_name_keywords=extra_adamw_name_keywords or (),
        )
        if not muon_params:
            raise ValueError("Muon requires at least one eligible matrix parameter")

        adamw_groups, _ = AdamW.get_adamw_param_groups(
            model,
            weight_decay=adamw_config.get("adamw_weight_decay", 1e-2),
            no_decay_params=no_decay_params,
            allowed_param_ids=[id(parameter) for parameter in adamw_params],
        )

        _info_rank0(
            "Muon optimizer split: %s Muon parameters, %s AdamW parameters",
            len(muon_names),
            len(adamw_names),
        )
        _info_rank0("Muon parameters (first 5): %s", muon_names[:5])
        _info_rank0("AdamW parameters (first 5): %s", adamw_names[:5])

        self.optimizer = get_hyper_optimizer(
            model=self.model,
            muon_params=muon_params,
            adamw_params=adamw_groups,
            muon_kwargs=muon_config,
            adamw_kwargs=adamw_config,
        )

    @staticmethod
    def split_muon_adamw_params(
            model: nn.Module,
            extra_adamw_name_keywords: Sequence[str] = (),
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[str], List[str]]:
        """Route matrix parameters to Muon and the remaining parameters to AdamW."""
        adamw_keywords = tuple(
            keyword.lower()
            for keyword in (
                *Muon._DEFAULT_ADAMW_NAME_KEYWORDS,
                *extra_adamw_name_keywords,
            )
        )
        muon_params, adamw_params = [], []
        muon_names, adamw_names = [], []

        for name, parameter in model.named_parameters():
            setattr(parameter, "model_name", name)
            if not parameter.requires_grad:
                continue

            is_matrix = parameter.ndim >= 2
            is_adamw = any(keyword in name.lower() for keyword in adamw_keywords)
            if is_matrix and not is_adamw:
                muon_params.append(parameter)
                muon_names.append(name)
            else:
                adamw_params.append(parameter)
                adamw_names.append(name)

        return muon_params, adamw_params, muon_names, adamw_names

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer


__all__ = ["AdamW", "Muon"]
