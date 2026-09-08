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
"""Parameter-name classification and optimizer parameter-group construction."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn  # pylint: disable=forbidden-backend-import

logger = logging.getLogger(__name__)

_DEFAULT_ADAMW_NAME_KEYWORDS: Tuple[str, ...] = (
    "embed",
    "lm_head",
    "out_head",
    "out_proj.bias",
)


def get_parameter_names(
        model: nn.Module,
        forbidden_param_names: Optional[Sequence[str]],
) -> List[str]:
    """Return canonical parameter names that do not match no-decay keywords.

    Transparent module wrappers may expose different internal names through
    ``named_children()`` while normalizing their public ``named_parameters()``
    names. Deriving and consuming names from the same public traversal keeps
    optimizer grouping stable when those wrappers are installed.

    Args:
        model: Model whose parameters should be classified.
        forbidden_param_names: Case-insensitive keywords excluded from weight decay.

    Returns:
        Parameter names eligible for weight decay.
    """
    forbidden_names = tuple(name.lower() for name in forbidden_param_names or ())
    return [
        name
        for name, _ in model.named_parameters()
        if not any(forbidden in name.lower() for forbidden in forbidden_names)
    ]


def get_adamw_param_groups(
        model: "nn.Module",
        weight_decay: float = 1e-2,
        no_decay_params: Optional[Sequence[str]] = None,
        param_groups: Optional[Sequence[Dict[str, Any]]] = None,
        allowed_param_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split model parameters into decaying and non-decaying groups.

    The function also adds ``model_name`` optimizer metadata to each selected
    parameter.

    Args:
        model: Module whose trainable parameters are routed.
        weight_decay: Default decay coefficient for trainable parameters.
        no_decay_params: Name keywords excluded from weight decay.
        param_groups: Optional prebuilt groups to return unchanged.
        allowed_param_ids: Optional parameter identities eligible for routing.

    Returns:
        Parameter groups and names routed to AdamW.
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
            logger.info_rank0(f"Parameters without weight decay: {no_decay_parameter_names}")
            param_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

    return param_groups, adamw_names


def split_muon_adamw_params(
        model: nn.Module,
        extra_adamw_name_keywords: Sequence[str] = (),
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[str], List[str]]:
    """Route matrix parameters to Muon and remaining parameters to AdamW.

    Args:
        model: Module whose trainable parameters are routed.
        extra_adamw_name_keywords: Additional names reserved for AdamW.

    Returns:
        Muon parameters, AdamW parameters, and their respective names.
    """
    adamw_keywords = tuple(
        keyword.lower()
        for keyword in (
            *_DEFAULT_ADAMW_NAME_KEYWORDS,
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


__all__ = [
    "get_adamw_param_groups",
    "get_parameter_names",
    "split_muon_adamw_params",
]
