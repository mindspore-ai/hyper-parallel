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
"""YAML-targeted builders for individual and composed optimizers.

Parameter-name and parameter-group logic lives in
:mod:`hyper_parallel.components.optim.parameter_groups` and the core routing helpers; the optimizer
algorithm implementations stay in ``hyper_parallel.core.optimizer``.
"""

__all__ = ["AdamW", "Muon", "Sinkhorn", "ComposedOptimizer"]

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer import _build_optimizer_groups, get_hyper_optimizer
from hyper_parallel.components.optim.parameter_groups import (
    _DEFAULT_ADAMW_NAME_KEYWORDS,
    get_adamw_param_groups,
    split_muon_adamw_params,
)

logger = logging.getLogger(__name__)


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
            adamw_config: AdamW hyperparameters resolved from YAML.
            model: Module whose trainable parameters are optimized.
            no_decay_params: Optional names excluded from weight decay.
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
        """Split model parameters into decaying and non-decaying groups.

        Delegates to
        :func:`hyper_parallel.components.optim.parameter_groups.get_adamw_param_groups`;
        retained on the class so existing call sites keep working.

        Args:
            model: Module whose trainable parameters are routed.
            weight_decay: Default decay coefficient for trainable parameters.
            no_decay_params: Name keywords excluded from weight decay.
            param_groups: Optional prebuilt groups to return unchanged.
            allowed_param_ids: Optional parameter identities eligible for routing.

        Returns:
            Parameter groups and names routed to AdamW.
        """
        return get_adamw_param_groups(
            model,
            weight_decay=weight_decay,
            no_decay_params=no_decay_params,
            param_groups=param_groups,
            allowed_param_ids=allowed_param_ids,
        )

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer


class Muon:
    """Build a mixed Muon and AdamW optimizer from YAML configuration."""

    _DEFAULT_ADAMW_NAME_KEYWORDS = _DEFAULT_ADAMW_NAME_KEYWORDS

    def __init__(
            self,
            muon_config: dict,
            adamw_config: dict,
            model: nn.Module,
            extra_adamw_name_keywords: Optional[List[str]] = None,
            no_decay_params: Optional[List[str]] = None,
    ) -> None:
        """Build a mixed Muon and fallback AdamW runtime for ``model``.

        Args:
            muon_config: Muon hyperparameters resolved from YAML.
            adamw_config: Fallback AdamW hyperparameters resolved from YAML.
            model: Module whose trainable parameters are optimized.
            extra_adamw_name_keywords: Additional names routed to AdamW.
            no_decay_params: Optional names excluded from weight decay.
        """
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

        logger.info_rank0(
            "Muon optimizer split: %s Muon parameters, %s AdamW parameters",
            len(muon_names),
            len(adamw_names),
        )
        logger.info_rank0("Muon parameters (first 5): %s", muon_names[:5])
        logger.info_rank0("AdamW parameters (first 5): %s", adamw_names[:5])

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
        """Route matrix parameters to Muon and remaining parameters to AdamW.

        Delegates to
        :func:`hyper_parallel.components.optim.parameter_groups.split_muon_adamw_params`;
        retained on the class so existing call sites keep working.

        Args:
            model: Module whose trainable parameters are routed.
            extra_adamw_name_keywords: Additional names reserved for AdamW.

        Returns:
            Muon parameters, AdamW parameters, and their respective names.
        """
        return split_muon_adamw_params(
            model,
            extra_adamw_name_keywords=extra_adamw_name_keywords,
        )

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer


def _compile_param_group_specs(configured_groups):
    """Separate parameter selectors from ordinary optimizer-group options."""
    if not isinstance(configured_groups, (list, tuple)):
        raise ValueError("Configured param_groups must be a list of mappings")
    selectors, options = [], []
    for spec in configured_groups:
        if not isinstance(spec, dict):
            raise ValueError("Each configured parameter group must be a mapping")
        group_options = dict(spec)
        patterns = group_options.pop("param_patterns", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if not patterns or "params" in group_options:
            raise ValueError("Configured param_groups require param_patterns and cannot contain runtime params")
        selectors.append([re.compile(pattern) for pattern in patterns])
        options.append(group_options)
    return selectors, options


def _resolve_param_groups(
        model: nn.Module,
        groups: List[Dict[str, Any]],
        configured_groups: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve named group options before constructing an optimizer."""
    if not configured_groups:
        return groups
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(param, []).append(name)
    selectors, options = _compile_param_group_specs(configured_groups)
    matched = set()
    result = []
    for group in groups:
        partitions = {}
        for param in group["params"]:
            matches = [index for index, patterns in enumerate(selectors)
                       if any(pattern.search(name) for pattern in patterns for name in aliases[param])]
            if len(matches) > 1:
                raise ValueError(f"Overlapping param_groups for {aliases[param]}")
            index = matches[0] if matches else None
            if index is not None:
                matched.add(index)
            partitions.setdefault(index, []).append(param)
        for index, params in partitions.items():
            result.append({**group, **(options[index] if index is not None else {}), "params": params})
    if len(matched) != len(selectors):
        raise ValueError("Configured param_groups match no parameters in their optimizer family")
    return result


class ComposedOptimizer:
    """Build Muon, Sinkhorn, and AdamW from declarative Trainer configuration."""

    def __init__(
            self,
            model: nn.Module,
            muon: Optional[Dict[str, Any]] = None,
            sinkhorn: Optional[Dict[str, Any]] = None,
            adamw: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Route model parameters and construct enabled optimizer families.

        Args:
            model: Model after distributed parameter transformations.
            muon: Core Muon options and optional param_patterns; None disables it.
            sinkhorn: Core Sinkhorn options and optional param_patterns.
            adamw: Core AdamW options and optional param_patterns. Each enabled
                family may contain param_groups, a list of param_patterns plus
                absolute optimizer options such as lr or weight_decay. Unmatched
                parameters retain the family defaults and semantic decay policy.
        """
        configs = {name: dict(config) for name, config in
                   (("muon", muon), ("sinkhorn", sinkhorn), ("adamw", adamw)) if config is not None}
        specs = {name: config.pop("param_groups", []) for name, config in configs.items()}
        if not any(specs.values()):
            self.optimizer = get_hyper_optimizer(model, **configs)
            return
        groups = _build_optimizer_groups(model, configs)
        arguments = {}
        for name, config in configs.items():
            arguments[f"{name}_params"] = _resolve_param_groups(model, groups[name], specs[name])
            arguments[f"{name}_kwargs"] = {key: value for key, value in config.items() if key != "param_patterns"}
        self.optimizer = get_hyper_optimizer(model, **arguments)

    def get_optimizer(self) -> Any:
        """Return the composed core optimizer expected by Trainer."""
        return self.optimizer


class Sinkhorn:
    """Build Sinkhorn for embeddings/output weights with an AdamW fallback."""

    def __init__(
            self,
            sinkhorn_config: Dict[str, Any],
            adamw_config: Dict[str, Any],
            model: nn.Module,
    ) -> None:
        """Construct a two-family runtime using the shared semantic routing rules.

        Args:
            sinkhorn_config: Sinkhorn options, optionally prefixed with ``sinkhorn_``.
                ``param_patterns`` explicitly selects additional matrix parameters.
            adamw_config: Fallback options, optionally prefixed with ``adamw_``.
            model: Module whose trainable parameters are optimized. Embeddings and
                output weights default to Sinkhorn; remaining parameters use AdamW.
        """
        sinkhorn = {key.removeprefix("sinkhorn_"): value for key, value in sinkhorn_config.items()}
        adamw = {key.removeprefix("adamw_"): value for key, value in adamw_config.items()}
        self.optimizer = get_hyper_optimizer(model, sinkhorn=sinkhorn, adamw=adamw)

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer
