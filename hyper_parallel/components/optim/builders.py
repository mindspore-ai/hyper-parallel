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
"""YAML-targeted optimizer builders: AdamW/Muon selection.

Parameter-name and parameter-group logic lives in
:mod:`hyper_parallel.components.optim.parameter_groups`; the optimizer
algorithm implementations stay in ``hyper_parallel.core.optimizer``.
"""

__all__ = ["AdamW", "Muon", "Sinkhorn"]

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer import get_hyper_optimizer
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
            excluded_param_ids: Optional[Sequence[int]] = None,
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
            excluded_param_ids=excluded_param_ids,
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
            excluded_param_ids: Optional[Sequence[int]] = None,
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
            excluded_param_ids=excluded_param_ids,
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
            excluded_param_ids: Optional[Sequence[int]] = None,
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
            excluded_param_ids=excluded_param_ids,
        )
        if not muon_params:
            raise ValueError("Muon requires at least one eligible matrix parameter")

        adamw_groups, _ = AdamW.get_adamw_param_groups(
            model,
            weight_decay=adamw_config.get("adamw_weight_decay", 1e-2),
            no_decay_params=no_decay_params,
            allowed_param_ids=[id(parameter) for parameter in adamw_params],
            excluded_param_ids=excluded_param_ids,
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
            excluded_param_ids: Optional[Sequence[int]] = None,
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
            excluded_param_ids=excluded_param_ids,
        )

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer


class Sinkhorn:
    """Build Sinkhorn for embeddings/heads, Muon for matrices, and AdamW otherwise."""

    def __init__(
            self,
            sinkhorn_config: dict,
            muon_config: dict,
            adamw_config: dict,
            model: nn.Module,
            no_decay_params: Optional[List[str]] = None,
            excluded_param_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Partition trainable parameters and build the existing chained runtime.

        Args:
            sinkhorn_config: Sinkhorn kwargs, optional ``param_patterns`` and
                ``param_groups`` containing selectors and group overrides.
            muon_config: Muon kwargs, including model-owned callbacks.
            adamw_config: AdamW kwargs and optional ``param_patterns``.
            model: Model whose parameters are partitioned by identity.
            no_decay_params: Name keywords exempt from AdamW decay. Defaults
                to bias and scale parameters; normalization weights decay.
            excluded_param_ids: Parameters managed by another optimizer.
        """
        configs = {"sinkhorn": dict(sinkhorn_config), "muon": dict(muon_config), "adamw": dict(adamw_config)}
        selectors = {name: self._patterns(config.pop("param_patterns", ())) for name, config in configs.items()}
        override_patterns, sinkhorn_groups = self._sinkhorn_overrides(configs["sinkhorn"].pop("param_groups", ()))
        aliases = self._aliases(model, excluded_param_ids)
        if not aliases:
            raise ValueError("Sinkhorn builder requires at least one trainable parameter")
        roles = self._roles(model)
        selected = {name: [] for name in configs}
        for param, names in aliases.items():
            family, group_index = self._route(param, names, roles, selectors, override_patterns)
            selected[family].append(param)
            if family == "sinkhorn":
                sinkhorn_groups[group_index]["params"].append(param)
        adamw_groups = self._adamw_groups(selected["adamw"], aliases, roles[1], no_decay_params)
        logger.info_rank0("Optimizer split: %s", {name: len(params) for name, params in selected.items()})
        self.optimizer = get_hyper_optimizer(
            model=model,
            muon_params=selected["muon"],
            adamw_params=adamw_groups,
            sinkhorn_params=[group for group in sinkhorn_groups if group["params"]],
            muon_kwargs=configs["muon"],
            adamw_kwargs=configs["adamw"],
            sinkhorn_kwargs=configs["sinkhorn"],
        )

    @classmethod
    def _sinkhorn_overrides(cls, overrides):
        """Separate pattern selectors from Sinkhorn parameter-group overrides."""
        patterns = []
        groups = [{"params": []}]
        for group in overrides:
            if "params" in group:
                raise ValueError("Sinkhorn builder groups use param_patterns, not params")
            selector = cls._patterns(group.get("param_patterns", ()))
            if not selector:
                raise ValueError("Sinkhorn parameter group overrides require param_patterns")
            patterns.append(selector)
            groups.append({**{key: value for key, value in group.items() if key != "param_patterns"}, "params": []})
        return patterns, groups

    @classmethod
    def _route(cls, param, names, roles, selectors, override_patterns):
        """Resolve one parameter's optimizer and reject conflicting selectors."""
        matches = [index for index, patterns in enumerate(override_patterns) if cls._matches(names, patterns)]
        if len(matches) > 1:
            raise ValueError(f"Overlapping Sinkhorn parameter groups for {names}")
        explicit = {name for name, patterns in selectors.items() if cls._matches(names, patterns)}
        if matches:
            explicit.add("sinkhorn")
        if len(explicit) > 1:
            raise ValueError(f"Overlapping optimizer selectors for {names}")
        family = next(iter(explicit)) if explicit else cls._default_family(param, names, *roles)
        if family == "sinkhorn" and param.ndim != 2:
            raise ValueError(f"Sinkhorn requires a 2D matrix: {names}")
        if family == "muon" and param.ndim < 2:
            raise ValueError(f"Muon requires a matrix: {names}")
        return family, matches[0] + 1 if matches else 0

    @staticmethod
    def _patterns(patterns):
        if isinstance(patterns, str):
            patterns = [patterns]
        return [re.compile(pattern) for pattern in patterns]

    @staticmethod
    def _matches(names, patterns):
        return any(pattern.search(name) for pattern in patterns for name in names)

    @staticmethod
    def _aliases(model, excluded_param_ids):
        excluded = set(excluded_param_ids or ())
        aliases = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            if param.requires_grad and id(param) not in excluded:
                aliases.setdefault(param, []).append(name)
        return aliases

    @staticmethod
    def _roles(model):
        embeddings, norms = set(), set()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                embeddings.add(id(module.weight))
            if re.search(r"norm(?:[123]d|gated)?$", type(module).__name__, re.IGNORECASE):
                norms.update(id(param) for param in module.parameters(recurse=False))
        return embeddings, norms

    @staticmethod
    def _default_family(param, names, embedding_ids, norm_ids):
        if id(param) in norm_ids:
            return "adamw"
        if id(param) in embedding_ids or any(
                re.search(r"(?:^|\.)(?:lm_head|out_head|output_head)\.weight$", name) for name in names):
            return "sinkhorn"
        return "muon" if param.ndim >= 2 else "adamw"

    @staticmethod
    def _adamw_groups(params, aliases, norm_ids, no_decay_params):
        """Split AdamW parameters by decay policy, including normalization weights."""
        decay, no_decay = [], []
        for param in params:
            names = aliases[param]
            if no_decay_params is None:
                excluded = (param.ndim < 2 and id(param) not in norm_ids) or any(
                    name.rsplit(".", 1)[-1] in ("bias", "scale", "scales", "scaling_factor") for name in names
                )
            else:
                excluded = any(keyword.lower() in name.lower() for keyword in no_decay_params for name in names)
            (no_decay if excluded else decay).append(param)
        groups = [{"params": decay}] if decay else []
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return groups

    def get_optimizer(self) -> Any:
        """Return the core chained optimizer runtime."""
        return self.optimizer
