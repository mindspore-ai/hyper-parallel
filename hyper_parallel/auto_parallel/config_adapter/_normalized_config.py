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
"""Shared type definitions for auto parallel strategy search configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from hyper_parallel.auto_parallel._model_spec import ModelSpec, model_fields


@dataclass
class NormalizedConfig:
    """Aggregate container for a parallel strategy search task.

    Holds all configuration sections as plain dicts for maximum
    compatibility with PR631's Config class (sapp_nd.nd.common.config).
    ``model_spec`` holds the model IR in its mapping form: the fields of
    :class:`~hyper_parallel.auto_parallel._model_spec.ModelSpec`, which owns
    the list of them and what each one requires. Call :meth:`model` for the
    typed, validated object. Keys the IR does not declare are carried
    through untouched for the adapter's own use, such as the micro-batch
    size; how the run loads, computes and optimizes the model is ``run``'s.

    Args:
        model_spec: Model architecture parameters, in the shape
            :class:`ModelSpec` defines.
        cluster_spec: Hardware cluster description.
        search_space: Parallel dimension candidate values, e.g.
            ``{"dp": [1,2,4], "tp": [1,2,4,8], "pp": [1,2], "cp": [1], "ep": [1]}``.
        constraint: User-imposed constraints (global_batch_size,
            memory_limit_gb, fixed_*_degree).
        estimator: Estimation algorithm parameters.  Optional key:
            ``cp_algo`` (``"colossalai_cp"`` | ``"ulysses_cp"`` | ``"hybrid_cp"``,
            default ``"colossalai_cp"``).
        pp_config: Pipeline-parallel specific configuration.
        run: What the train.yaml states about the run, in its own sections
            and names: the model's dtypes, FSDP's precision and resharding,
            the optimizer, gradient clipping and the accelerator's settings.
            The search runner lays the strategy it searches over it, so the
            rest reaches the cost model as the train.yaml states it.
        parallelism_summary: One line naming what every parallelism
            dimension resolved to, so a dimension that was silently pinned
            or silently freed is visible.
        resolved_strategy: Final resolved strategy, populated after search.
    """

    model_spec: Dict[str, Any] = field(default_factory=dict)
    cluster_spec: Dict[str, Any] = field(default_factory=dict)
    search_space: Dict[str, List[int]] = field(default_factory=dict)
    constraint: Dict[str, Any] = field(default_factory=dict)
    estimator: Dict[str, Any] = field(default_factory=dict)
    pp_config: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    parallelism_summary: str = ""
    resolved_strategy: Optional[Dict[str, Any]] = None

    def model(self) -> ModelSpec:
        """Return ``model_spec`` as the typed, validated model IR.

        Returns:
            The parsed spec.

        Raises:
            ModelSpecError: If a required field is missing or the declared
                fields contradict each other.
        """
        return ModelSpec.from_dict(model_fields(self.model_spec))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all config sections to a nested dictionary."""
        result: Dict[str, Any] = {
            "model_spec": dict(self.model_spec),
            "cluster_spec": dict(self.cluster_spec),
            "search_space": dict(self.search_space),
            "constraint": dict(self.constraint),
            "estimator": dict(self.estimator),
            "pp_config": dict(self.pp_config),
            "run": dict(self.run),
        }
        if self.resolved_strategy is not None:
            result["resolved_strategy"] = dict(self.resolved_strategy)
        return result


@dataclass
class ValidationError:
    """A single validation error or warning discovered during config validation.

    Args:
        field_path: Dot-separated path to the offending field
            (e.g. ``"model_spec.dim"``).
        message: Human-readable description of the problem.
        severity: Error severity level (``"error"`` or ``"warning"``).
    """

    field_path: str
    message: str
    severity: Literal["error", "warning"] = "error"


ValidationSeverity = Literal["error", "warning"]
