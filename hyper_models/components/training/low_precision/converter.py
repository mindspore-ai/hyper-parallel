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
"""Build-time conversion of Dense Linear modules to A5 MXFP8."""

import fnmatch
import logging
from typing import Optional, Type

from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_models.components.training.low_precision.config import LowPrecisionConfig
from hyper_models.components.training.low_precision.modules import NpuQuantLinear

logger = logging.getLogger(__name__)


class LowPrecisionConversionError(ValueError):
    """Report that the requested Dense conversion cannot be applied."""

_ROUTED_EXPERT_CONTAINER_NAMES = frozenset(
    {
        "experts",
        "local_experts",
        "grouped_experts",
    }
)
_SHARED_EXPERT_CONTAINER_NAMES = frozenset(
    {
        "shared_expert",
        "shared_experts",
    }
)


def _matches(fqn: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(fqn, pattern) for pattern in patterns)


def _subtree_selected(
    module: nn.Module,
    fqn: str,
    include_fqns: list[str],
) -> bool:
    """Check actual module paths instead of approximating glob prefixes."""

    for relative_fqn, _ in module.named_modules(remove_duplicate=False):
        candidate_fqn = f"{fqn}.{relative_fqn}" if relative_fqn else fqn
        if _matches(candidate_fqn, include_fqns):
            return True
    return False


def _is_routed_expert_container(name: str, module: nn.Module) -> bool:
    normalized_name = name.lower()
    if normalized_name in _SHARED_EXPERT_CONTAINER_NAMES:
        return False
    class_name = type(module).__name__.lower()
    return (
        normalized_name in _ROUTED_EXPERT_CONTAINER_NAMES
        or normalized_name.endswith("_experts")
        or "groupedexperts" in class_name
        or "grouped_experts" in class_name
    )


def _is_aligned(linear: nn.Linear) -> bool:
    return linear.in_features % 32 == 0 and linear.out_features % 32 == 0


def _peft_base_tuner_type() -> Optional[Type[nn.Module]]:
    """Resolve PEFT lazily so the optional package is not a hard dependency."""

    try:
        from peft.tuners.tuners_utils import (  # pylint: disable=import-outside-toplevel
            BaseTunerLayer,
        )
    except ImportError:
        return None
    return BaseTunerLayer


def apply_low_precision(
    model: nn.Module,
    config: LowPrecisionConfig,
) -> nn.Module:
    """Convert selected Dense Linear modules before sharding.

    The traversal does not read tensor data or import torch_npu, so it is safe
    for meta tensors. Routed-expert containers are left to Sharding Plan/Apply.
    PEFT wrappers are selected by their original logical FQN and only their
    exact ``base_layer`` Linear is replaced.

    Args:
        model: Model to convert in place.
        config: Validated low-precision configuration.

    Returns:
        The original model object after in-place conversion.

    Raises:
        LowPrecisionConversionError: If the configuration selects no supported
            target or a selected target cannot be converted.
    """

    if not config.enabled:
        return model

    peft_base_type = _peft_base_tuner_type()
    include_coverage = {pattern: 0 for pattern in config.include_fqns}
    converted_fqns: list[str] = []
    skipped_targets: dict[str, str] = {}
    replacements: list[tuple[nn.Module, str, nn.Linear, str]] = []

    def _record_candidate(logical_fqn: str) -> None:
        converted_fqns.append(logical_fqn)
        for pattern in include_coverage:
            if fnmatch.fnmatchcase(logical_fqn, pattern):
                include_coverage[pattern] += 1

    def _plan_linear_replacement(
        parent: nn.Module,
        child_name: str,
        linear: nn.Linear,
        *,
        logical_fqn: str,
    ) -> None:
        if not _is_aligned(linear):
            skipped_targets[logical_fqn] = "weight-shape-not-32-aligned"
            return
        replacements.append((parent, child_name, linear, logical_fqn))
        _record_candidate(logical_fqn)

    def _convert_children(parent: nn.Module, parent_fqn: str = "") -> None:
        # named_children() removes duplicate Module objects. Reading _modules
        # is intentional here so every registered FQN is evaluated.
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            child_fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
            if _matches(child_fqn, config.exclude_fqns):
                continue
            if _is_routed_expert_container(child_name, child):
                if config.include_fqns and _subtree_selected(
                    child,
                    child_fqn,
                    config.include_fqns,
                ):
                    skipped_targets[child_fqn] = (
                        "routed-experts-require-moe-plan"
                    )
                continue
            if peft_base_type is not None and isinstance(child, peft_base_type):
                if config.include_fqns and not _matches(
                    child_fqn,
                    config.include_fqns,
                ):
                    continue
                base_layer = getattr(child, "base_layer", None)
                if type(base_layer) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
                    skipped_targets[child_fqn] = (
                        "peft-base-layer-is-not-exact-nn-linear"
                    )
                    continue
                _plan_linear_replacement(
                    child,
                    "base_layer",
                    base_layer,
                    logical_fqn=child_fqn,
                )
                continue
            if type(child) is nn.Linear:  # pylint: disable=unidiomatic-typecheck
                if config.include_fqns and not _matches(
                    child_fqn,
                    config.include_fqns,
                ):
                    continue
                _plan_linear_replacement(
                    parent,
                    child_name,
                    child,
                    logical_fqn=child_fqn,
                )
                continue
            if isinstance(child, nn.Linear):
                if not config.include_fqns or _matches(
                    child_fqn,
                    config.include_fqns,
                ):
                    skipped_targets[child_fqn] = (
                        "linear-subclass-is-not-supported"
                    )
                continue
            _convert_children(child, child_fqn)

    _convert_children(model)
    if skipped_targets:
        details = ", ".join(
            f"{fqn}: {reason}"
            for fqn, reason in sorted(skipped_targets.items())
        )
        raise LowPrecisionConversionError(
            f"Low-precision conversion is incomplete: {details}."
        )
    unmatched_patterns = [
        pattern for pattern, count in include_coverage.items() if count == 0
    ]
    if unmatched_patterns:
        raise LowPrecisionConversionError(
            "Low-precision include_fqns matched no supported target: "
            f"{', '.join(sorted(unmatched_patterns))}."
        )
    if not converted_fqns:
        raise LowPrecisionConversionError(
            "Low-precision conversion selected no supported target. "
            "Only exact nn.Linear modules and PEFT wrapper base_layer Linear "
            "modules are supported by the Dense converter."
        )

    # Prepare every replacement before mutating the model. Reuse one converted
    # shell for aliases of the same Linear so module and Parameter sharing are
    # both preserved.
    converted_by_source: dict[int, NpuQuantLinear] = {}
    prepared_replacements: list[tuple[nn.Module, str, NpuQuantLinear]] = []
    for parent, child_name, linear, logical_fqn in replacements:
        source_id = id(linear)
        converted = converted_by_source.get(source_id)
        if converted is None:
            converted = NpuQuantLinear.from_linear(
                linear,
                fqn=logical_fqn,
            )
            converted_by_source[source_id] = converted
        prepared_replacements.append((parent, child_name, converted))
    for parent, child_name, converted in prepared_replacements:
        setattr(parent, child_name, converted)

    logger.info(
        "Low-precision conversion succeeded: converted=%d",
        len(converted_fqns),
    )
    logger.debug("Converted low-precision targets: %s", converted_fqns)
    return model
