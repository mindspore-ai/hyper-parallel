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
"""Explicit autograd targets carried alongside ordinary forward values."""
from typing import Any, NamedTuple, Optional, Tuple


class BackwardTarget(NamedTuple):
    """One tensor root and its optional sensitivity for backward.

    Args:
        tensor: Tensor that acts as an autograd root.
        gradient: Optional sensitivity matching ``tensor``. A scalar tensor may
            omit it and use the platform's implicit ones-like sensitivity.
    """

    tensor: Any
    gradient: Optional[Any] = None


class AuxiliaryOutput(NamedTuple):
    """Ordinary forward value plus stage-local explicit backward targets.

    ``value`` is the only part sent across a pipeline boundary. Targets remain
    local to the stage that created their autograd graphs.

    Args:
        value: Ordinary model output consumed by the next layer or PP stage.
        backward_targets: Backward roots owned by the current model/stage.
    """

    value: Any
    backward_targets: Tuple[BackwardTarget, ...]


def split_backward_targets(output: Any) -> Tuple[Any, Tuple[BackwardTarget, ...]]:
    """Split an optional :class:`AuxiliaryOutput` without guessing tuple shape.

    Args:
        output: A regular forward value or :class:`AuxiliaryOutput`.

    Returns:
        Pair of the regular value and a tuple of explicit backward targets.
    """
    if isinstance(output, AuxiliaryOutput):
        return output.value, output.backward_targets
    return output, ()


def attach_backward_targets(value: Any, *target_groups: Tuple[BackwardTarget, ...]) -> Any:
    """Attach non-empty target groups to ``value`` while preserving order.

    Args:
        value: Ordinary forward value.
        *target_groups: Target tuples collected from child modules.

    Returns:
        ``value`` unchanged when no targets exist, otherwise an
        :class:`AuxiliaryOutput` containing every target in declaration order.
    """
    targets = tuple(target for group in target_groups for target in group)
    if not targets:
        return value
    return AuxiliaryOutput(value, targets)


def backward_targets(targets: Tuple[BackwardTarget, ...]) -> None:
    """Execute one multi-root backward through the active platform backend.

    Args:
        targets: Non-empty tuple of explicit backward targets.

    Raises:
        ValueError: If no targets are provided.
    """
    if not targets:
        raise ValueError("backward_targets requires at least one BackwardTarget.")
    # Lazy import avoids a cycle while a backend constructs its PipelineStageBase.
    from hyper_parallel.platform import get_platform  # pylint: disable=C0415
    tensors = tuple(target.tensor for target in targets)
    gradients = tuple(target.gradient for target in targets)
    get_platform().backward(tensors, gradients)
