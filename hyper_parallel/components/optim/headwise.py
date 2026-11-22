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
"""Model-independent head views for Muon's logical Newton-Schulz matrices."""

from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from functools import partial

import torch


def _reshape_heads(
    param_fqn: str,
    ns_input: torch.Tensor,
    *,
    rules: tuple[tuple[str, int], ...],
) -> list[torch.Tensor]:
    """Return one storage-sharing view for a declared logical parameter."""
    widths = {width for pattern, width in rules if fnmatchcase(param_fqn, pattern)}
    if not widths:
        return [ns_input]
    if len(widths) != 1:
        raise ValueError(f"Conflicting head dimensions for parameter {param_fqn!r}")
    width = next(iter(widths))
    if ns_input.ndim != 2 or ns_input.shape[0] % width:
        raise ValueError(f"Head-wise Muon requires a 2D matrix with rows divisible by {width}: {param_fqn}")
    return [ns_input.view(ns_input.shape[0] // width, width, ns_input.shape[1])]


def build_reshape_fn(head_dims: Mapping[str, int]) -> Callable[[str, torch.Tensor], list[torch.Tensor]]:
    """Build Muon's reshape hook from parameter patterns and logical head widths.

    Args:
        head_dims: Case-sensitive FQN glob to row-axis head width. Matching rules
            must agree on the width. Patterns and widths are copied at construction.
            Nonmatching parameters retain their original matrix. The caller owns
            semantic parameter selection and must provide the complete NS matrix,
            rather than a shard cutting through a logical head.

    Returns:
        A Muon ``reshape_fn`` preserving dtype, device and working tensor storage.

    Raises:
        ValueError: Patterns or head dimensions are invalid. The returned hook
            also rejects conflicting matches, nonmatrices and nonintegral heads.
    """
    if not isinstance(head_dims, Mapping):
        raise ValueError("head_dims must map parameter patterns to positive head dimensions")
    rules = []
    for pattern, width in head_dims.items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Head-wise Muon parameter patterns must be nonempty strings")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("Head dimensions must be positive integers")
        rules.append((pattern, width))
    return partial(_reshape_heads, rules=tuple(rules))
