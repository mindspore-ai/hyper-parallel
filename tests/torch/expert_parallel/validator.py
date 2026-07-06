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
"""Test-only validation utilities for mesh and MoE compatibility checks."""

from typing import List
from tests.torch.expert_parallel.templates import get_template


def validate_mesh_dimensions(
    dp: int, ep: int, tp: int, cp: int, world_size: int
) -> None:
    """Validate mesh dimensions against total devices."""
    errors: List[str] = []

    if dp <= 0:
        errors.append(f"dp must be positive, got {dp}")
    if ep <= 0:
        errors.append(f"ep must be positive, got {ep}")
    if tp <= 0:
        errors.append(f"tp must be positive, got {tp}")
    if cp <= 0:
        errors.append(f"cp must be positive, got {cp}")

    product = dp * ep * tp * cp
    if product != world_size:
        errors.append(
            f"dp*ep*tp*cp = {dp}*{ep}*{tp}*{cp} = {product} != world_size={world_size}\n"
            f"Fix suggestion: adjust world_size to {product} or modify dimensions"
        )

    if ep > 0 and dp > 1 and dp % ep != 0:
        errors.append(
            f"dp={dp} must be divisible by ep={ep} when dp > 1\n"
            f"Fix suggestion: set dp to a multiple of ep (e.g., {ep}, {ep*2})"
        )

    if errors:
        raise ValueError("Mesh configuration invalid:\n" + "\n".join(errors))


def validate_moe_compatibility(
    num_experts: int, ep: int, hidden_dim: int, tp: int
) -> None:
    """Validate MoE model parameters against parallelism degrees."""
    errors: List[str] = []

    if ep > 0 and num_experts % ep != 0:
        errors.append(
            f"num_experts={num_experts} must be divisible by ep={ep}\n"
            f"Fix suggestion: set num_experts to a multiple of ep"
        )

    if tp > 0 and hidden_dim % tp != 0:
        errors.append(
            f"hidden_dim={hidden_dim} must be divisible by tp={tp}\n"
            f"Fix suggestion: set hidden_dim to a multiple of tp"
        )

    if errors:
        raise ValueError("MoE model compatibility check failed:\n" + "\n".join(errors))


def validate_template(
    template_name: str,
    world_size: int,
    num_experts: int,
    hidden_dim: int,
    **overrides: int,
) -> None:
    """Validate a full template configuration."""
    config = get_template(template_name, **overrides)
    dp, ep, tp, cp = config["dp"], config["ep"], config["tp"], config["cp"]
    validate_mesh_dimensions(dp, ep, tp, cp, world_size)
    validate_moe_compatibility(num_experts, ep, hidden_dim, tp)
