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
"""Qwen3-VL-MoE parameter sharding roles not covered by generic naming rules."""

from typing import Any


def build_parameter_sharding_rules() -> list[tuple[list[str] | str, Any]]:
    """Return roles for the vision tower, which no default naming rule covers.

    The vision tower (``model.visual.*``) is small and replicated across
    devices by convention in VL fine-tuning; declaring REPLICATED makes its
    gradient-sync semantics explicit so the plan-time coverage check
    (HP-PLAN-004) passes for standard configurations. A user who freezes the
    tower (``requires_grad=False``) exits the coverage check automatically,
    so both trainable and frozen scenarios stay correct.
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        ("model.visual.", ParamRole.REPLICATED),
    ]


__all__ = ["build_parameter_sharding_rules"]
