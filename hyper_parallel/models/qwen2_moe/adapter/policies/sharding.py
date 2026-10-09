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
"""Declare Qwen2-MoE parameter sharding roles."""

from typing import Any


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Keep the scalar shared-expert gate replicated."""
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [(["shared_expert_gate"], ParamRole.REPLICATED)]


__all__ = ["build_parameter_sharding_rules"]
