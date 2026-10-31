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
"""Declare Xing4.0 MLA parameter sharding roles."""

from typing import Any


def build_parameter_sharding_rules() -> list[tuple[list[str], Any]]:
    """Return the attention parameter roles that generic naming rules do not infer.

    The mHC, stream-collapse head and MTP fusion parameters are declared replicated by the
    recipe's ``plan_overrides``.

    Returns:
        ``(name patterns, role)`` pairs checked before the generic naming rules.
    """
    # Lazy: discovery imports this module without the distributed runtime.
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=import-outside-toplevel
        ParamRole,
    )

    return [
        (["linear_qkv"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
    ]


__all__ = ["build_parameter_sharding_rules"]
