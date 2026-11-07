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
"""Bind Xing40 sequencewise router auxiliary statistics to their CP group."""

from typing import Any

from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.xing40.modeling_xing40 import Xing40MoE


@inner_wrapper
def xing40_moe_cp_wrapper(target_module: Xing40MoE, mesh: Any, tp_mesh: Any,
                            cp_mesh: Any, ep_mesh: Any) -> list:
    """Bind the CP group while preserving the MoE forward.

    Args:
        target_module: Xing40 MoE whose sequence statistics need the CP group.
        mesh: Unused full mesh.
        tp_mesh: Tensor-parallel mesh or None.
        cp_mesh: Two-rank context-parallel mesh.
        ep_mesh: Expert-parallel mesh or None.

    Returns:
        An empty list because no forward rewrite is needed.
    """
    del mesh
    if not isinstance(target_module, Xing40MoE):
        raise TypeError('Xing40 MoE required')
    if (cp_mesh is None or cp_mesh.size() != 2 or (tp_mesh is not None and tp_mesh.size() != 1)
            or (ep_mesh is not None and ep_mesh.size() != 1)):
        raise ValueError('Xing40 initial CP adapter requires CP2, TP1 and EP1')
    if target_module.aux_loss_cp_group is not None:
        raise ValueError('Xing40 MoE CP group already configured')
    target_module.aux_loss_cp_group = cp_mesh.get_group()
    return []
