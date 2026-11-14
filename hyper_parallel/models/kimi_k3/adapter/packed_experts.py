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
"""K3 SiTU binding for the common expert-slot implementation."""

from functools import partial

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.functional.activation_situ import situ_packed
from hyper_parallel.components.modules.expert_compute import grouped_expert_slots as compute_expert_slots


def grouped_expert_slots(hidden_states: torch.Tensor, local_expert_indices: torch.Tensor,
                         weights: tuple[torch.Tensor, ...], *, beta: float, linear_beta: float,
                         situ_backend: str, permute_backend: str = "torch", recompute: str = "none") -> torch.Tensor:
    """Bind the model's activation without owning dispatch or expert parameters.

    Args:
        hidden_states: Dispatched expert-space slots.
        local_expert_indices: Local expert ID for each slot.
        weights: Complete local w1/w2/w3 or w13/w2 matrices.
        beta: Gate saturation scale.
        linear_beta: Up-branch saturation scale.
        situ_backend: Explicit SiTU backend.
        permute_backend: Native or NPU permutation.
        recompute: Local expert replay scope.

    Returns:
        Unweighted expert outputs in incoming slot order and dtype.
    """
    activation = partial(situ_packed, beta=beta, linear_beta=linear_beta, backend=situ_backend)
    return compute_expert_slots(hidden_states, local_expert_indices, weights, activation=activation,
                                permute_backend=permute_backend, recompute=recompute)
