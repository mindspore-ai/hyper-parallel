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
"""Reserved Mega member; fusion belongs to the core executor, not EPContext."""

from __future__ import annotations

from torch import Tensor

from hyper_parallel.components.modules.moe.grouped_experts.grouped_experts import GroupedExpertsModule


class MegaGroupedExperts(GroupedExpertsModule):
    """Reserve one fused routed execution over the canonical parameters.

    A future non-Module runtime holder will own a create_parameters=False core
    executor. Each call must pass current unsharded weights after FSDP hooks;
    no copied expert parameters or cross-step weight pointers are retained.
    """

    def forward(
        self, hidden_states: Tensor, topk_ids: Tensor | None = None, topk_weights: Tensor | None = None,
        **kwargs: Tensor,
    ) -> Tensor:
        """Reserve fused transport/compute/restore without first running ordinary stages.

        Args:
            hidden_states: Original source tokens [...,H], before ordinary slot preparation.
            topk_ids: Global model-selected expert IDs [T,K].
            topk_weights: Final differentiable routing weights [T,K].
            **kwargs: Source aliases accepted by the family interface.
        """
        raise NotImplementedError("Mega executor binding, probability contract and lifecycle are not implemented")
