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
"""Compatibility entry for the shared quantile-balancing implementation."""

from collections.abc import Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.quantile_balancing import QuantileBalancer as LocalQuantileBalancer
from hyper_parallel.distributed.expert_parallel.balancing import finalize_quantile_balancers


class QuantileBalancer(LocalQuantileBalancer):
    """Preserve the historical distributed finalize_step entry point."""

    @torch.no_grad()
    def finalize_step(self, correction_bias: torch.Tensor, process_groups: Sequence[Any] = ()) -> bool:
        """Commit through the public distributed reducer.

        Args:
            correction_bias: Router-owned FP32 bias.
            process_groups: Orthogonal groups containing distinct tokens.

        Returns:
            Whether the pooled batch updated the bias.
        """
        result = finalize_quantile_balancers(((self, correction_bias),), process_groups=process_groups)
        return bool(result[0].item())


__all__ = ["QuantileBalancer", "finalize_quantile_balancers"]
