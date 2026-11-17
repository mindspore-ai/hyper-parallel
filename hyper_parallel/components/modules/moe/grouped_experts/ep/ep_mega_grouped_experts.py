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
"""Reserved EP/Mega binding; no second EP transport or Mega resource manager."""

from hyper_parallel.components.modules.moe.grouped_experts.mega_grouped_experts import MegaGroupedExperts
from hyper_parallel.distributed.module_binding import ParallelBinding


class EPMegaGroupedExperts(MegaGroupedExperts):
    """Reserve the job-mesh entry for the same fused Mega execution point.

    The final protocol must convey expert ownership and an explicit EP group
    to the core runtime. It must not compose ordinary dispatch/combine around
    an executor that already performs those transfers.
    """

    def bind_parallel(self, binding: ParallelBinding) -> None:
        """Reserve borrowing the HP job mesh for the future fused runtime.

        Args:
            binding: Job-owned parallel resources; no expert parameters are transferred.
        """
        raise NotImplementedError("EP/Mega mesh binding and fused protocol are not implemented")
