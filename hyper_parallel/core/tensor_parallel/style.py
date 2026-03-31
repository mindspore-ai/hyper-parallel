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
"""Abstract parallel style for declarative module parallelism."""
from abc import ABC, abstractmethod
from typing import Optional

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform

platform = get_platform()
Module = platform.Module


class ParallelStyle(ABC):
    """Abstract base class for parallel styles applied to nn.Module submodules.

    Subclasses implement ``apply`` to wrap a module with the desired
    parallel communication behaviour (e.g. all-to-all for context parallel).

    ``src_data_rank`` mirrors PyTorch's tensor-parallel contract: it can be set by
    :func:`parallelize_module` for styles that scatter/broadcast global tensors.
    HyperParallel styles may ignore it until they integrate ``distribute_tensor``.
    """

    src_data_rank: Optional[int] = 0

    @abstractmethod
    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Apply this parallel style to *module* in-place and return it.

        Args:
            module: The submodule to be parallelised.
            device_mesh: The device mesh describing the cluster topology.

        Returns:
            The (possibly wrapped) module with parallelism applied.
        """
