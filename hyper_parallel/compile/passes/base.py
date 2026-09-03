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
"""
Graph Pass Base - Base Class for Graph Transformation Passes

All parallel partitioning passes and optimization passes inherit from this
class. The config a pass receives is typed as ``PassConfig`` so passes
can rely on ``enable_overlap`` / ``fsdp_enabled`` / ``fsdp_degree``
/ ``tp_size`` being present without each pass re-declaring the contract.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..parallel_config import PassConfig

if TYPE_CHECKING:
    from torch import fx
    from ..sharding_config import PassPlan


class GraphPass(ABC):
    """Graph Optimization Pass Base Class.

    All parallel partitioning passes and optimization passes inherit from
    this class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Pass name."""
        pass  # pylint: disable=W0107

    @abstractmethod
    def run(
        self,
        graph_module: "fx.GraphModule",
        pass_config: PassConfig,
        **kwargs: Any,
    ) -> "fx.GraphModule":
        """Execute graph transformation.

        Args:
            graph_module: Input graph.
            pass_config: Parallel configuration.

        Returns:
            Transformed graph.
        """
        pass  # pylint: disable=W0107

    def __repr__(self) -> str:
        """Return a developer-facing representation of the pass."""
        return f"{self.__class__.__name__}(name={self.name})"


__all__ = ["GraphPass"]
