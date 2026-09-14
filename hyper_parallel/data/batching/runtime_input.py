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
"""Model-owned runtime input extensions for generic DataLoader batches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeInputContext:
    """Framework execution facts available to a model runtime adapter.

    ``options`` deliberately carries feature-specific policy. The framework
    owns batch movement and parallel coordinates, while a model adapter may
    translate those facts into attention metadata, modality routing inputs,
    cache descriptors, or another forward-only contract.
    """

    source_type: str
    local_input_shape: Sequence[int]
    parallel_ranks: Mapping[str, int]
    parallel_sizes: Mapping[str, int]
    options: Mapping[str, Any]


class RuntimeInputAdapter(ABC):
    """Extend a generic batch with model-owned forward inputs."""

    @abstractmethod
    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Build additional model inputs without mutating ``batch``.

        Args:
            batch: Device-resident, parallel-local batch. Global metadata such
                as ``cu_seq_lens`` remains available when required.
            context: Source, local geometry, parallel coordinates, and
                feature-specific recipe options.

        Returns:
            Mapping merged into the model forward inputs. Keys must not replace
            fields already owned by the generic batch path.
        """
        raise NotImplementedError


__all__ = ["RuntimeInputAdapter", "RuntimeInputContext"]
