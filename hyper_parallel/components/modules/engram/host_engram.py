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
"""Host-table Engram; planner, sparse optimizer and checkpoint policy stay caller-owned."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_engram import EngramModule
from .host_engram_table import HostEngramTable
from .parallel_engram import host_sparse_lookup


class HostEngramModule(EngramModule):
    """Reuse fusion math and stage only touched rows from an FP32 CPU table."""

    _sparse_lookup = staticmethod(host_sparse_lookup)

    def _create_embedding(self, rows: int, width: int) -> nn.Embedding:
        """Defer the full FP32 table to the caller-owned materialization lifecycle."""
        return nn.Embedding(rows, width, device="meta", dtype=torch.float32)


    def __init__(self, *, config: Any = None, module: nn.Module | None = None,
                 module_fqn: str = "", context: Any = None) -> None:
        """Create a meta table or adopt the source table before planner materialization."""
        super().__init__(config=config, module=module, module_fqn=module_fqn, context=context)
        if isinstance(self.embed, HostEngramTable):
            return
        training = self.embed.training
        source = module if module is not None else config
        weight = self.embed.weight
        if tuple(weight.shape) != (self.padded_num_embeddings, self.embed.embedding_dim):
            raise ValueError("Host Engram replacement requires the full physical table before planning")
        if module is None:
            weight = nn.Parameter(torch.empty(weight.shape, device="meta", dtype=torch.float32))
        self.embed = HostEngramTable(
            source_weight=weight, logical_rows=self.logical_num_embeddings,
            max_pending_entries=getattr(source, "engram_max_pending_entries", 1000000),
            max_sparse_rows_per_step=getattr(source, "engram_max_sparse_rows_per_step", 1000000),
        )
        self.embed.train(training)
