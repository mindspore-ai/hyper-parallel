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
"""Structure-preserving DeepSeek Engram replacement with a Host table."""
# pylint: disable=forbidden-backend-import,unsupported-binary-operation

from collections.abc import Mapping
from typing import Any

from torch import nn

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.models.deepseek_v41.adapter.engram.host_table import HostEngramTable
from hyper_parallel.models.replacement import module_replacement


@module_replacement
class HostEngramModule(EngramModule):
    """Reuse dense Engram components and preserve the full meta table FQN."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Preserve all source parameters while replacing only the table module."""
        embed_training = module.embed.training
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        del context
        self.embed = HostEngramTable(
            source_weight=module.embed.weight,
            logical_rows=self.logical_num_embeddings,
            physical_rows=self.padded_num_embeddings,
            width=module.embed.embedding_dim,
            max_pending_entries=module.engram_max_pending_entries,
            max_sparse_rows_per_step=module.engram_max_sparse_rows_per_step,
        )
        self.embed.train(embed_training)
