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
"""Configure DeepSeek-V4.1 model-specific checkpoint replay invariants."""

from typing import Any

from hyper_parallel.core.activation_memory.wrapper import (
    CheckpointExcludeWrapper,
    CheckpointWrapper,
)
from hyper_parallel.distributed.encoder_parallel.attention import encoder_attention_checkpoint_contexts
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower


def _enable_indexer_selection_exclude(module: Any, configured: set[int]) -> None:
    """Pin indexer selection in one replayed subtree, pruning excluded regions."""
    if isinstance(module, CheckpointExcludeWrapper):
        return
    if getattr(module, "is_index_source", False):
        indexer = getattr(module, "indexer", None)
        enable = getattr(indexer, "enable_selection_exclude", None)
        if callable(enable) and id(indexer) not in configured:
            if getattr(indexer, "_selection_region", None) is None:
                enable()
            configured.add(id(indexer))
    for child in module.children():
        _enable_indexer_selection_exclude(child, configured)


def _configure_decoder_replay(model: Any) -> None:
    """Keep Full/Reindex discrete selection stable during decoder replay."""
    configured: set[int] = set()
    for module in model.modules():
        if isinstance(module, CheckpointWrapper):
            _enable_indexer_selection_exclude(module._wrapped_module, configured)  # pylint: disable=protected-access


def configure_deepseek_v41_activation_checkpointing(model: Any) -> None:
    """Install decoder and vision replay rules after generic wrappers are applied."""
    _configure_decoder_replay(model)
    for module in model.modules():
        if not isinstance(module, DeepseekV41VisionTower):
            continue
        for block in module.blocks:
            if isinstance(block, CheckpointWrapper):
                block.checkpoint_kwargs.update(
                    context_fn=encoder_attention_checkpoint_contexts,
                    # All ranks must replay every Ulysses/FSDP collective.
                    early_stop=False,
                )


__all__ = ["configure_deepseek_v41_activation_checkpointing"]
