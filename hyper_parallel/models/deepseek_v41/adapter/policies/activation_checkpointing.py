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
"""Configure context-preserving DeepSeek-V4.1 vision checkpoint replay."""

from typing import Any

from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed.encoder_parallel.attention import encoder_attention_checkpoint_contexts
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower


def configure_deepseek_v41_activation_checkpointing(model: Any) -> None:
    """Install vision replay context after generic wrappers are applied."""
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
