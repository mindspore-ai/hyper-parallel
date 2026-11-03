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
"""Configure DeepSeek-V4.1-specific activation-checkpoint replay invariants."""

from typing import Any

from hyper_parallel.core.activation_memory.wrapper import (
    CheckpointExcludeWrapper,
    CheckpointWrapper,
)
from hyper_parallel.distributed.encoder_parallel.attention import encoder_attention_checkpoint_contexts
from hyper_parallel.models.deepseek_v41.adapter.policies.shared_state import configure_shared_state
from hyper_parallel.models.deepseek_v41.vision import DeepseekV41VisionTower


def _recomputed_modules(module: Any) -> tuple[Any, ...]:
    """Collect one replay subtree without crossing checkpoint-exclude boundaries."""
    modules = [module]
    for child in module.children():
        if isinstance(child, CheckpointExcludeWrapper):
            continue
        modules.extend(_recomputed_modules(child))
    return tuple(modules)


def configure_deepseek_v41_activation_checkpointing(model: Any) -> None:
    """Install model-specific replay protections after generic wrappers are applied."""
    checkpoint_wrappers = tuple(
        module for module in model.modules() if isinstance(module, CheckpointWrapper)
    )
    recompute_modules = tuple(
        module
        for wrapper in checkpoint_wrappers
        for module in _recomputed_modules(wrapper.get_submodule("_wrapped_module"))
    )
    configure_shared_state(
        model,
        roles=("full", "reindex"),
        recompute_modules=recompute_modules,
    )

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
