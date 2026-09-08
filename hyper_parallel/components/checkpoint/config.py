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
"""CheckpointingConfig typed config for checkpoint save/load.

Following design doc 04_checkpoint.md.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CheckpointingConfig:
    """Checkpointing typed config.

    Following design doc 04_checkpoint.md.

    ``model_save_format``, ``save_consolidated``, ``staging_dir`` and
    ``best_metric_key`` are declared by the design doc but not yet consumed by
    :class:`~hyper_parallel.trainer.callbacks.checkpoint_callback.CheckpointerCallback`;
    they are reserved for the consolidated HuggingFace-format export.
    """

    # Master switch for the *write* path only. Restoring is governed solely by
    # ``restore_from``, so ``save_ckpt=false`` with a ``restore_from`` set means
    # "start from this checkpoint but do not write any more" rather than
    # silently doing nothing.
    save_ckpt: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_steps: int = 0  # save every N optimizer steps (0 = disabled)
    save_epochs: int = 1  # save every N epochs
    is_async: bool = False
    is_peft: bool = False  # persist trainable (adapter) weights only
    save_optimizer: bool = True
    save_train_state: bool = True
    # Where the per-step training state (step/epoch, lr_scheduler, dataloader
    # position, RNG) is written:
    #   True  --- one torch.save file per rank under ``extra_state/``. Always
    #             correct, and the only safe choice when any of that state
    #             differs across ranks.
    #   False --- embedded in the DCP state dict. Fewer files, but DCP
    #             deduplicates entries that share an FQN across ranks, so every
    #             rank restores whichever rank's copy won. Only use this when the
    #             extra state is rank-invariant (deterministic training, uniform
    #             input shapes).
    # The load side auto-detects which layout a checkpoint used, so this flag can
    # be flipped between runs.
    save_extra_state_per_rank: bool = True

    restore_from: Optional[str] = None  # "LATEST" or specific path
    restore_optimizer: bool = True
    restore_train_state: bool = True  # step/epoch, lr_scheduler, dataloader, RNG

    model_save_format: str = "safetensors"
    # Whether to additionally export merged HF weights: "none" (never) |
    # "final" (only at the end of training) | "every" (on every save).
    # The disabled setting is uniformly "none" (a YAML-safe Literal value),
    # replacing the old design's "false" enum value ("false" would be parsed
    # as a bool by PyYAML; see the decision in 04_checkpoint.md section 4.2).
    save_consolidated: Literal["none", "final", "every"] = "final"
    staging_dir: Optional[str] = None
    best_metric_key: str = "default"
