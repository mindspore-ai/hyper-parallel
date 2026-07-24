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
"""CheckpointingConfig — typed config for checkpoint save/load.

Following design doc 04_checkpoint.md §4.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CheckpointingConfig:
    """Checkpointing typed config.

    Following design doc 04_checkpoint.md.
    """

    enabled: bool = True
    checkpoint_dir: str = "./checkpoints"
    model_save_format: str = "safetensors"
    save_consolidated: str = "final"  # "none" | "final" | "every"
    is_peft: bool = False
    is_async: bool = False
    staging_dir: Optional[str] = None
    best_metric_key: str = "default"
    restore_from: Optional[str] = None  # "LATEST" or specific path

    def build(
        self,
        dp_rank: int = 0,
        tp_rank: int = 0,
        pp_rank: int = 0,
        moe_mesh: Any = None,
        process_group: Any = None,
    ) -> "Checkpointer":
        """Build Checkpointer from config.

        Stub — returns a Checkpointer with this config.
        """
        from hyper_models.components.checkpoint.checkpointing import Checkpointer
        return Checkpointer(config=self)