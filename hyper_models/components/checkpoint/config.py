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
from typing import Literal, Optional


@dataclass
class CheckpointingConfig:
    """Checkpointing typed config.

    Following design doc 04_checkpoint.md.
    """

    enabled: bool = True
    checkpoint_dir: str = "./checkpoints"
    model_save_format: str = "safetensors"
    # 是否额外输出合并的 HF 权重："none"（从不）| "final"（仅训练结束）| "every"（每次保存）。
    # 关闭档统一为 "none"（YAML 安全的 Literal 取值），替代旧设计枚举的 "false"
    # （"false" 会被 PyYAML 解析为 bool，见 04_checkpoint.md §4.2 口径裁决）。
    save_consolidated: Literal["none", "final", "every"] = "final"
    is_peft: bool = False
    is_async: bool = False
    staging_dir: Optional[str] = None
    best_metric_key: str = "default"
    restore_from: Optional[str] = None  # "LATEST" or specific path
