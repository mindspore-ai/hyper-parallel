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
"""Opt-in MLA kernel registration followed by the unchanged TextTrainer."""

import torch.distributed as dist

from hyper_parallel.models.deepseek_v3.adapter.data.packed_mla import register_head_chunk_mla_attention
from hyper_parallel.trainer import TextTrainer
from hyper_parallel.trainer.config.parser import parse_training_args


def main() -> None:
    """Train from a complete user-supplied configuration, without downloading assets."""
    register_head_chunk_mla_attention()
    config = parse_training_args()
    try:
        TextTrainer(config).train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
