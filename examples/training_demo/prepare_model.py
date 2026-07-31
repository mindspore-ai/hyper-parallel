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
"""Prepare the tiny local Llama checkpoint used by the Trainer demo."""

import logging
from pathlib import Path

from transformers import LlamaConfig, LlamaForCausalLM

from hyper_models.config.manager import parse_training_args
from hyper_models.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)


def prepare_tiny_model(model_dir: Path) -> None:
    """Create the demo checkpoint when it does not already exist.

    Args:
        model_dir: Directory that stores the generated checkpoint.
    """
    if (model_dir / "config.json").exists():
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)
    logger.info("Wrote tiny Llama checkpoint to %s", model_dir)


def main() -> None:
    """Resolve the demo configuration and prepare its model checkpoint."""
    config: TrainerConfig = parse_training_args()
    model_dir = Path(
        config.model.pretrained_model_name_or_path
        or "./outputs/training_demo/tiny_llama"
    ).resolve()
    prepare_tiny_model(model_dir)


if __name__ == "__main__":
    main()
