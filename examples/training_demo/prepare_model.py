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
"""Prepare the local random Qwen3 checkpoint used by the Trainer demo."""

import logging
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from hyper_models.config.manager import parse_training_args
from hyper_models.trainer.config import TrainerConfig

logger = logging.getLogger(__name__)

QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B"
_MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _checkpoint_is_complete(model_dir: Path) -> bool:
    """Return whether the local model and tokenizer assets are complete."""
    has_weights = any((model_dir / filename).exists() for filename in _MODEL_WEIGHT_FILES)
    return (
        (model_dir / "config.json").exists()
        and (model_dir / "tokenizer_config.json").exists()
        and has_weights
    )


def prepare_qwen3_checkpoint(model_dir: Path) -> None:
    """Create a random Qwen3-30B-A3B checkpoint from its Hub config.

    Args:
        model_dir: Directory that stores the generated model and tokenizer.
    """
    if _checkpoint_is_complete(model_dir):
        logger.info("Reusing prepared Qwen3 checkpoint at %s", model_dir)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Preparing random %s checkpoint at %s", QWEN3_MODEL_ID, model_dir)
    config = AutoConfig.from_pretrained(QWEN3_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.save_pretrained(
        model_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(model_dir)
    logger.info("Wrote random Qwen3 checkpoint and tokenizer to %s", model_dir)


def main() -> None:
    """Resolve the demo configuration and prepare its Qwen3 checkpoint."""
    config: TrainerConfig = parse_training_args()
    model_dir = Path(
        config.model.pretrained_model_name_or_path
        or "./outputs/training_demo/qwen3_30b_a3b"
    ).resolve()
    prepare_qwen3_checkpoint(model_dir)


if __name__ == "__main__":
    main()
