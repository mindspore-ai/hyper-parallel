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
"""End-to-end multi-card training skeleton example.

This script demonstrates the full Hyper-Parallel training skeleton:
  - YAML config parsing
  - distributed initialization + DeviceMesh (TP/CP/DP/EP topology)
  - dummy data pipeline
  - model build (tiny local GPT-2)
  - optimizer / LR scheduler / loss / step scheduler
  - FSDP2 (stub — called but no-op)
  - checkpoint save/load (stub — called but no-op)
  - training loop producing loss and gradients

Run with torchrun, e.g.:

    torchrun --nproc_per_node=4 examples/training_skeleton/run.py \
        examples/training_skeleton/train.yaml

The example deliberately uses tiny models and dummy data; loss values are not
meant to be meaningful. The goal is to exercise every major code path of the
skeleton on multiple cards.
"""

import logging
import sys
from pathlib import Path

import torch.distributed as dist

# Make the repository root importable.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from hyper_models.components.distributed.infrastructure import initialize_distributed
from hyper_models.config.manager import parse_training_args
from hyper_models.recipes import RECIPE_REGISTRY
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)


def _ensure_tiny_model(model_dir: Path) -> None:
    """Create a tiny local GPT-2 checkpoint on rank 0, barrier for others."""
    if dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        return

    if not (model_dir / "config.json").exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        config = GPT2Config(
            vocab_size=1000,
            n_positions=64,
            n_embd=64,
            n_layer=2,
            n_head=4,
            n_inner=256,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )
        model = GPT2LMHeadModel(config)
        model.save_pretrained(model_dir)
        logger.info("Wrote tiny GPT-2 checkpoint to %s", model_dir)

    if dist.is_initialized():
        dist.barrier()


def main() -> None:
    # Initialize distributed early so that rank 0 can prepare the checkpoint
    # and all ranks agree on the topology before building the recipe.
    initialize_distributed("nccl")

    cfg = parse_training_args()

    # Resolve the model path relative to the CWD / YAML location.
    model_dir = Path(cfg.model.weights_path or "./outputs/tiny_model").resolve()
    _ensure_tiny_model(model_dir)
    cfg.model.weights_path = str(model_dir)
    if cfg.model.tokenizer_path is None:
        cfg.model.tokenizer_path = str(model_dir)

    recipe = RECIPE_REGISTRY[cfg.recipe]()
    recipe.setup(cfg)
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
