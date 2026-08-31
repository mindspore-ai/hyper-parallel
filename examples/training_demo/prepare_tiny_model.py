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
"""Prepare the deterministic compact Qwen3-MoE checkpoint for the demo."""

import argparse
import json
import logging
from pathlib import Path

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, set_seed


logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("examples/training_demo/tiny_qwen3_moe")


def build_tiny_qwen3_config() -> Qwen3MoeConfig:
    """Build the compact Qwen3-MoE configuration used by ``train.yaml``."""
    return Qwen3MoeConfig(
        vocab_size=256,
        # Keep enough attention heads after TP2 for Pure Ulysses CP2:
        # Q/KV become 4/2 heads per rank, both divisible by Ulysses degree 2.
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=64,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        output_router_logits=False,
        use_cache=False,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation="sdpa",
    )


def prepare_tiny_model(output_dir: Path) -> None:
    """Write the compact checkpoint unless it already exists.

    Args:
        output_dir: Directory that stores the model configuration and weights.
    """
    config_path = output_dir / "config.json"
    weights_path = output_dir / "model.safetensors"
    if config_path.exists() and weights_path.exists():
        expected = build_tiny_qwen3_config()
        with config_path.open("r", encoding="utf-8") as config_file:
            existing = json.load(config_file)
        shape_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_experts",
            "num_experts_per_tok",
        )
        if all(existing.get(name) == getattr(expected, name) for name in shape_fields):
            logger.info("Reusing compact Qwen3-MoE checkpoint at %s", output_dir)
            return
        logger.info("Regenerating incompatible Qwen3-MoE checkpoint at %s", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    model = Qwen3MoeForCausalLM(build_tiny_qwen3_config())
    model.save_pretrained(output_dir, safe_serialization=True)
    logger.info("Wrote compact Qwen3-MoE checkpoint to %s", output_dir)


def main() -> None:
    """Prepare the checkpoint requested by the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_tiny_model(args.output_dir)


if __name__ == "__main__":
    main()
