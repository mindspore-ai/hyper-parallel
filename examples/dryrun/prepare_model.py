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
"""Write local config-only models used by the Dry-run examples."""

import argparse
from pathlib import Path

from transformers import (
    LlamaConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3MoeConfig,
)


def prepare_tiny_configs(output_dir: Path) -> None:
    """Write config-only model directories used by the Dry-run examples.

    Args:
        output_dir: Parent directory for the two local model configs.
    """
    llama_dir = output_dir / "tiny_llama"
    moe_dir = output_dir / "tiny_qwen3_moe"
    llama_dir.mkdir(parents=True, exist_ok=True)
    moe_dir.mkdir(parents=True, exist_ok=True)
    LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8192,
    ).save_pretrained(llama_dir)
    Qwen3MoeConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=8192,
    ).save_pretrained(moe_dir)


def prepare_8b_configs(output_dir: Path) -> None:
    """Write dense and MoE models with roughly eight billion parameters.

    The dense shape follows the Llama-3 8B family. The MoE model has roughly
    8.6B total parameters, of which only two experts are active per token.

    Args:
        output_dir: Parent directory for the config-only model directories.
    """
    llama_dir = output_dir / "llama_8b"
    moe_dir = output_dir / "qwen3_moe_8b"
    llama_dir.mkdir(parents=True, exist_ok=True)
    moe_dir.mkdir(parents=True, exist_ok=True)
    LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        tie_word_embeddings=False,
    ).save_pretrained(llama_dir)
    Qwen3MoeConfig(
        vocab_size=65536,
        hidden_size=2048,
        intermediate_size=5632,
        moe_intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_experts=32,
        num_experts_per_tok=2,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
    ).save_pretrained(moe_dir)


def prepare_legacy_validation_configs(output_dir: Path) -> None:
    """Write config-only Qwen3.5 models converted from old Trainer probes.

    Args:
        output_dir: Parent directory for the config-only model directories.
    """
    dense_one_layer_dir = output_dir / "qwen3_5_0_8b_one_layer"
    dense_four_layer_dir = output_dir / "qwen3_5_0_8b_four_layers"
    moe_eight_expert_dir = output_dir / "qwen3_5_moe_eight_experts"
    moe_visibility_dir = output_dir / "qwen3_5_moe_visibility"
    for model_dir in (
            dense_one_layer_dir,
            dense_four_layer_dir,
            moe_eight_expert_dir,
            moe_visibility_dir,
    ):
        model_dir.mkdir(parents=True, exist_ok=True)

    dense_kwargs = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "vocab_size": 248320,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "linear_num_value_heads": 16,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_key_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": True,
    }
    Qwen3_5TextConfig(num_hidden_layers=1, **dense_kwargs).save_pretrained(dense_one_layer_dir)
    Qwen3_5TextConfig(
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        **dense_kwargs,
    ).save_pretrained(dense_four_layer_dir)

    moe_kwargs = {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "vocab_size": 512,
        "hidden_size": 2048,
        "num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_key_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        # These converted probes isolate MoE routing memory. Keeping their
        # single layer on the full-attention branch also avoids conflating the
        # EP report with Qwen3.5 GatedDeltaNet state buffers.
        "layer_types": ["full_attention"],
        "tie_word_embeddings": False,
    }
    Qwen3_5MoeTextConfig(
        num_experts=8,
        num_experts_per_tok=2,
        **moe_kwargs,
    ).save_pretrained(moe_eight_expert_dir)
    Qwen3_5MoeTextConfig(
        num_experts=256,
        num_experts_per_tok=8,
        **moe_kwargs,
    ).save_pretrained(moe_visibility_dir)


def prepare_micro_batch_validation_configs(output_dir: Path) -> None:
    """Write execution-safe dense and MoE configs for four-rank validation."""
    dense_dir = output_dir / "micro_batch_validation_llama"
    moe_dir = output_dir / "micro_batch_validation_qwen3_moe"
    dense_dir.mkdir(parents=True, exist_ok=True)
    moe_dir.mkdir(parents=True, exist_ok=True)
    LlamaConfig(
        vocab_size=248320,
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        tie_word_embeddings=True,
    ).save_pretrained(dense_dir)
    Qwen3MoeConfig(
        vocab_size=32768,
        hidden_size=1024,
        intermediate_size=3584,
        moe_intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=16,
        num_experts_per_tok=2,
        max_position_embeddings=8192,
        tie_word_embeddings=False,
    ).save_pretrained(moe_dir)


def main() -> None:
    """Parse the output directory and generate both config-only models."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dryrun/models"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    prepare_tiny_configs(output_dir)
    prepare_8b_configs(output_dir)
    prepare_legacy_validation_configs(output_dir)
    prepare_micro_batch_validation_configs(output_dir)


if __name__ == "__main__":
    main()
