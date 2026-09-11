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
"""Prepare deterministic tokenizer and scaled-Engram assets for V4.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer

from hyper_parallel.models.deepseek_v41.configuration import (
    build_scaled_engram_buckets,
)


def _build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Build the normalization-equivalent token IDs used by Engram hashing."""
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    token_map = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        token_map[token_id] = key_to_id.setdefault(key, len(key_to_id))
    return token_map, len(key_to_id)


def _build_hash_multipliers(
        layer_ids: list[int],
        max_ngram_size: int,
        compressed_vocab_size: int,
) -> list[list[int]]:
    """Reproduce the official per-layer odd hash multipliers."""
    maximum = np.iinfo(np.int64).max
    upper_bound = max(1, (maximum // compressed_vocab_size) // 2)
    multipliers = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=upper_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        multipliers.append((values * 2 + 1).tolist())
    return multipliers


def prepare_deepseek_v41_assets(
        model_dir: Path,
        output_path: Path,
        *,
        bucket_base: int = 4096,
        num_hidden_layers: int = 4,
        table_pad_multiple: int = 16,
) -> None:
    """Create a local-only asset file for the validation crop.

    Args:
        model_dir: Local DeepSeek-V4.1 repository with tokenizer/config files.
        output_path: JSON file to create.
        bucket_base: Approximate rows per n-gram/head hash bucket.
        num_hidden_layers: Validation crop depth.
        table_pad_multiple: Stable row-count multiple for EP resharding.

    Raises:
        ValueError: If the source assets are not DeepSeek-V4.1 or the crop
            does not cover Engram and shared-attention reuse.
    """
    with (model_dir / "config.json").open("r", encoding="utf-8") as config_file:
        source_config = json.load(config_file)
    if source_config.get("model_type") != "deepseek_v41":
        raise ValueError("model_dir must contain a DeepSeek-V4.1 config.json")
    if num_hidden_layers != 4:
        raise ValueError("the validation integration currently requires exactly 4 decoder layers")
    if table_pad_multiple < 1:
        raise ValueError("table_pad_multiple must be positive")

    text_config = source_config["text_config"]
    layer_ids = [
        layer_id for layer_id in text_config["engram_layer_ids"]
        if layer_id < num_hidden_layers
    ]
    primes, table_sizes = build_scaled_engram_buckets(
        layer_ids,
        bucket_base=bucket_base,
        max_ngram_size=text_config["engram_max_ngram_size"],
        num_heads=text_config["engram_n_heads"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    token_map, compressed_vocab_size = _build_compressed_token_map(tokenizer)
    expected_vocab_size = text_config["engram_compressed_vocab_size"]
    if compressed_vocab_size != expected_vocab_size:
        raise ValueError(
            "tokenizer normalization does not match the V4.1 Engram contract: "
            f"expected {expected_vocab_size}, got {compressed_vocab_size}"
        )

    output = {
        "source_model_type": source_config["model_type"],
        "num_hidden_layers": num_hidden_layers,
        "bucket_base": bucket_base,
        "layer_ids": layer_ids,
        "primes": primes,
        "num_embeddings": table_sizes,
        "padded_num_embeddings": [
            (size + table_pad_multiple - 1) // table_pad_multiple * table_pad_multiple
            for size in table_sizes
        ],
        "table_pad_multiple": table_pad_multiple,
        "max_ngram_size": text_config["engram_max_ngram_size"],
        "num_heads": text_config["engram_n_heads"],
        "head_dim": text_config["engram_head_dim"],
        "pad_token_id": text_config["engram_pad_token_id"],
        "compressed_vocab_size": compressed_vocab_size,
        "token_map": token_map,
        "multipliers": _build_hash_multipliers(
            layer_ids,
            text_config["engram_max_ngram_size"],
            compressed_vocab_size,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=False)
        output_file.write("\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare DeepSeek-V4.1 validation assets")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bucket-base", type=int, default=4096)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--table-pad-multiple", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare the requested local assets."""
    args = _parse_args(argv)
    prepare_deepseek_v41_assets(
        Path(args.model_dir).expanduser().resolve(),
        Path(args.output).expanduser().resolve(),
        bucket_base=args.bucket_base,
        num_hidden_layers=args.num_hidden_layers,
        table_pad_multiple=args.table_pad_multiple,
    )


if __name__ == "__main__":
    main()
