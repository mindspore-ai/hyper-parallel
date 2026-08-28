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
"""Prepare deterministic Offline and Online data for the cropped-model demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from hyper_parallel.auto_models.components.datasets.tools.indexed_dataset import IndexedDatasetBuilder


def prepare_parallel_data(output_dir: Path, *, num_samples: int = 128, seq_length: int = 128) -> None:
    """Write Online JSONL and Offline Indexed Dataset files.

    Offline samples contain ``seq_length + 1`` tokens because the Indexed
    dataset creates inputs from ``text[:-1]`` and already-shifted labels from
    ``text[1:]``. Online samples remain unshifted and are tokenized at runtime.

    Args:
        output_dir: Directory containing generated demo data.
        num_samples: Number of deterministic documents to generate.
        seq_length: Number of model input tokens in each Offline sample.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "parallel_online.jsonl"
    data_prefix = output_dir / "parallel_offline_text_document"

    with json_path.open("w", encoding="utf-8") as json_file:
        for sample_index in range(num_samples):
            text = (
                f"HyperParallel cropped Qwen3 MoE online sample {sample_index}. "
                "This deterministic sentence exercises packed causal language model training. "
            ) * 4
            json_file.write(json.dumps({"text": text}) + "\n")

    builder = IndexedDatasetBuilder(str(data_prefix.with_suffix(".bin")), dtype=np.uint16)
    for sample_index in range(num_samples):
        document = [
            ((sample_index * seq_length + token_offset) % 1000) + 1
            for token_offset in range(seq_length + 1)
        ]
        builder.add_document(document, [len(document)])
    builder.finalize(str(data_prefix.with_suffix(".idx")))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse data preparation arguments."""
    parser = argparse.ArgumentParser(description="Prepare cropped Qwen3-MoE demo data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seq-length", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate both data representations used by the demos."""
    args = _parse_args(argv)
    prepare_parallel_data(
        Path(args.output_dir).expanduser().resolve(),
        num_samples=args.num_samples,
        seq_length=args.seq_length,
    )


if __name__ == "__main__":
    main()
