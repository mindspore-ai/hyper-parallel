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
"""Prepare deterministic online and indexed data for the parallel training demo."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from hyper_models.components.datasets.tools.indexed_dataset import IndexedDatasetBuilder


def prepare_parallel_data(output_dir: Path, *, num_samples: int = 64, seq_length: int = 32) -> None:
    """Write equivalent JSONL and Megatron indexed datasets.

    Args:
        output_dir: Directory containing the generated validation data.
        num_samples: Number of deterministic documents to generate.
        seq_length: Number of model input tokens per indexed sample.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "parallel_online.jsonl"
    data_prefix = output_dir / "parallel_offline_text_document"
    bin_path = data_prefix.with_suffix(".bin")
    idx_path = data_prefix.with_suffix(".idx")

    records = []
    documents = []
    for sample_index in range(num_samples):
        tokens = [((sample_index * seq_length + offset) % 250) + 1 for offset in range(seq_length + 1)]
        records.append({"input_ids": tokens[:-1], "labels": tokens[1:]})
        documents.append(tokens)

    with json_path.open("w", encoding="utf-8") as json_file:
        for record in records:
            json_file.write(json.dumps(record) + "\n")

    builder = IndexedDatasetBuilder(str(bin_path), dtype=np.uint16)
    for document in documents:
        builder.add_document(document, [len(document)])
    builder.finalize(str(idx_path))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse data preparation arguments."""
    parser = argparse.ArgumentParser(description="Prepare TP/CP/EP/FSDP demo data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare both data representations used by the parallel demo."""
    args = _parse_args(argv)
    prepare_parallel_data(
        Path(args.output_dir).expanduser().resolve(),
        num_samples=args.num_samples,
        seq_length=args.seq_length,
    )


if __name__ == "__main__":
    main()
