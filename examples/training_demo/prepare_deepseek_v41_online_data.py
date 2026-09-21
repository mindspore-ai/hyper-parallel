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
"""Generate deterministic long documents for V4.1 Online 4K validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def prepare_online_data(output_path: Path, *, num_samples: int = 128, sequence_length: int = 4096) -> None:
    """Write local long-text JSONL records for Online tokenization.

    Args:
        output_path: JSONL file to create.
        num_samples: Number of deterministic source documents.
        sequence_length: Target model sequence length.
    """
    if num_samples <= 0 or sequence_length <= 0:
        raise ValueError("num_samples and sequence_length must be positive")
    repetitions = max(16, sequence_length // 4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for sample_index in range(num_samples):
            sentence = (
                f"DeepSeek V4.1 Online validation document {sample_index}. "
                "Engram hashing and shared compressed attention remain active. "
            )
            output_file.write(json.dumps({"text": sentence * repetitions}) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare V4.1 Online validation data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Generate the requested JSONL dataset."""
    args = _parse_args(argv)
    prepare_online_data(
        Path(args.output).expanduser().resolve(),
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
