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
"""Write deterministic local JSONL documents for the generic HF text demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def prepare_online_data(output_path: Path, *, num_samples: int = 128,
                        sequence_length: int = 2048) -> None:
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
                f"HF text bring-up document {sample_index}. "
                "Deterministic local data keeps the smoke run reproducible "
                "across nodes and reruns. "
            )
            output_file.write(json.dumps({"text": sentence * repetitions}) + "\n")


def build_uniform_text(unit: str, target_tokens: int, count_tokens,
                       fillers=(" a", " the", " x", " of")) -> str:
    """Build text that tokenizes to exactly ``target_tokens``.

    Repeating a unit only lands on the target when the unit's token length
    divides it, which is a coincidence rather than a design. So: repeat to
    the largest count that stays under the target, then append a filler that
    adds exactly one token until the count matches, and verify the result.
    Raises when no filler behaves as a single token, because an approximate
    length reintroduces the packing tail this function exists to remove.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    low, high = 1, 2
    while count_tokens(unit * high) <= target_tokens:
        low, high = high, high * 2
        if high > 1 << 20:
            raise ValueError("unit does not grow; cannot reach the target")
    while low < high:                      # largest count with length <= target
        middle = (low + high + 1) // 2
        if count_tokens(unit * middle) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    text = unit * low
    length = count_tokens(text)

    single = next((f for f in fillers if count_tokens(text + f) == length + 1),
                  None)
    if single is None and length != target_tokens:
        raise ValueError("no single-token filler available for this tokenizer; "
                         "pass a different filler set")
    while length < target_tokens:
        text += single
        length = count_tokens(text)

    if length != target_tokens:
        raise ValueError(f"built {length} tokens, wanted {target_tokens}")
    return text


def prepare_uniform_data(output_path: Path, *, num_samples: int,
                         sequence_length: int, tokenizer_path: str,
                         sequences_per_document: int = 4) -> int:
    """Write documents whose token length is an exact multiple of a sequence.

    Cross-layout equivalence needs every layout to consume the *same* tokens
    per step. With variable-length documents the packer emits a partial tail
    whose placement depends on the partition, so two layouts legitimately see
    different token counts and the comparison stops being about parallelism.
    Documents sized to an exact multiple of ``sequence_length`` leave no tail.

    Returns the per-document token count actually achieved.
    """
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              local_files_only=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    target = sequence_length * sequences_per_document
    text = build_uniform_text("reproducible bring-up token ", target,
                              count_tokens)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for _ in range(num_samples):
            output_file.write(json.dumps({"text": text}) + "\n")
    return target


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare text-demo data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--uniform-tokenizer", default=None,
                        help="model directory whose tokenizer sizes every "
                             "document to an exact multiple of the sequence "
                             "length; required for cross-layout equivalence")
    parser.add_argument("--sequences-per-document", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.uniform_tokenizer:
        tokens = prepare_uniform_data(
            Path(args.output),
            num_samples=args.num_samples,
            sequence_length=args.sequence_length,
            tokenizer_path=args.uniform_tokenizer,
            sequences_per_document=args.sequences_per_document,
        )
        print(f"[text-data] {args.num_samples} documents of exactly "
              f"{tokens} tokens ({args.sequences_per_document} sequences each)")
        return
    prepare_online_data(
        Path(args.output),
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
    )


if __name__ == "__main__":
    main()
