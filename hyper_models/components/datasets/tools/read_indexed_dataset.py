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
"""
Inspect token sequences stored in a Megatron indexed Dataset.

python -m hyper_models.components.datasets.tools.read_indexed_dataset \
  --path outputs/training_demo/tiny_llama_wikitext2/data_no_pack/wikitext2_real_text_document.idx \
  --tokenizer outputs/training_demo/tiny_llama_wikitext2/tokenizer \
  --num-samples 5
"""

import argparse
import logging
from collections.abc import Sequence

from transformers import AutoTokenizer

from hyper_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_models.components.datasets.llm.indexed_data_reader import IndexedDataReader

logger = get_dataset_logger(__name__)


def _path_prefix(path: str) -> str:
    """Remove an optional indexed Dataset file suffix from a path."""
    return path[:-4] if path.endswith((".bin", ".idx")) else path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse indexed Dataset inspection arguments."""
    parser = argparse.ArgumentParser(description="Inspect a Megatron indexed Dataset")
    parser.add_argument("--path", required=True, help="Dataset prefix or its .bin/.idx path")
    parser.add_argument("--tokenizer", help="Local Hugging Face tokenizer path or repository")
    parser.add_argument(
        "--tokenizer-type", choices=("huggingface", "pretokenized"), default="huggingface",
        help="Use pretokenized when only vocabulary metadata is available",
    )
    parser.add_argument("--vocab-size", type=int, help="Vocabulary size for pretokenized data")
    parser.add_argument("--eod-token-id", type=int, help="EOD token ID for pretokenized data")
    parser.add_argument("--start-index", type=int, default=0, help="First sequence index to read")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sequences to read")
    parser.add_argument("--max-token-ids", type=int, default=80, help="Maximum token IDs printed per sequence")
    parser.add_argument("--max-text-chars", type=int, default=1000, help="Maximum decoded characters per sequence")
    parser.add_argument("--skip-special-tokens", action="store_true", help="Remove special tokens when decoding")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Print indexed Dataset metadata and selected token sequences."""
    args = _parse_args(argv)
    if args.start_index < 0 or args.num_samples < 0:
        raise ValueError("start-index and num-samples must be non-negative")

    if args.max_token_ids < 0 or args.max_text_chars < 0:
        raise ValueError("max-token-ids and max-text-chars must be non-negative")

    reader = IndexedDataReader(_path_prefix(args.path))
    tokenizer = None
    if args.tokenizer_type == "huggingface" and args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.tokenizer_type == "pretokenized":
        if args.vocab_size is None or args.eod_token_id is None:
            raise ValueError("pretokenized mode requires vocab-size and eod-token-id")
        logger.info(
            "Pretokenized metadata: vocab_size=%d, eod_token_id=%d; decoded text is unavailable",
            args.vocab_size, args.eod_token_id,
        )
    sequence_lengths = reader.sequence_lengths
    logger.info(
        "Dataset: sequences=%d, documents=%d, min_length=%d, max_length=%d, mean_length=%.2f",
        len(reader), len(reader.document_indices) - 1, int(sequence_lengths.min()), int(sequence_lengths.max()),
        float(sequence_lengths.mean()),
    )

    stop_index = min(args.start_index + args.num_samples, len(reader))
    for index in range(args.start_index, stop_index):
        token_ids = reader[index].tolist()
        displayed_ids = token_ids[:args.max_token_ids]
        suffix = " ..." if len(displayed_ids) < len(token_ids) else ""
        logger.info("[%d] length=%d\ntoken_ids=%s%s", index, len(token_ids), displayed_ids, suffix)

        if args.tokenizer_type == "pretokenized":
            eod_positions = [position for position, token_id in enumerate(token_ids) if token_id == args.eod_token_id]
            logger.info("eod_positions=%s", eod_positions)

        if tokenizer is not None:
            text = tokenizer.decode(token_ids, skip_special_tokens=args.skip_special_tokens)
            text_suffix = " ..." if len(text) > args.max_text_chars else ""
            logger.info("text=%r%s", text[:args.max_text_chars], text_suffix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
