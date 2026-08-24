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

"""Download a Hugging Face dataset and convert it to indexed binary data.

This module is both importable and runnable as a CLI entry point::

    python -m hyper_parallel.auto_models.components.datasets.tools.huggingface_offline \\
        --dataset Salesforce/wikitext \\
        --dataset-subset wikitext-103-raw-v1 \\
        --output-prefix ./offline_datasets/my_dataset \\
        --tokenizer gpt2 \\
        --json-keys text \\
        --workers 8
"""

import argparse
import os
from pathlib import Path
from typing import Any, List

from hyper_parallel.auto_models.components.datasets.dataset_logging import get_dataset_logger
from hyper_parallel.auto_models.components.datasets.tools.offline_config import OfflinePreparationConfig
from hyper_parallel.auto_models.components.datasets.tools.offline_preparation import (
    prepare_offline_dataset,
)

logger = get_dataset_logger(__name__)


def _download_jsonl(config: OfflinePreparationConfig) -> Path:
    """Download and normalize the configured Hugging Face split as JSONL."""
    json_path = config.resolved_json_path()
    source_path = Path(config.dataset_name_or_path).expanduser()
    local_suffixes = (".json", ".jsonl", ".json.gz", ".jsonl.gz")
    if source_path.is_dir() or source_path.name.lower().endswith(local_suffixes):
        logger.info("Using local JSON dataset input at %s", json_path)
        return json_path

    if json_path.is_file():
        logger.info("Reusing downloaded Hugging Face dataset at %s", json_path)
        return json_path

    logger.info(
        "Downloading Hugging Face dataset %s (config=%s, split=%s)",
        config.dataset_name_or_path,
        config.dataset_subset_name,
        config.dataset_split,
    )
    from datasets import load_dataset  # pylint: disable=C0415

    if config.num_proc is not None and config.num_proc <= 0:
        raise ValueError("num_proc must be greater than zero")

    load_dataset_kwargs: dict[str, Any] = {
        "path": config.dataset_name_or_path,
        "split": config.dataset_split,
    }
    optional_load_dataset_kwargs = {
        "name": config.dataset_subset_name,
        "revision": config.revision,
        "cache_dir": config.cache_dir,
        "data_dir": config.data_dir,
        "data_files": config.data_files,
        "num_proc": config.num_proc,
    }
    load_dataset_kwargs.update({key: value for key, value in optional_load_dataset_kwargs.items() if value is not None})
    dataset = load_dataset(**load_dataset_kwargs)
    keys = config.json_keys_list()
    missing_keys = [key for key in keys if key not in dataset.column_names]
    if missing_keys:
        raise ValueError(
            f"Dataset {config.dataset_name_or_path} does not contain configured "
            f"keys {missing_keys}; "
            f"available columns: {dataset.column_names}"
        )
    dataset = dataset.select_columns(keys)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(
        str(json_path),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    logger.info("Saved Hugging Face dataset to %s", json_path)
    return json_path


def _parse_bool(value: str) -> bool:
    """Parse an explicit command-line boolean value."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, but received {value!r}")


def _add_huggingface_arguments(parser: argparse.ArgumentParser) -> None:
    """Register Hugging Face source and output arguments."""
    dataset_group = parser.add_argument_group("dataset")
    dataset_arguments = (
        (("--dataset",), {"required": True, "help": "Dataset ID or local JSON/JSONL path."}),
        (("--dataset-subset",), {"default": None, "help": "Optional dataset subset."}),
        (("--dataset-split",), {"default": "train", "help": "Dataset split."}),
        (("--revision",), {"default": None, "help": "Optional dataset revision."}),
        (("--cache-dir",), {"default": None, "help": "Optional dataset cache directory."}),
        (("--data-dir",), {"default": None, "help": "Optional repository data directory."}),
        (("--data-files",), {"nargs": "+", "default": None, "help": "Optional source data files."}),
        (("--num-proc",), {"type": int, "default": None, "help": "Dataset preparation process count."}),
        (("--json-keys",), {"nargs": "+", "default": ["text"], "help": "JSON fields to tokenize."}),
    )
    for flags, options in dataset_arguments:
        dataset_group.add_argument(*flags, **options)
    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-prefix", required=True, help="Generated .bin/.idx path prefix.")
    output_group.add_argument("--download-dir", default=None, help="Raw JSONL download directory.")


def _add_conversion_arguments(parser: argparse.ArgumentParser) -> None:
    """Register tokenizer, preprocessing, parallelism, and benchmark arguments."""
    groups = {
        "tokenizer": (
            (("--tokenizer",), {"required": True, "help": "Tokenizer name or local path."}),
            (("--tokenizer-use-fast",), {"type": _parse_bool, "default": True, "help": "Use fast tokenizer."}),
            (("--trust-remote-code",), {"action": "store_true", "help": "Allow tokenizer remote code."}),
            (("--chat-template",), {"default": None, "help": "Optional tokenizer chat template."}),
            (("--add-special-tokens",), {"nargs": "+", "default": None, "help": "Additional special tokens."}),
        ),
        "preprocessing": (
            (("--split-sentences",), {"action": "store_true", "help": "Split text into sentences."}),
            (("--keep-newlines",), {"action": "store_true", "help": "Preserve newline runs."}),
            (("--lang",), {"default": "english", "help": "Punkt language."}),
            (("--append-eod",), {"type": _parse_bool, "default": True, "help": "Append an EOD token."}),
            (("--pack-to-seq-len",), {"type": int, "default": None, "help": "Fixed packed sequence length."}),
        ),
        "parallelism": (
            (("--workers",), {"type": int, "default": 8, "help": "Worker process count."}),
            (("--partitions",), {"type": int, "default": 1, "help": "Data partition count."}),
            (("--keep-sequential-samples",), {"action": "store_true", "help": "Keep sample order."}),
            (("--keep-partition-files",), {"action": "store_true", "help": "Keep partition files."}),
        ),
        "benchmark": (
            (("--find-optimal-num-workers",), {"action": "store_true", "help": "Benchmark worker counts."}),
            (
                ("--workers-to-check",),
                {"nargs": "+", "type": int, "default": [16, 32, 64], "help": "Candidate worker counts."},
            ),
            (("--max-documents",), {"type": int, "default": 100_000, "help": "Benchmark document limit."}),
            (("--log-interval",), {"type": int, "default": 1000, "help": "Progress-report interval."}),
        ),
    }
    for title, arguments in groups.items():
        group = parser.add_argument_group(title)
        for flags, options in arguments:
            group.add_argument(*flags, **options)


def _get_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse the Hugging Face offline preparation CLI arguments."""
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset and convert to .bin/.idx")
    _add_huggingface_arguments(parser)
    _add_conversion_arguments(parser)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Download, tokenize, and convert a dataset to ``.bin/.idx`` files."""
    args = _get_args(argv)

    config = OfflinePreparationConfig(
        dataset_name_or_path=args.dataset,
        dataset_subset_name=args.dataset_subset,
        dataset_split=args.dataset_split,
        revision=args.revision,
        cache_dir=args.cache_dir,
        data_dir=args.data_dir,
        data_files=args.data_files,
        num_proc=args.num_proc,
        json_keys=args.json_keys,
        output_prefix=args.output_prefix,
        download_dir=args.download_dir,
        tokenizer_name_or_path=args.tokenizer,
        tokenizer_use_fast=args.tokenizer_use_fast,
        trust_remote_code=args.trust_remote_code,
        chat_template=args.chat_template,
        add_special_tokens=args.add_special_tokens,
        split_sentences=args.split_sentences,
        keep_newlines=args.keep_newlines,
        lang=args.lang,
        workers=args.workers,
        partitions=args.partitions,
        append_eod=args.append_eod,
        pack_to_seq_len=args.pack_to_seq_len,
        keep_sequential_samples=args.keep_sequential_samples,
        keep_partition_files=args.keep_partition_files,
        find_optimal_num_workers=args.find_optimal_num_workers,
        workers_to_check=args.workers_to_check,
        max_documents=args.max_documents,
        log_interval=args.log_interval,
    )

    if int(os.environ.get("RANK", "0")) != 0:
        return

    _download_jsonl(config)
    prepare_offline_dataset(config.to_offline_args())


if __name__ == "__main__":
    main()
