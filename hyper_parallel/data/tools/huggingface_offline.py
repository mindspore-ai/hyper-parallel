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

    python -m hyper_parallel.data.tools.huggingface_offline \\
        --dataset Salesforce/wikitext \\
        --dataset-subset wikitext-103-raw-v1 \\
        --output-prefix ./offline_datasets/my_dataset \\
        --tokenizer gpt2 \\
        --json-keys text \\
        --workers 8
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Any, List

from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.tools.offline_config import OfflinePreparationConfig
from hyper_parallel.data.tools.offline_record_transform import parse_role_map
from hyper_parallel.data.tools.offline_preparation import (
    prepare_offline_dataset,
)

logger = get_dataset_logger(__name__)


def _download_jsonl(config: OfflinePreparationConfig) -> Path:
    """Download and normalize the configured Hugging Face split as JSONL."""
    json_path = config.resolved_json_path()
    source_path = Path(config.dataset_name_or_path).expanduser()
    local_json_suffixes = (".json", ".jsonl")
    local_file_suffixes = local_json_suffixes + (
        ".json.gz", ".jsonl.gz", ".csv", ".csv.gz", ".parquet", ".arrow", ".txt", ".txt.gz",
    )
    if source_path.is_dir():
        json_input_files = _resolve_local_json_inputs(source_path)
        if json_input_files:
            return _materialize_local_files(config, json_input_files, json_path)
        local_files = _resolve_local_files(source_path)
        return _materialize_local_files(config, local_files, json_path)

    if not source_path.exists() and any(char in config.dataset_name_or_path for char in "*?["):
        local_files = [Path(name).resolve() for name in sorted(glob.glob(config.dataset_name_or_path))]
        if local_files:
            return _materialize_local_files(config, local_files, json_path)

    if source_path.is_file() and source_path.name.lower().endswith(local_file_suffixes):
        if source_path.name.lower().endswith(local_json_suffixes):
            logger.info("Using local JSON dataset input at %s", source_path)
            return source_path.resolve()
        return _materialize_local_files(config, [source_path.resolve()], json_path)

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
    if not config.uses_record_transform():
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


def _infer_local_hf_builder(path: Path) -> str:
    """Infer a Hugging Face Datasets builder for a local tabular/text file."""
    lowered = path.name.lower()
    for suffix, builder in (
        (".jsonl.gz", "json"),
        (".json.gz", "json"),
        (".csv.gz", "csv"),
        (".txt.gz", "text"),
        (".jsonl", "json"),
        (".json", "json"),
        (".csv", "csv"),
        (".parquet", "parquet"),
        (".arrow", "arrow"),
        (".txt", "text"),
    ):
        if lowered.endswith(suffix):
            return builder
    raise ValueError(f"Unsupported local Hugging Face data file format: {path}")


def _infer_text_format(path: Path) -> str:
    """Return the underlying data format for plain and compressed text files."""
    lowered = path.name.lower()
    if lowered.endswith((".json.gz", ".jsonl.gz", ".json", ".jsonl")):
        return "json"
    if lowered.endswith((".csv.gz", ".csv")):
        return "csv"
    if lowered.endswith((".txt.gz", ".txt")):
        return "text"
    if lowered.endswith(".parquet"):
        return "parquet"
    if lowered.endswith(".arrow"):
        return "arrow"
    raise ValueError(f"Unsupported local Hugging Face data file format: {path}")


def _resolve_local_json_inputs(source_path: Path) -> list[Path]:
    """Find local JSON/JSONL files recursively in stable order."""
    suffixes = (".json", ".jsonl", ".json.gz", ".jsonl.gz")
    if source_path.is_dir():
        files = [path.resolve() for path in source_path.rglob("*") if path.is_file()]
        return sorted((path for path in files if path.name.lower().endswith(suffixes)), key=str)
    if source_path.is_file() and source_path.name.lower().endswith(suffixes):
        return [source_path.resolve()]
    return []


def _resolve_local_files(source_path: Path) -> list[Path]:
    """Find supported local HF data files recursively in stable order."""
    suffixes = (
        ".json", ".jsonl", ".json.gz", ".jsonl.gz", ".csv", ".csv.gz",
        ".parquet", ".arrow", ".txt", ".txt.gz",
    )
    files = [path.resolve() for path in source_path.rglob("*") if path.is_file()]
    return sorted((path for path in files if path.name.lower().endswith(suffixes)), key=str)


def _materialize_local_files(
        config: OfflinePreparationConfig,
        input_files: list[Path],
        json_path: Path,
) -> Path:
    """Load local files through Hugging Face Datasets and normalize to JSONL."""
    if not input_files:
        raise ValueError(f"No supported local data files found under {config.dataset_name_or_path!r}")

    from datasets import load_dataset  # pylint: disable=C0415

    file_formats = {_infer_text_format(path) for path in input_files}
    if len(file_formats) != 1:
        raise ValueError(f"Local dataset files must use one format, got {sorted(file_formats)!r}")
    file_format = file_formats.pop()
    builder = "text" if file_format == "text" else _infer_local_hf_builder(input_files[0])
    dataset = load_dataset(
        builder,
        data_files={config.dataset_split: [str(path) for path in input_files]},
        split=config.dataset_split,
        cache_dir=config.cache_dir,
    )
    if not config.uses_record_transform():
        keys = config.json_keys_list()
        missing_keys = [key for key in keys if key not in dataset.column_names]
        if missing_keys:
            raise ValueError(
                f"Local dataset does not contain configured keys {missing_keys}; "
                f"available columns: {dataset.column_names}"
            )
        dataset = dataset.select_columns(keys)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(json_path), orient="records", lines=True, force_ascii=False)
    logger.info("Normalized %d local source files (loader=%s) to %s", len(input_files), builder, json_path)
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
        (("--dataset",), {"required": True, "help": "Dataset ID or local HF JSON/CSV/TXT/Parquet/Arrow path."}),
        (("--dataset-subset",), {"default": None, "help": "Optional dataset subset."}),
        (("--dataset-split",), {"default": "train", "help": "Dataset split."}),
        (("--revision",), {"default": None, "help": "Optional dataset revision."}),
        (("--cache-dir",), {"default": None, "help": "Optional dataset cache directory."}),
        (("--data-dir",), {"default": None, "help": "Optional repository data directory."}),
        (("--data-files",), {"nargs": "+", "default": None, "help": "Optional source data files."}),
        (("--num-proc",), {"type": int, "default": None, "help": "Dataset preparation process count."}),
        (("--json-keys",), {"nargs": "+", "default": ["text"], "help": "JSON fields to tokenize."}),
        (("--text-template",), {"default": None, "help": "Python format template for one source record."}),
        (("--conversation-key",), {"default": None, "help": "Conversation list field to render."}),
        (("--role-key",), {"default": "role", "help": "Conversation role field."}),
        (("--content-key",), {"default": "content", "help": "Conversation content field."}),
        (("--role-map",), {"type": parse_role_map, "default": None, "help": "JSON role alias map."}),
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
        text_template=args.text_template,
        conversation_key=args.conversation_key,
        role_key=args.role_key,
        content_key=args.content_key,
        role_map=args.role_map,
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

    _ = _download_jsonl(config)
    prepare_offline_dataset(config.to_offline_args())


if __name__ == "__main__":
    main()
