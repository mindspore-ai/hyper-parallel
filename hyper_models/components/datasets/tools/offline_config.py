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

"""Configuration for Hugging Face offline dataset preparation."""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


_DatasetDataFiles = Union[
    str,
    List[str],
    Dict[str, Union[str, List[str]]],
]


@dataclass
class OfflinePreparationConfig:
    """Values required to download and preprocess one Hugging Face dataset."""

    dataset_name_or_path: str
    output_prefix: str
    download_dir: Optional[str] = None
    dataset_subset_name: Optional[str] = None
    dataset_split: str = "train"
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[_DatasetDataFiles] = None
    num_proc: Optional[int] = None
    json_keys: Union[str, List[str]] = "text"
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_use_fast: bool = True
    trust_remote_code: bool = False
    chat_template: Optional[str] = None
    add_special_tokens: Optional[List[str]] = None
    split_sentences: bool = False
    keep_newlines: bool = False
    lang: str = "english"
    workers: int = 1
    partitions: int = 1
    append_eod: bool = True
    pad_to_seq_len: Optional[int] = None
    keep_sequential_samples: bool = True
    keep_partition_files: bool = False
    find_optimal_num_workers: bool = False
    workers_to_check: List[int] = field(default_factory=lambda: [16, 32, 64])
    max_documents: int = 100_000
    log_interval: int = 1000

    @staticmethod
    def _slug(value: str) -> str:
        """Normalize a repository, subset, or tokenizer name for file paths."""
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
        if not normalized:
            raise ValueError(f"Could not derive a file name from {value!r}")
        return normalized

    def dataset_label(self) -> str:
        """Derive a local dataset label from the subset or repository ID.

        Returns:
            File-system-safe dataset label.
        """
        source = (
            self.dataset_subset_name
            or self.dataset_name_or_path.rsplit("/", maxsplit=1)[-1]
        )
        return self._slug(source)

    def json_keys_list(self) -> List[str]:
        """Return validated JSON key names as a list.

        Returns:
            JSON keys consumed by the offline encoder.

        Raises:
            ValueError: If no valid JSON key is configured.
        """
        keys = (
            [self.json_keys]
            if isinstance(self.json_keys, str)
            else list(self.json_keys)
        )
        if not keys or any(
            not isinstance(key, str) or not key.strip()
            for key in keys
        ):
            raise ValueError("json_keys must contain one or more column names")
        return keys

    def download_root(self) -> Path:
        """Resolve the root directory for downloaded raw JSONL datasets.

        Returns:
            Custom download directory, or the default
            ``./download_datasets/{dataset_label}/``.
        """
        if self.download_dir and self.download_dir.strip():
            return Path(self.download_dir).expanduser().resolve()
        return (
            Path("download_datasets") / self.dataset_label()
        ).expanduser().resolve()

    def resolved_json_path(self) -> Path:
        """Resolve the configured or derived local JSONL path.

        Returns:
            Local JSONL destination path.
        """
        source_path = Path(self.dataset_name_or_path).expanduser()
        local_suffixes = (".json", ".jsonl", ".json.gz", ".jsonl.gz")
        if source_path.is_dir():
            return source_path.resolve()
        if source_path.name.lower().endswith(local_suffixes):
            resolved_path = source_path.resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Local JSON dataset does not exist: {resolved_path}"
                )
            return resolved_path

        file_name = f"{self.dataset_label()}-{self._slug(self.dataset_split)}.jsonl"
        return self.download_root() / file_name

    def resolved_output_prefix(self) -> Path:
        """Resolve the Megatron output file prefix.

        Returns:
            Output prefix without the ``.bin/.idx`` suffixes.
        """
        if not self.output_prefix.strip():
            raise ValueError("output_prefix must be a non-empty path")
        return Path(self.output_prefix).expanduser().resolve()

    def to_offline_args(self) -> argparse.Namespace:
        """Convert this typed config to offline preprocessing arguments.

        Returns:
            Complete preprocessing arguments for ``prepare_offline_dataset``.
        """
        return argparse.Namespace(
            dataset_name_or_path=str(self.resolved_json_path()),
            output_prefix=str(self.resolved_output_prefix()),
            json_keys=self.json_keys_list(),
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            tokenizer_use_fast=self.tokenizer_use_fast,
            trust_remote_code=self.trust_remote_code,
            chat_template=self.chat_template,
            add_special_tokens=self.add_special_tokens,
            split_sentences=self.split_sentences,
            keep_newlines=self.keep_newlines,
            lang=self.lang,
            append_eod=self.append_eod,
            pad_to_seq_len=self.pad_to_seq_len,
            keep_sequential_samples=self.keep_sequential_samples,
            keep_partition_files=self.keep_partition_files,
            workers=self.workers,
            partitions=self.partitions,
            find_optimal_num_workers=self.find_optimal_num_workers,
            workers_to_check=self.workers_to_check,
            max_documents=self.max_documents,
            log_interval=self.log_interval,
        )


__all__ = ["OfflinePreparationConfig"]
