# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Regression tests for local Hugging Face source format normalization."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hyper_parallel.data.tools.huggingface_offline import (
    _infer_local_hf_builder,
    _infer_text_format,
    _materialize_local_files,
    _resolve_local_files,
)
from hyper_parallel.data.tools.offline_config import OfflinePreparationConfig


@pytest.mark.parametrize(
    ("file_name", "expected_builder"),
    [
        ("part.json", "json"),
        ("part.jsonl.gz", "json"),
        ("part.csv", "csv"),
        ("part.csv.gz", "csv"),
        ("part.parquet", "parquet"),
        ("part.arrow", "arrow"),
        ("part.txt", "text"),
        ("part.txt.gz", "text"),
    ],
)
def test_infer_local_huggingface_builder(file_name, expected_builder):
    """Infer Datasets builders for supported local source files."""
    assert _infer_local_hf_builder(Path(file_name)) == expected_builder


def test_resolve_local_files_recursively_and_in_stable_order(tmp_path):
    """Discover supported files recursively while ignoring unrelated files."""
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "01.parquet"
    second = nested / "02.parquet"
    first.touch()
    second.touch()
    (nested / "notes.md").touch()

    assert _resolve_local_files(tmp_path) == [first.resolve(), second.resolve()]


def test_materialize_local_parquet_through_datasets(tmp_path):
    """Normalize a tabular source before indexed tokenization."""
    source = tmp_path / "train.parquet"
    source.touch()
    output = tmp_path / "normalized" / "train.jsonl"
    config = OfflinePreparationConfig(
        dataset_name_or_path=str(source),
        output_prefix=str(tmp_path / "indexed" / "train"),
        tokenizer_name_or_path="test-tokenizer",
        json_keys=["text"],
    )
    dataset = Mock(column_names=["text", "metadata"])
    selected = Mock()
    dataset.select_columns.return_value = selected

    with patch("datasets.load_dataset", return_value=dataset) as load_dataset:
        assert _materialize_local_files(config, [source], output) == output

    load_dataset.assert_called_once_with(
        "parquet",
        data_files={"train": [str(source.resolve())]},
        split="train",
        cache_dir=None,
    )
    dataset.select_columns.assert_called_once_with(["text"])
    selected.to_json.assert_called_once_with(
        str(output), orient="records", lines=True, force_ascii=False,
    )


def test_materialize_local_files_rejects_mixed_formats(tmp_path):
    """Reject mixed source formats in one conversion call."""
    json_file = tmp_path / "part.jsonl"
    csv_file = tmp_path / "part.csv"
    json_file.touch()
    csv_file.touch()
    config = OfflinePreparationConfig(
        dataset_name_or_path=str(tmp_path),
        output_prefix=str(tmp_path / "out"),
        tokenizer_name_or_path="test-tokenizer",
    )

    with pytest.raises(ValueError, match="one format"):
        _materialize_local_files(config, [json_file, csv_file], tmp_path / "out.jsonl")
