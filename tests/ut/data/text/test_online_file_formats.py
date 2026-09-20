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
"""Unit tests for local Online Dataset file format resolution."""

import pytest

from hyper_parallel.data.text.online.online_utils import resolve_online_data_files


@pytest.mark.parametrize(
    ("file_name", "expected_format"),
    [
        ("part.json", "json"),
        ("part.jsonl", "json"),
        ("part.json.gz", "json"),
        ("part.jsonl.gz", "json"),
        ("part.csv", "csv"),
        ("part.csv.gz", "csv"),
        ("part.txt", "text"),
        ("part.txt.gz", "text"),
        ("part.parquet", "parquet"),
        ("part.arrow", "arrow"),
    ],
)
def test_resolve_online_data_files_supports_hf_file_formats(tmp_path, file_name, expected_format):
    """Resolve common local HF file extensions to their Datasets builders."""
    data_file = tmp_path / file_name
    data_file.touch()

    files, loader_format = resolve_online_data_files(str(data_file))

    assert files == [str(data_file)]
    assert loader_format == expected_format


def test_resolve_online_data_files_discovers_nested_shards_in_stable_order(tmp_path):
    """Recursively discover supported files and preserve deterministic ordering."""
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "02.parquet"
    first = tmp_path / "01.parquet"
    ignored = nested / "notes.md"
    second.touch()
    first.touch()
    ignored.touch()

    files, loader_format = resolve_online_data_files(str(tmp_path))

    assert files == [str(first), str(second)]
    assert loader_format == "parquet"


def test_resolve_online_data_files_rejects_mixed_formats(tmp_path):
    """Reject mixed source file formats with a stable diagnostic."""
    (tmp_path / "part.jsonl").touch()
    (tmp_path / "part.parquet").touch()

    with pytest.raises(ValueError, match="one supported format"):
        resolve_online_data_files(str(tmp_path))
