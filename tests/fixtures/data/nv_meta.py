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
"""Small real SQLite/tar datasets for nv-meta integration tests."""

from __future__ import annotations

import io
import json
import sqlite3
import tarfile
from pathlib import Path
from typing import Any


def make_nv_meta(
    root: Path,
    records: list[dict[str, bytes]],
    *,
    keys: list[str] | None = None,
    metadata: dict[str, dict[str, Any]] | None = None,
    shard_name: str = "samples.tar",
) -> Path:
    """Write a prepared dataset with actual tar content offsets.

    Args:
        root: Dataset root, normally below a test temporary directory.
        records: One part-name/payload mapping per sample.
        keys: Optional globally unique sample keys, in record order.
        metadata: Optional media entry-key to JSON metadata mapping.
        shard_name: Name of the uncompressed tar shard.

    Returns:
        The dataset root. Reader tests should pass ``cache_dir=root / 'cache'``.
    """
    root = Path(root)
    keys = keys if keys is not None else [f"sample_{index:06d}" for index in range(len(records))]
    if len(keys) != len(records) or len(set(keys)) != len(keys):
        raise ValueError("Fixture keys must be unique and match the record count")
    meta = root / ".nv-meta"
    meta.mkdir(parents=True, exist_ok=True)
    shard = root / shard_name
    shard.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard, "w") as archive:
        for key, record in zip(keys, records):
            for part, payload in record.items():
                member = tarfile.TarInfo(f"{key}.{part}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    with tarfile.open(shard, "r") as archive:
        members = {member.name: member for member in archive.getmembers()}
    connection = sqlite3.connect(meta / "index.sqlite")
    try:
        connection.executescript(
            "CREATE TABLE tar_files (tar_file_id INTEGER PRIMARY KEY, path TEXT NOT NULL);"
            "CREATE TABLE samples (sample_key TEXT UNIQUE NOT NULL, tar_file_id INTEGER NOT NULL, "
            "sample_index INTEGER NOT NULL, byte_size INTEGER NOT NULL, PRIMARY KEY(tar_file_id,sample_index));"
            "CREATE TABLE sample_parts (tar_file_id INTEGER NOT NULL, sample_index INTEGER NOT NULL, "
            "part_name TEXT NOT NULL, content_byte_offset INTEGER NOT NULL, content_byte_size INTEGER NOT NULL, "
            "PRIMARY KEY(tar_file_id,sample_index,part_name));"
            "CREATE TABLE media_metadata (entry_key TEXT PRIMARY KEY, metadata_json TEXT NOT NULL);"
        )
        connection.execute("INSERT INTO tar_files VALUES (?,?)", (0, shard_name))
        for index, (key, record) in enumerate(zip(keys, records)):
            connection.execute("INSERT INTO samples VALUES (?,?,?,?)", (key, 0, index, sum(map(len, record.values()))))
            for part, payload in record.items():
                connection.execute("INSERT INTO sample_parts VALUES (?,?,?,?,?)",
                                   (0, index, part, members[f"{key}.{part}"].offset_data, len(payload)))
        connection.executemany("INSERT INTO media_metadata VALUES (?,?)",
                               ((key, json.dumps(value)) for key, value in (metadata or {}).items()))
        connection.commit()
    finally:
        connection.close()
    (meta / "dataset.yaml").write_text("__module__: megatron.energon\n__class__: StandardWebdatasetFactory\n",
                                        encoding="utf-8")
    (meta / "split.yaml").write_text(json.dumps({"split_parts": {"train": [shard_name], "val": [], "test": []}}),
                                      encoding="utf-8")
    (meta / ".info.json").write_text(json.dumps({"shard_counts": {shard_name: len(records)}}), encoding="utf-8")
    return root
