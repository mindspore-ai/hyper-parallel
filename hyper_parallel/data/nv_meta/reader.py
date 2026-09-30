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
"""SQLite/tar reader for the prepared Megatron-Energon ``.nv-meta`` layout."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import operator
import re
import shutil
import sqlite3
import struct
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import chain, groupby
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from filelock import FileLock

from hyper_parallel.data.constants import ONLINE_SPLIT_NAMES
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.online.source_views import SOURCE_INFO_KEY, SourceInfo


logger = get_dataset_logger(__name__)


def _load_yaml(path: Path) -> Any:
    """Load one metadata YAML file without evaluating Python objects."""
    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except ImportError as error:  # pragma: no cover - dependency diagnostics
        raise ImportError(".nv-meta requires PyYAML to read dataset.yaml/split.yaml") from error
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first present key from a schema-versioned metadata object."""
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _expand_selectors(values: Sequence[str]) -> set[str]:
    """Expand Energon's brace-collapsed shard and sample names once."""
    expanded = set()
    for value in values:
        if "{" in value:
            try:
                from braceexpand import braceexpand  # pylint: disable=import-outside-toplevel
            except ImportError as error:
                raise ImportError("Brace-collapsed nv-meta splits require braceexpand") from error
            expanded.update(braceexpand(value))
        else:
            expanded.add(value)
    return expanded


_CACHE_VERSION = 1
_SAMPLE_DTYPE = np.dtype([
    ("part_start", "<u8"), ("part_count", "<u4"), ("tar_id", "<u4"),
    ("sample_index", "<u8"), ("sample_size", "<u8"), ("key_start", "<u8"),
    ("key_size", "<u4"), ("bytes", "<u8"), ("media_bytes", "<f8"),
    ("pixels", "<f8"), ("frames", "<f8"), ("duration", "<f8"),
])
_PART_DTYPE = np.dtype([
    ("name_id", "<u4"), ("offset", "<u8"), ("size", "<u8"),
    ("metadata_start", "<u8"), ("metadata_size", "<u4"),
])
_SAMPLE_STRUCT = struct.Struct("<QIIQQQIQdddd")
_PART_STRUCT = struct.Struct("<IQQQI")
_MEDIA_METRICS = ("media_bytes", "pixels", "frames", "duration")


@dataclass(frozen=True, slots=True)
class NvMetaPartLocation:
    """Indexed byte range for one sample part, without opening its payload."""

    name: str
    tar_path: str
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class NvMetaSampleIndex:
    """Stable metadata record exposed to generic read schedulers.

    The record contains only index information and optional metadata costs.  A
    caller can build a rank/worker read plan from it and defer payload reads to
    :meth:`NvMetaDataset.__getitem__`.
    """

    logical_index: int
    sample_key: str
    sample_index: int
    tar_file_id: int
    tar_path: str
    sample_size: int
    parts: tuple[NvMetaPartLocation, ...]
    costs: Mapping[str, float] = field(default_factory=dict)


class _TarHandlePool:
    """Bounded per-process tar readers; handles are never shared after fork."""

    def __init__(self, max_open_shards: int = 64) -> None:
        """Bound the process-local cache of open shard readers."""
        if int(max_open_shards) <= 0:
            raise ValueError(".nv-meta max_open_shards must be positive")
        self._pid = os.getpid()
        self._max_open_shards = int(max_open_shards)
        self._handles: OrderedDict[str, Any] = OrderedDict()
        self._sizes: dict[str, int | None] = {}

    def read(self, path: str, offset: int, size: int) -> bytes:
        """Seek directly to an indexed payload and read exactly its bytes."""
        if offset < 0 or size < 0:
            raise ValueError(f".nv-meta payload range must be non-negative: offset={offset} size={size}")
        self._reset_if_forked()
        handle = self._handles.pop(path, None)
        if handle is None:
            if "://" in path and not path.startswith("file://"):
                try:
                    import fsspec  # pylint: disable=import-outside-toplevel
                except ImportError as error:
                    raise ImportError(
                        "Remote .nv-meta shards require the optional fsspec package"
                    ) from error
                handle = fsspec.open(path, "rb", cache_type="none").open()  # pylint: disable=no-member
            else:
                local_path = urlparse(path).path if path.startswith("file://") else path
                handle = open(local_path, "rb")  # pylint: disable=consider-using-with
            try:
                signature = handle.read(6)
                if signature.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd")):
                    raise ValueError("nv-meta byte-range reads require uncompressed tar shards")
            except Exception:
                handle.close()
                raise
            try:
                file_size = os.fstat(handle.fileno()).st_size
            except (AttributeError, OSError):
                file_size = getattr(handle, "size", None)
            self._sizes[path] = file_size
        self._handles[path] = handle
        while len(self._handles) > self._max_open_shards:
            stale_path, stale_handle = self._handles.popitem(last=False)
            stale_handle.close()
            self._sizes.pop(stale_path, None)
        file_size = self._sizes[path]
        if file_size is not None and offset + size > file_size:
            raise IOError(f".nv-meta payload range exceeds shard: {path} offset={offset} size={size}")
        handle.seek(offset)
        payload = handle.read(size)
        if len(payload) != size:
            raise IOError(f".nv-meta payload truncated: {path} offset={offset} size={size}")
        return payload

    def close(self) -> None:
        """Close all process-local handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles = OrderedDict()
        self._sizes = {}

    def _reset_if_forked(self) -> None:
        process_id = os.getpid()
        if process_id != self._pid:
            self.close()
            self._pid = process_id


class NvMetaDataset:
    """Read prepared samples by stable index; Online views own ordering and sharding.

    The prepared index is compiled once into a fingerprinted, read-only cache.
    Ranks and workers map that cache on first access; neither pickling nor
    construction copies the sample index into the Python heap. Payload handles
    and the decoded metadata LRU belong to each process.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        split: str = "train",
        exclude: Sequence[str] | None = None,
        required_parts: str | Sequence[str] | None = None,
        metadata_cache_size: int = 4096,
        max_open_shards: int = 64,
        cache_dir: str | os.PathLike[str] | None = None,
        cache_timeout: float = 600,
        read_buffer_size: int = 8 * 1024 * 1024,
    ) -> None:
        """Configure prepared metadata and bounded caches; log effective options at DEBUG.

        ``required_parts`` accepts a part name or sequence; None reads all parts.
        Cache and handle limits apply per reader in each process. ``cache_timeout``
        is the index-cache lock wait, not a payload read timeout. ``read_buffer_size``
        bounds coalescing of adjacent reads, not an individual part's size.
        """
        self.root = self._resolve_root(path)
        self.dataset_root = self.root.parent if self.root.name == ".nv-meta" else self.root
        self.split = self._normalize_split(split)
        if isinstance(required_parts, str):
            required_parts = (required_parts,)
        self.required_parts = (
            {part.lstrip(".") for part in required_parts}
            if required_parts is not None
            else None
        )
        if self.required_parts is not None and (not self.required_parts or "" in self.required_parts):
            raise ValueError("required_parts must contain non-empty part names")
        if int(metadata_cache_size) < 0:
            raise ValueError(".nv-meta metadata_cache_size must be non-negative")
        self.metadata_cache_size = int(metadata_cache_size)
        self.max_open_shards = int(max_open_shards)
        self._handles = _TarHandlePool(self.max_open_shards)
        self.read_buffer_size = int(read_buffer_size)
        cache_timeout = float(cache_timeout)
        if self.read_buffer_size <= 0 or not 0 < cache_timeout < math.inf:
            raise ValueError("read_buffer_size and cache_timeout must be positive")
        self._samples = None
        self._parts = None
        self._keys = None
        self._media = None
        self._media_metadata_cache: OrderedDict[int, Mapping[str, Any]] = OrderedDict()
        self._metadata = self._load_metadata()
        self._shard_paths = self._resolve_split_shards(self.split)
        self._shard_patterns = self._compile_split_patterns(self._shard_paths)
        excluded = [exclude] if isinstance(exclude, str) else (exclude or ())
        self._excluded = _expand_selectors(excluded) | self._resolve_split_excludes(self.split)
        self._index_sqlite_path = self._sqlite_path()
        default_cache = Path(os.environ.get("XDG_CACHE_HOME", os.environ.get("LOCALAPPDATA", Path.home() / ".cache")))
        cache_root = Path(cache_dir) if cache_dir is not None else default_cache / "hyper_parallel" / "nv_meta"
        cache_root = cache_root.expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        fingerprint = self._cache_fingerprint()
        self.fingerprint = fingerprint
        self._cache_path = cache_root / fingerprint
        self._manifest = self._open_cache(cache_timeout)
        self._part_names = tuple(self._manifest["part_names"])
        self._shard_map = {int(key): value for key, value in self._manifest["shards"].items()}
        self._has_media_metadata = self._manifest["has_media_metadata"]
        logger.debug(
            "nv-meta reader: path=%s split=%s required_parts=%s metadata_cache_size=%d "
            "max_open_shards=%d cache_dir=%s cache_timeout=%s read_buffer_size=%d",
            self.dataset_root, self.split, self.required_parts,
            self.metadata_cache_size, self.max_open_shards, self._cache_path.parent, cache_timeout,
            self.read_buffer_size,
        )

    def _open_cache(self, timeout: float) -> dict[str, Any]:
        """Serialize cache writers, then load only the small immutable manifest."""
        cache_root = self._cache_path.parent
        if not (self._cache_path / "manifest.json").is_file():
            with FileLock(str(cache_root / f"{self.fingerprint}.lock"), timeout=timeout):
                if not (self._cache_path / "manifest.json").is_file():
                    self._build_cache(cache_root, self.fingerprint)
        with (self._cache_path / "manifest.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _compile_split_patterns(selectors: set[str]) -> list[re.Pattern[str]]:
        """Keep literal shard selection constant-time; compile only real patterns."""
        patterns = []
        for selector in selectors:
            if any(character in selector for character in "*?[]()^$|+\\"):
                try:
                    patterns.append(re.compile(selector))
                except re.error:
                    patterns.append(re.compile(fnmatch.translate(selector)))
        return patterns

    @staticmethod
    def _resolve_root(path: str | os.PathLike[str]) -> Path:
        if isinstance(path, str) and "://" in path:
            raise ValueError(
                "Remote .nv-meta URLs require a local metadata cache; "
                "materialize the index directory before building this provider"
            )
        root = Path(path).expanduser().resolve()
        if root.is_file() and root.suffix == ".nv-meta":
            root = root.parent
        elif root.is_dir() and (root / ".nv-meta").is_dir() and not (root / "dataset.yaml").exists():
            root = root / ".nv-meta"
        if not root.is_dir():
            raise FileNotFoundError(f".nv-meta directory does not exist: {root}")
        return root

    @staticmethod
    def _normalize_split(split: str) -> str:
        normalized = {"val": "valid", "validation": "valid"}.get(str(split), str(split))
        if normalized not in ONLINE_SPLIT_NAMES:
            raise ValueError(f"Unsupported .nv-meta split {split!r}; expected {ONLINE_SPLIT_NAMES!r}")
        return normalized

    def _load_metadata(self) -> dict[str, Any]:
        dataset_path = self.root / "dataset.yaml"
        split_path = self.root / "split.yaml"
        if not split_path.exists():
            split_path = self.root / "split.json"
        info_path = self.root / ".info.json"
        if not info_path.exists():
            info_path = self.root / ".info.yaml"
        if not dataset_path.exists() or not split_path.exists() or not info_path.exists():
            missing = [str(path.name) for path in (dataset_path, split_path, info_path) if not path.exists()]
            raise FileNotFoundError(f".nv-meta is missing required metadata: {', '.join(missing)}")
        info = _load_yaml(info_path)
        dataset = _load_yaml(dataset_path)
        split = _load_yaml(split_path)
        if not isinstance(dataset, Mapping) or not isinstance(split, Mapping) or not isinstance(info, Mapping):
            raise ValueError(".nv-meta metadata files must contain mappings")
        return {"dataset": dict(dataset), "split": dict(split), "info": dict(info)}

    def _resolve_split_shards(self, split: str) -> set[str]:
        split_doc = self._metadata["split"]
        aliases = ("valid", "val", "validation") if split == "valid" else (split,)
        value = _first(split_doc, *aliases)
        if isinstance(value, Mapping):
            value = _first(value, "shards", "files", "paths", default=[])
        if value is None:
            # Some manifests use a top-level ``splits`` mapping.
            splits = split_doc.get("splits")
            if isinstance(splits, Mapping):
                value = _first(splits, *aliases)
        if value is None:
            # Energon's prepared format uses ``split_parts`` for the shard
            # lists and reserves ``exclude`` for whole shards/samples.
            split_parts = split_doc.get("split_parts")
            if isinstance(split_parts, Mapping):
                value = _first(split_parts, *aliases)
        if value is None:
            return set()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, Sequence):
            raise ValueError(f".nv-meta split {split!r} must list shard paths")
        names = set()
        for item in value:
            if isinstance(item, Mapping):
                item = _first(item, "path", "name", "shard", "tar")
            if item is not None:
                names.add(self._normalize_shard_name(str(item)))
        return _expand_selectors(names)

    def _resolve_split_excludes(self, split: str) -> set[str]:
        split_doc = self._metadata["split"]
        excludes = split_doc.get("exclude", split_doc.get("excludes", {}))
        if isinstance(excludes, Mapping):
            aliases = ("valid", "val", "validation") if split == "valid" else (split,)
            excludes = _first(excludes, *aliases, default=[])
        if isinstance(excludes, str):
            excludes = [excludes]
        if not isinstance(excludes, Sequence):
            return set()
        return _expand_selectors([str(item).lstrip("/") for item in excludes])

    @staticmethod
    def _normalize_shard_name(value: str) -> str:
        return value.replace("\\", "/").lstrip("./")

    def _resolve_shard_path(self, value: str) -> str:
        if "://" in value and not value.startswith("file://"):
            return value
        if value.startswith("file://"):
            value = urlparse(value).path
        candidate = (self.root / value).resolve()
        if not candidate.exists() and self.dataset_root != self.root:
            candidate = (self.dataset_root / value).resolve()
        try:
            candidate.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(f".nv-meta shard escapes dataset root: {value!r}") from error
        return str(candidate)

    def _sqlite_path(self) -> Path:
        configured = _first(self._metadata["info"], "index", "index_sqlite", "index_path")
        candidates = [self.root / str(configured)] if configured else []
        candidates.extend((self.root / "index.sqlite", self.root / "index.db"))
        candidates.extend(sorted(self.root.glob("*.sqlite")))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f".nv-meta index.sqlite not found under {self.root}")

    def _cache_fingerprint(self) -> str:
        """Identify one immutable prepared index and its selected view."""
        stat = self._index_sqlite_path.stat()
        wal = Path(str(self._index_sqlite_path) + "-wal")
        if wal.exists() and wal.stat().st_size:
            raise ValueError("Checkpoint the prepared SQLite WAL before constructing an nv-meta reader")
        identity = {
            "version": _CACHE_VERSION, "index": str(self._index_sqlite_path),
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "metadata": self._metadata, "split": self.split,
            "manifest_files": {
                name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                for name in ("dataset.yaml", "split.yaml", "split.json", ".info.json", ".info.yaml", "index.uuid")
                if (self.root / name).is_file()
            },
            "exclude": sorted(self._excluded),
            "parts": None if self.required_parts is None else sorted(self.required_parts),
        }
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def _build_cache(self, cache_root: Path, fingerprint: str) -> None:
        """Stream a new index under the writer lock and publish it atomically."""
        temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}-", dir=cache_root))
        try:
            manifest = self._compile_cache(temporary)
            if self._cache_fingerprint() != fingerprint:
                raise RuntimeError("Prepared metadata changed while building the nv-meta cache")
            with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._cache_path)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _compile_cache(self, directory: Path) -> dict[str, Any]:
        """Write sample, part, key and metadata files with bounded build memory."""
        with ExitStack() as stack:
            connection = sqlite3.connect(self._index_sqlite_path.as_uri() + "?mode=ro", uri=True)
            stack.callback(connection.close)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-8192")
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"samples", "sample_parts"}.issubset(tables):
                raise ValueError("nv-meta requires samples and sample_parts tables with payload offsets")
            columns = {
                table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
                for table in ("samples", "sample_parts", "media_metadata")
            }
            if not {"sample_index", "tar_file_id"}.issubset(columns["samples"]):
                raise ValueError("nv-meta samples must contain sample_index and tar_file_id")
            if not {"sample_index", "tar_file_id", "part_name"}.issubset(columns["sample_parts"]):
                raise ValueError("nv-meta sample_parts must contain sample_index, tar_file_id and part_name")
            shards = self._load_shard_map(connection, tables)
            streams = {name: stack.enter_context((directory / f"{name}.bin").open("wb"))
                       for name in ("samples", "parts", "keys", "metadata")}
            manifest = {
                "samples": 0, "parts": 0, "part_names": [], "shards": shards,
                "has_media_metadata": {"entry_key", "metadata_json"}.issubset(columns["media_metadata"]),
            }
            names: dict[str, int] = {}
            for tar_id, tar_path in sorted(shards.items()):
                shard_name = self._selected_shard_name(tar_path)
                if shard_name is None:
                    continue
                rows = self._cache_rows(connection, columns, tar_id)
                self._compile_shard(rows, tar_id, tar_path, shard_name, streams, manifest, names)
            manifest["part_names"] = list(names)
            for stream in streams.values():
                stream.flush()
                os.fsync(stream.fileno())
            manifest["file_sizes"] = {name: stream.tell() for name, stream in streams.items()}
            return manifest

    def _selected_shard_name(self, tar_path: str) -> str | None:
        """Resolve split aliases once per shard, before executing sample queries."""
        if "://" in tar_path and not tar_path.startswith("file://"):
            url_path = self._normalize_shard_name(urlparse(tar_path).path)
            pieces = url_path.split("/")
            name = pieces[-1]
            aliases = {"/".join(pieces[start:]) for start in range(len(pieces))}
        else:
            local_path = urlparse(tar_path).path if tar_path.startswith("file://") else tar_path
            name = self._normalize_shard_name(os.path.relpath(local_path, self.dataset_root))
            aliases = set()
        aliases.update((name, self._normalize_shard_name(tar_path), name.rsplit("/", 1)[-1]))
        if not self._matches_split_shard(aliases) or aliases.intersection(self._excluded):
            return None
        return name

    @staticmethod
    def _cache_rows(connection: sqlite3.Connection, columns: dict[str, set[str]], tar_id: int) -> Any:
        """Join the selected shard using structural sample and part identities."""
        key_column = "sample_key" if "sample_key" in columns["samples"] else "key"
        if key_column not in columns["samples"]:
            raise ValueError("nv-meta samples must contain sample_key")
        offset, size = "content_byte_offset", "content_byte_size"
        if not {offset, size}.issubset(columns["sample_parts"]):
            offset, size = "byte_offset", "byte_size"
        if not {offset, size}.issubset(columns["sample_parts"]):
            raise ValueError("nv-meta sample_parts has no supported payload byte offsets")
        sample_size = "s.byte_size" if "byte_size" in columns["samples"] else "NULL"
        has_metadata = {"entry_key", "metadata_json"}.issubset(columns["media_metadata"])
        metadata = "m.metadata_json" if has_metadata else "NULL"
        media_join = ""
        if has_metadata:
            media_join = (
                f" LEFT JOIN media_metadata m ON m.entry_key = "
                f"s.{key_column} || '.' || ltrim(p.part_name, '.')"
            )
            if "tar_file_id" in columns["media_metadata"]:
                media_join += " AND m.tar_file_id=s.tar_file_id"
        return connection.execute(
            f"SELECT s.sample_index, s.{key_column} AS sample_key, {sample_size} AS sample_size, "
            f"p.part_name, p.{offset} AS offset, p.{size} AS size, {metadata} AS metadata_json "
            "FROM samples s LEFT JOIN sample_parts p "
            "ON p.tar_file_id=s.tar_file_id AND p.sample_index=s.sample_index"
            + media_join + " WHERE s.tar_file_id=? ORDER BY s.sample_index, p." + offset,
            (tar_id,),
        )

    def _compile_shard(
        self, rows: Any, tar_id: int, tar_path: str, shard_name: str,
        streams: Mapping[str, Any], manifest: dict[str, Any], names: dict[str, int],
    ) -> None:
        """Append a shard, retaining at most one sample's metadata."""
        for sample_index, sample_rows in groupby(rows, key=lambda row: row["sample_index"]):
            first = next(sample_rows)
            key = str(first["sample_key"])
            if f"{shard_name}/{key}" in self._excluded:
                continue
            key_bytes = key.encode("utf-8")
            key_start = streams["keys"].tell()
            streams["keys"].write(key_bytes)
            part_start = manifest["parts"]
            costs = [0.0] * 4
            byte_cost = 0
            full_size = 0
            seen = set()
            for row in chain((first,), sample_rows):
                if row["part_name"] is None:
                    continue
                part_name = str(row["part_name"]).lstrip(".")
                offset, size = self._validate_range(int(row["offset"]), int(row["size"]), tar_path)
                if part_name in seen:
                    raise ValueError(f"Duplicate nv-meta part or media metadata for {shard_name}/{key}.{part_name}")
                seen.add(part_name)
                full_size += size
                if self.required_parts is not None and part_name not in self.required_parts:
                    continue
                name_id = names.setdefault(part_name, len(names))
                metadata_bytes = (row["metadata_json"] or "").encode("utf-8")
                metadata_start = streams["metadata"].tell()
                streams["metadata"].write(metadata_bytes)
                streams["parts"].write(_PART_STRUCT.pack(name_id, offset, size, metadata_start, len(metadata_bytes)))
                manifest["parts"] += 1
                byte_cost += size
                part_costs = self._media_costs(metadata_bytes, size)
                costs = [left + right for left, right in zip(costs, part_costs)]
            if not seen:
                raise ValueError(f"nv-meta sample {shard_name}/{key} contains no indexed parts")
            streams["samples"].write(_SAMPLE_STRUCT.pack(
                part_start, manifest["parts"] - part_start, tar_id, int(sample_index),
                full_size if first["sample_size"] is None else int(first["sample_size"]),
                key_start, len(key_bytes), byte_cost, *costs,
            ))
            manifest["samples"] += 1

    @staticmethod
    def _media_costs(payload: bytes, part_size: int) -> tuple[float, float, float, float]:
        """Reduce metadata to finite scalar costs while compiling the index."""
        if not payload:
            return 0.0, 0.0, 0.0, 0.0
        metadata = json.loads(payload)
        if not isinstance(metadata, Mapping):
            raise ValueError("nv-meta media_metadata must contain a JSON object")
        width = float(metadata.get("width", metadata.get("video_width", 0)))
        height = float(metadata.get("height", metadata.get("video_height", 0)))
        if any(not math.isfinite(value) or value < 0 for value in (width, height)):
            raise ValueError("nv-meta media dimensions must be finite and non-negative")
        values = (
            metadata.get("media_bytes", part_size),
            width * height,
            _first(metadata, "video_num_frames", "num_frames", "frames", default=0),
            _first(metadata, "video_duration", "audio_duration", "duration", default=0),
        )
        costs = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or value < 0 for value in costs):
            raise ValueError("nv-meta media costs must be finite and non-negative")
        return costs

    def _ensure_index(self) -> None:
        """Map immutable files on first access, including after worker spawn."""
        if self._samples is not None:
            return
        for name, dtype in (("samples", _SAMPLE_DTYPE), ("parts", _PART_DTYPE)):
            if self._manifest["file_sizes"][name] != self._manifest[name] * dtype.itemsize:
                raise ValueError(f"Corrupt nv-meta index cache count: {name}")
        mapped = {}
        for name, dtype in (("samples", _SAMPLE_DTYPE), ("parts", _PART_DTYPE),
                            ("keys", np.dtype("u1")), ("metadata", np.dtype("u1"))):
            path = self._cache_path / f"{name}.bin"
            size = self._manifest["file_sizes"][name]
            if path.stat().st_size != size or size % dtype.itemsize:
                for value in mapped.values():
                    if isinstance(value, np.memmap):
                        value._mmap.close()
                raise ValueError(f"Corrupt nv-meta index cache: {path}")
            mapped[name] = np.memmap(path, dtype=dtype, mode="r") if size else np.empty(0, dtype=dtype)
        self._samples, self._parts = mapped["samples"], mapped["parts"]
        self._keys, self._media = mapped["keys"], mapped["metadata"]
    @staticmethod
    def _validate_range(offset: int, size: int, tar_path: str) -> tuple[int, int]:
        """Reject malformed SQLite ranges before a worker opens a shard."""
        if offset < 0 or size < 0:
            raise ValueError(
                f".nv-meta has a negative payload range for {tar_path}: "
                f"offset={offset} size={size}"
            )
        return offset, size

    def _matches_split_shard(self, shard_names: set[str]) -> bool:
        """Match exact names and Energon's regex-style ``split_parts`` selectors."""
        return bool(self._shard_paths.intersection(shard_names)) or any(
            pattern.fullmatch(name) for pattern in self._shard_patterns for name in shard_names
        )

    def _load_shard_map(self, connection: sqlite3.Connection, tables: set[str]) -> dict[int, str]:
        shard_map: dict[int, str] = {}
        if "tar_files" in tables:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(tar_files)")}
            id_column = "tar_file_id" if "tar_file_id" in columns else "id"
            path_column = next(
                (name for name in ("path", "name", "tar_path", "filename", "tar_file_name") if name in columns),
                None,
            )
            if path_column is not None:
                for row in connection.execute(f"SELECT {id_column}, {path_column} FROM tar_files"):
                    shard_map[int(row[0])] = self._resolve_shard_path(str(row[1]))
        if shard_map:
            return shard_map
        for tar_id, name in self._manifest_shard_entries():
            shard_map[tar_id] = self._resolve_shard_path(name)
        if not shard_map:
            raise ValueError(".nv-meta index has no tar_files table or shard paths in .info.json")
        return shard_map

    def _manifest_shard_entries(self) -> Iterator[tuple[int, str]]:
        """Resolve metadata layouts while preserving the prepared tar ID order."""
        info_shards = _first(self._metadata["info"], "shards", "tar_files", default=[])
        if not info_shards:
            shard_counts = self._metadata["info"].get("shard_counts", {})
            if isinstance(shard_counts, Mapping):
                # Energon's .info.json records counts keyed by shard name; the
                # insertion order is the SQLite tar_file_id order.
                info_shards = list(shard_counts.keys())
        if isinstance(info_shards, Mapping):
            for key, value in info_shards.items():
                yield int(key), str(value)
            return
        if not info_shards:
            split_doc = self._metadata["split"]
            info_shards = split_doc.get("shards", split_doc.get("files", []))
        if isinstance(info_shards, Sequence) and not isinstance(info_shards, (str, bytes)):
            for index, value in enumerate(info_shards):
                path = value.get("path", value.get("name")) if isinstance(value, Mapping) else value
                if path is not None:
                    yield index, str(path)

    def __len__(self) -> int:
        """Return the selected sample count without mapping the index."""
        return self._manifest["samples"]

    def _normalize_index(self, index: int) -> int:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("nv-meta sample index out of range")
        self._ensure_index()
        return index

    def _sample_key(self, index: int) -> str:
        sample = self._samples[index]
        start, size = int(sample["key_start"]), int(sample["key_size"])
        return self._keys[start:start + size].tobytes().decode("utf-8")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read selected parts, coalescing nearby byte ranges within a budget."""
        index = self._normalize_index(index)
        sample = self._samples[index]
        sample_key = self._sample_key(index)
        tar_path = self._shard_map[int(sample["tar_id"])]
        start, count = int(sample["part_start"]), int(sample["part_count"])
        parts = self._read_parts(tar_path, self._parts[start:start + count])
        metadata = {}
        for part_index in range(start, start + count):
            part_metadata = self._load_part_metadata(part_index)
            if part_metadata:
                name = self._part_names[int(self._parts[part_index]["name_id"])]
                metadata[name] = part_metadata
        record = dict(parts)
        record.update(parts=parts, sample_key=sample_key, __key__=sample_key)
        record[SOURCE_INFO_KEY] = SourceInfo(
            provider="nv-meta", sample_key=sample_key, source_path=str(self.root), shard_path=tar_path,
        )
        if metadata:
            record["metadata"] = {"media": metadata}
        return record

    def _read_parts(self, tar_path: str, locations: np.ndarray) -> dict[str, bytes]:
        """Merge small adjacent reads without caching entire remote shards."""
        parts = {}
        position = 0
        while position < len(locations):
            first = locations[position]
            begin = int(first["offset"])
            end = begin + int(first["size"])
            stop = position + 1
            while stop < len(locations):
                candidate = locations[stop]
                candidate_start = int(candidate["offset"])
                candidate_end = candidate_start + int(candidate["size"])
                if candidate_start - end > 65536 or candidate_end - begin > self.read_buffer_size:
                    break
                end = max(end, candidate_end)
                stop += 1
            payload = self._handles.read(tar_path, begin, end - begin)
            for location in locations[position:stop]:
                name = self._part_names[int(location["name_id"])]
                offset = int(location["offset"]) - begin
                size = int(location["size"])
                parts[name] = payload if offset == 0 and size == len(payload) else payload[offset:offset + size]
            position = stop
        return parts

    def cost_for_index(self, index: int, metric: str | Mapping[str, float]) -> float:
        """Read precomputed scalar costs without scanning metadata or payloads."""
        index = self._normalize_index(index)
        if isinstance(metric, Mapping):
            total = 0.0
            for name, weight in metric.items():
                weight = float(weight)
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError("Read cost weights must be finite and non-negative")
                if weight:
                    total += weight * self.cost_for_index(index, name)
            return total
        normalized = str(metric).lower()
        if normalized in ("sample", "samples", "count"):
            return 1.0
        if normalized not in ("bytes", *_MEDIA_METRICS):
            raise ValueError(f"Unsupported nv-meta read cost {metric!r}")
        if normalized != "bytes" and not self._has_media_metadata:
            raise ValueError("Media read costs require media_metadata; use the bytes metric instead")
        return float(self._samples[index][normalized])

    def index_metadata(self, index: int, *, metrics: Sequence[str] = ()) -> NvMetaSampleIndex:
        """Return one sample's locations and costs without payload I/O."""
        index = self._normalize_index(index)
        sample = self._samples[index]
        tar_id = int(sample["tar_id"])
        start, count = int(sample["part_start"]), int(sample["part_count"])
        parts = tuple(
            NvMetaPartLocation(self._part_names[int(part["name_id"])], self._shard_map[tar_id],
                               int(part["offset"]), int(part["size"]))
            for part in self._parts[start:start + count]
        )
        return NvMetaSampleIndex(
            logical_index=index, sample_key=self._sample_key(index), sample_index=int(sample["sample_index"]),
            tar_file_id=tar_id, tar_path=self._shard_map[tar_id], sample_size=int(sample["sample_size"]),
            parts=parts, costs={metric: self.cost_for_index(index, metric) for metric in metrics},
        )

    def __getstate__(self) -> dict[str, Any]:
        """Serialize cache references; never copy mmap contents to a worker."""
        state = dict(self.__dict__)
        for name in ("_samples", "_parts", "_keys", "_media", "_handles"):
            state[name] = None
        state["_media_metadata_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(state)
        self._handles = _TarHandlePool(self.max_open_shards)

    def close(self) -> None:
        """Release this process's file handles and read-only mappings."""
        self._handles.close()
        self._media_metadata_cache.clear()
        for name in ("_samples", "_parts", "_keys", "_media"):
            value = getattr(self, name)
            if isinstance(value, np.memmap):
                value._mmap.close()
            setattr(self, name, None)

    def _load_part_metadata(self, part_index: int) -> Mapping[str, Any]:
        if part_index in self._media_metadata_cache:
            value = self._media_metadata_cache.pop(part_index)
            self._media_metadata_cache[part_index] = value
            return value
        part = self._parts[part_index]
        start, size = int(part["metadata_start"]), int(part["metadata_size"])
        if not size:
            return {}
        value = json.loads(self._media[start:start + size].tobytes())
        if self.metadata_cache_size:
            self._media_metadata_cache[part_index] = value
            while len(self._media_metadata_cache) > self.metadata_cache_size:
                self._media_metadata_cache.popitem(last=False)
        return value
