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
"""Pre-cut safetensors Engram tables into sliceable Host row fragments."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _source_files(directory: Path) -> dict[str, Path]:
    """Find logical tensor keys in a standard safetensors checkpoint."""
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return {key: directory / filename for key, filename in payload["weight_map"].items()}
    single = directory / "model.safetensors"
    if not single.is_file():
        raise ValueError("Expected model.safetensors or model.safetensors.index.json")
    with safe_open(str(single), framework="pt", device="cpu") as source:
        return {key: single for key in source.keys()}


def _convert_key(directory: Path, source_file: Path, key: str,
                 chunk_rows: int) -> dict:
    """Read one full source table in bounded row slices and write fragments."""
    fragment_dir = directory / "engram_host_rows"
    fragment_dir.mkdir(exist_ok=True)
    stem = hashlib.sha256(key.encode()).hexdigest()[:16]
    fragments = []
    with safe_open(str(source_file), framework="pt", device="cpu") as source:
        tensor_slice = source.get_slice(key)
        shape = tensor_slice.get_shape()
        if len(shape) != 2:
            raise ValueError(f"Engram table must be rank two: {key}")
        for start in range(0, shape[0], chunk_rows):
            end = min(start + chunk_rows, shape[0])
            part = tensor_slice[start:end]
            if part.dtype != torch.float32:
                raise ValueError(f"Engram Host source must be FP32: {key}")
            relative = Path("engram_host_rows") / f"{stem}_{start:012d}.safetensors"
            target = directory / relative
            if target.exists():
                raise FileExistsError(f"Engram Host fragment already exists: {target}")
            save_file({key: part.contiguous()}, target)
            fragments.append({"start": start, "rows": end - start,
                              "file": str(relative), "key": key})
    return {"shape": list(shape), "fragments": fragments}


def main() -> None:
    """Convert selected logical tables and atomically publish their manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--key", action="append", required=True,
                        help="Logical safetensors key of an Engram embed.weight")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    directory = args.checkpoint_dir.resolve()
    sources = _source_files(directory)
    manifest_path = directory / "engram_host_rows.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or not isinstance(manifest.get("tables"), dict):
            raise ValueError("Invalid existing Engram Host row manifest")
    else:
        manifest = {"format_version": 1, "tables": {}}
    for key in args.key:
        if key not in sources:
            raise KeyError(f"Engram source key is absent: {key}")
        if key in manifest["tables"]:
            raise ValueError(f"Engram source key is already converted: {key}")
        manifest["tables"][key] = _convert_key(directory, sources[key], key, args.chunk_rows)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, manifest_path)


if __name__ == "__main__":
    main()
