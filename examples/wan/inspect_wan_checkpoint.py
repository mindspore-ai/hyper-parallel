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
"""Inspect selected Wan transformer checkpoint tensors.

This script reads the raw safetensors files only. It does not instantiate the
model and does not run any Hyper-Parallel or VeOmni loading code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


DEFAULT_KEYS = (
    "patch_embedding.weight",
    "patch_embedding.bias",
    "condition_embedder.time_embedder.linear_1.weight",
    "condition_embedder.time_embedder.linear_1.bias",
    "blocks.0.attn1.to_q.weight",
)

INDEX_NAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)
WEIGHT_NAMES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
)


@dataclass(frozen=True)
class CheckpointIndex:
    directory: Path
    files_by_key: dict[str, Path]

    def has_key(self, key: str) -> bool:
        return key in self.files_by_key

    def load_tensor(self, key: str) -> torch.Tensor:
        file_path = self.files_by_key[key]
        with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_tensor(key)

    def source_file(self, key: str) -> Path:
        return self.files_by_key[key]


def _resolve_transformer_dir(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir() and (path / "config.json").is_file():
        return path
    transformer_dir = path / "transformer"
    if transformer_dir.is_dir() and (transformer_dir / "config.json").is_file():
        return transformer_dir
    return path


def _index_from_json(directory: Path, index_path: Path) -> CheckpointIndex:
    with index_path.open("r", encoding="utf-8") as handle:
        index_data = json.load(handle)
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index without weight_map: {index_path}")
    files_by_key = {}
    for key, relative_file in weight_map.items():
        file_path = directory / relative_file
        if not file_path.is_file():
            raise FileNotFoundError(f"Shard for {key} does not exist: {file_path}")
        files_by_key[key] = file_path
    return CheckpointIndex(directory=directory, files_by_key=files_by_key)


def _index_single_file(directory: Path, file_path: Path) -> CheckpointIndex:
    with safe_open(str(file_path), framework="pt", device="cpu") as checkpoint:
        files_by_key = {key: file_path for key in checkpoint.keys()}
    return CheckpointIndex(directory=directory, files_by_key=files_by_key)


def _build_checkpoint_index(path: str) -> CheckpointIndex:
    directory = _resolve_transformer_dir(Path(path))
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")

    for index_name in INDEX_NAMES:
        index_path = directory / index_name
        if index_path.is_file():
            return _index_from_json(directory, index_path)

    extra_indexes = sorted(directory.glob("*.safetensors.index.json"))
    if extra_indexes:
        return _index_from_json(directory, extra_indexes[0])

    for weight_name in WEIGHT_NAMES:
        file_path = directory / weight_name
        if file_path.is_file():
            return _index_single_file(directory, file_path)

    safetensors_files = sorted(directory.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return _index_single_file(directory, safetensors_files[0])

    raise FileNotFoundError(
        "No model.safetensors, diffusion_pytorch_model.safetensors, or safetensors index "
        f"found under {directory}"
    )


def _tensor_md5(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    try:
        raw = tensor.view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError, ValueError):
        raw = tensor.float().numpy().tobytes()
    return hashlib.md5(raw).hexdigest()[:8]


def _format_tensor_stats(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    stats_tensor = tensor.float()
    return (
        f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
        f"max={stats_tensor.max().item():.10g} min={stats_tensor.min().item():.10g} "
        f"mean={stats_tensor.mean().item():.10g} norm={stats_tensor.norm().item():.10g} "
        f"md5={_tensor_md5(tensor)}"
    )


def _print_checkpoint(label: str, index: CheckpointIndex, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    print(f"[{label}] directory={index.directory}")
    tensors = {}
    for key in keys:
        if not index.has_key(key):
            print(f"  {key}: MISSING")
            continue
        tensor = index.load_tensor(key)
        tensors[key] = tensor
        source = index.source_file(key).name
        print(f"  {key}: {_format_tensor_stats(tensor)} file={source}")
    return tensors


def _print_comparison(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], keys: tuple[str, ...]) -> None:
    print("[compare]")
    for key in keys:
        if key not in left or key not in right:
            print(f"  {key}: skipped")
            continue
        left_tensor = left[key]
        right_tensor = right[key]
        if tuple(left_tensor.shape) != tuple(right_tensor.shape):
            print(f"  {key}: shape mismatch {tuple(left_tensor.shape)} vs {tuple(right_tensor.shape)}")
            continue
        equal = torch.equal(left_tensor, right_tensor)
        max_abs = (left_tensor.float() - right_tensor.float()).abs().max().item()
        print(f"  {key}: equal={equal} max_abs_diff={max_abs:.10g}")


def _parse_keys(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return DEFAULT_KEYS
    keys = []
    for value in values:
        keys.extend(part.strip() for part in value.split(",") if part.strip())
    return tuple(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or two Wan Diffusers roots or transformer checkpoint directories.",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        help="Checkpoint tensor keys. Values may be space-separated or comma-separated.",
    )
    args = parser.parse_args()
    if len(args.paths) > 2:
        raise SystemExit("Pass at most two paths when comparing checkpoints.")

    keys = _parse_keys(args.keys)
    indexes = [_build_checkpoint_index(path) for path in args.paths]
    loaded = [
        _print_checkpoint(f"path{i}", index, keys)
        for i, index in enumerate(indexes)
    ]
    if len(loaded) == 2:
        _print_comparison(loaded[0], loaded[1], keys)


if __name__ == "__main__":
    main()
