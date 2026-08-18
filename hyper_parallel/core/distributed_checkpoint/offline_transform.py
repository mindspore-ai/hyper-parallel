# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
"""Offline checkpoint conversion: Hugging Face safetensors layout and HyperParallel DCP.

Covers **full checkpoint** (unsharded weights) and **shard checkpoint** (e.g. HF multi-file safetensors),
``state_dict`` validation, HF-style save/load, and DCP read/write (see ``.agent/rules/offline_transform.md``).

Public symbols are listed in ``__all__``; sharding helpers and split types are internal.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader, FileSystemWriter
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner, _DcpMergeLoadPlanner
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

logger = logging.getLogger(__name__)

__all__ = [
    "save_state_dict_as_huggingface_format",
    "parse_checkpoint_from_huggingface",
    "full_state_dict_to_dcp_format",
    "dcp_to_full_state_dict",
    "convert_full_checkpoint_to_dcp",
]


def _validate_state_dict_for_active_platform(state_dict: dict[str, Any]) -> None:
    """Ensure values are allowed for the active platform before DCP / HF export.

    Applies to a full-weights ``state_dict`` (from a full checkpoint file or merged shard checkpoint content).

    PyTorch: ``torch.Tensor`` (including ``nn.Parameter``), ``bytes`` (e.g. optimizer blobs), or ``str`` placeholders
    (e.g. bitsandbytes serialization per HF ecosystem).
    MindSpore: tensor-like values only (``Parameter`` / ``Tensor``).

    Args:
        state_dict: Flat or nested mapping rejected here if values are not allowed types.

    Raises:
        TypeError: If a value has an unsupported type for the current platform.
    """
    if not state_dict:
        return
    platform = get_platform()
    if platform.platform_type == PlatformType.PYTORCH:
        import torch  # pylint: disable=import-outside-toplevel

        for key, value in state_dict.items():
            if isinstance(value, (torch.Tensor, bytes, str)):
                continue
            logger.warning(
                "Unsupported PyTorch offline checkpoint value type for key %r: %s "
                "(expected torch.Tensor, bytes, or str per project rules).",
                key,
                type(value).__name__,
            )
            raise TypeError(
                f"PyTorch offline checkpoint expects torch.Tensor, bytes, or str per key; "
                f"got {type(value).__name__} for key {key!r}."
            )
    elif platform.platform_type == PlatformType.MINDSPORE:
        for key, value in state_dict.items():
            if platform.is_tensor(value):
                continue
            logger.warning(
                "Unsupported MindSpore offline checkpoint value type for key %r: %s "
                "(expected tensor-like values per project rules).",
                key,
                type(value).__name__,
            )
            raise TypeError(
                f"MindSpore offline checkpoint expects tensor-like values per key; "
                f"got {type(value).__name__} for key {key!r}."
            )


_SIZE_UNITS = {
    "TB": 10**12,
    "GB": 10**9,
    "MB": 10**6,
    "KB": 10**3,
}

_SAFE_WEIGHTS_NAME = "model.safetensors"
_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
# Hugging Face-style shard filenames: ``model.safetensors`` or ``model-00001-of-00002.safetensors``.
_HF_SAFE_WEIGHTS_FILENAME_PATTERN = _SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors")


@dataclass
class _StateDictSplit:
    """Result of splitting a state dict into named shard files and an optional weight map."""

    metadata: dict[str, Any]
    filename_to_tensors: dict[str, list[str]]
    tensor_to_filename: dict[str, str]
    is_sharded: bool = False

    def __post_init__(self) -> None:
        self.is_sharded = len(self.filename_to_tensors) > 1


def _parse_size_to_int(size_as_str: str) -> int:
    """
    Parse a size expressed as a string with digits and unit (like `"5MB"`) to an integer (in bytes).

    Supported units are "TB", "GB", "MB", "KB".

    Args:
        size_as_str (`str`): The size to convert. Will be directly returned if an `int`.

    Example:

    ```py
    >>> _parse_size_to_int("5MB")
    5000000
    ```
    """
    size_as_str = size_as_str.strip()

    # Parse unit
    unit = size_as_str[-2:].upper()
    if unit not in _SIZE_UNITS:
        raise ValueError(f"Unit '{unit}' not supported. Supported units are TB, GB, MB, KB. Got '{size_as_str}'.")
    multiplier = _SIZE_UNITS[unit]

    # Parse value
    try:
        value = float(size_as_str[:-2].strip())
    except ValueError as e:
        raise ValueError(f"Could not parse the size value from '{size_as_str}': {e}") from e

    return int(value * multiplier)


def _build_shard_list_for_safetensors(
    state_dict: dict[str, Any],
    max_cap: int,
    get_storage_size: Any,
) -> list[dict[str, Any]]:
    """Assign tensors to shards in key order (greedy by ``max_cap`` bytes)."""
    shard_list: list[dict[str, Any]] = []
    current_shard: dict[str, Any] = {}
    current_shard_size = 0

    for key, tensor in state_dict.items():
        tensor_size = get_storage_size(tensor)

        if tensor_size > max_cap:
            shard_list.append({key: tensor})
            continue

        if current_shard_size + tensor_size > max_cap:
            shard_list.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        current_shard[key] = tensor
        current_shard_size += tensor_size

    if len(current_shard) > 0:
        shard_list.append(current_shard)
    return shard_list


def _total_bytes_unique_keys_in_shards(
    shard_list: list[dict[str, Any]],
    get_storage_size: Any,
) -> int:
    """Sum storage bytes once per parameter key (excludes str placeholders)."""
    total_size = 0
    seen_keys: set[str] = set()
    for shard in shard_list:
        for k, tensor in shard.items():
            if k in seen_keys:
                continue
            seen_keys.add(k)
            if isinstance(tensor, str):
                continue
            total_size += get_storage_size(tensor)
    return total_size


def _split_state_dict_into_shards(
    state_dict: dict[str, Any],
    max_shard_size: int | str,
) -> _StateDictSplit:
    """Shard ``state_dict`` for safetensors using the active platform's storage size metric.

    Shards are built by iterating ``state_dict`` in key order (no bin-packing optimization). If a single tensor exceeds
    ``max_shard_size``, it occupies its own shard (possibly larger than the cap).

    Shard filenames follow Hugging Face conventions (``model.safetensors`` or
    ``model-00001-of-00002.safetensors``).
    """
    platform = get_platform()
    get_storage_size = platform.get_tensor_storage_size

    filename_pattern = _HF_SAFE_WEIGHTS_FILENAME_PATTERN

    max_cap: int | str = max_shard_size
    if isinstance(max_cap, str):
        max_cap = _parse_size_to_int(max_cap)

    shard_list = _build_shard_list_for_safetensors(state_dict, max_cap, get_storage_size)
    nb_shards = len(shard_list)

    total_size = _total_bytes_unique_keys_in_shards(shard_list, get_storage_size)

    # If we only have one shard, we return it => no need to build the index
    if nb_shards == 1:
        filename = filename_pattern.format(suffix="")
        keys_in_shards = [k for k, t in state_dict.items() if not isinstance(t, str)]
        return _StateDictSplit(
            metadata={"total_size": total_size},
            filename_to_tensors={filename: keys_in_shards},
            tensor_to_filename={key: filename for key in keys_in_shards},
        )

    # Now that each tensor is assigned to a shard, let's assign a filename to each shard
    tensor_name_to_filename = {}
    filename_to_tensors = {}
    for idx, shard in enumerate(shard_list):
        filename = filename_pattern.format(suffix=f"-{idx + 1:05d}-of-{nb_shards:05d}")
        for key in shard:
            tensor_name_to_filename[key] = filename
        filename_to_tensors[filename] = list(shard.keys())

    # Build the index and return
    return _StateDictSplit(
        metadata={"total_size": total_size},
        filename_to_tensors=filename_to_tensors,
        tensor_to_filename=tensor_name_to_filename,
    )


def save_state_dict_as_huggingface_format(
    save_directory: str | os.PathLike[str],
    state_dict: dict[str, Any],
    max_shard_size: str | int = "5GB",
) -> None:
    """Write ``state_dict`` to ``save_directory`` as Hugging Face-style safetensors (optionally sharded).

    Args:
        save_directory: Output directory path.
        state_dict: Parameter tensors for the active platform, keyed by name.
        max_shard_size: Per-shard byte cap (int) or string such as ``\"5GB\"``.

    Returns:
        None
    """
    if state_dict is None:
        raise ValueError("state_dict is None.")

    _validate_state_dict_for_active_platform(state_dict)

    if os.path.isfile(save_directory):
        raise ValueError(f"The save_directory {save_directory} should be a directory, but a file.")

    os.makedirs(save_directory, exist_ok=True)

    platform = get_platform()
    weights_name = _SAFE_WEIGHTS_NAME
    state_dict_split = _split_state_dict_into_shards(state_dict, max_shard_size=max_shard_size)

    index = None
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }

    filename_to_tensors = state_dict_split.filename_to_tensors.items()

    for shard_file, tensors in filename_to_tensors:
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        platform.save_checkpoint(shard, os.path.join(save_directory, shard_file), ckpt_format="safetensors")
    if index is None:
        path_to_weights = os.path.join(save_directory, weights_name)
        logger.info("Model weights saved in %s", path_to_weights)
    else:
        save_index_file = os.path.join(save_directory, _SAFE_WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(
            "Model exceeds max shard size %s; split into %s shards. Weight map index: %s",
            max_shard_size,
            len(state_dict_split.filename_to_tensors),
            save_index_file,
        )


def parse_checkpoint_from_huggingface(resume_from_checkpoint: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a Hugging Face ``model.safetensors`` or sharded safetensors + index from a directory.

    Uses the active platform's :meth:`Platform.load_checkpoint` for safetensors.

    Args:
        resume_from_checkpoint: Directory containing ``model.safetensors`` or index + shard files.

    Returns:
        Flattened str-keyed state dict (framework tensors on CPU when applicable).
    """
    platform = get_platform()
    safe_weights_file = os.path.join(resume_from_checkpoint, _SAFE_WEIGHTS_NAME)

    state_dict: dict[str, Any] = {}

    if os.path.isfile(safe_weights_file):
        state_dict = platform.load_checkpoint(safe_weights_file, ckpt_format="safetensors")
        logger.info("Loaded safetensors checkpoint from %s", safe_weights_file)

    else:
        safe_index_file = os.path.join(resume_from_checkpoint, _SAFE_WEIGHTS_INDEX_NAME)
        if not os.path.isfile(safe_index_file):
            raise ValueError(f"Can't find a checkpoint index in {os.path.abspath(resume_from_checkpoint)}.")

        with open(safe_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = list(set(index["weight_map"].values()))

        total_size = 0
        for shard_file in shard_files:
            shard_path = os.path.join(resume_from_checkpoint, shard_file)
            shard_state_dict = platform.load_checkpoint(shard_path, ckpt_format="safetensors")
            for key, value in shard_state_dict.items():
                if key in state_dict:
                    logger.warning(
                        "Duplicate key %r when merging Hugging Face shards; keeping first occurrence.",
                        key,
                    )
                    continue
                state_dict[key] = value
                if platform.is_tensor(value):
                    size = platform.get_tensor_storage_size(value)
                    total_size += size
                    logger.debug("Loaded tensor %r, size %s bytes", key, size)
                else:
                    logger.debug("Loaded non-tensor entry %r", key)
        logger.info("Merged Hugging Face shards, total tensor bytes (sum per key): %s", total_size)
    _validate_state_dict_for_active_platform(state_dict)
    return state_dict


def _mindspore_full_checkpoint_format_for_path(path: str) -> str:
    """Infer MindSpore ``load_checkpoint`` ``format`` from a full-checkpoint file suffix."""
    lower = path.lower()
    if lower.endswith(".safetensors"):
        return "safetensors"
    return "ckpt"


def _torch_full_checkpoint_format_for_path(path: str) -> str:
    """Infer PyTorch :meth:`Platform.load_checkpoint` ``ckpt_format`` from a full-checkpoint file suffix."""
    lower = path.lower()
    if lower.endswith(".safetensors"):
        return "safetensors"
    return "pickle"


def full_state_dict_to_dcp_format(
    state_dict: dict[str, Any],
    dst_dir: str | os.PathLike[str],
) -> None:
    """Convert a full-weights ``state_dict`` into HyperParallel DCP layout under ``dst_dir``.

    Args:
        state_dict: Merged weights to store; types must match the active platform (see project conventions).
        dst_dir: Output directory for DCP files (``.metadata`` and shard files).

    Returns:
        None
    """
    _validate_state_dict_for_active_platform(state_dict)
    storage_writer = FileSystemWriter(dst_dir)
    planner = StandardSavePlanner()

    planner.configure_planner(state_dict=state_dict, is_coordinator=True)
    storage_writer.configure_writer(is_coordinator=True, rank=0, use_collectives=True)
    local_plan = planner.build_local_plan()
    local_data = storage_writer.optimize_local_plan(local_plan)

    all_local_plans, global_metadata = planner.build_global_plan([local_data])
    central_plan = storage_writer.optimize_global_plan(all_local_plans)[0]

    final_local_plan = planner.finalize_plan(central_plan)
    all_writes = storage_writer.execute_write(final_local_plan, planner)

    storage_writer.finalize_checkpoint(metadata=global_metadata, results=[all_writes])


def dcp_to_full_state_dict(src_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a **shard** DCP layout from ``src_dir`` into one merged full-weights ``state_dict``.

    Coordinator / single-process merge of the DCP shards on disk.

    Args:
        src_dir: Root directory of a saved DCP checkpoint.

    Returns:
        Merged state dictionary populated by the load planner and storage reader.
    """
    state_dict: dict[str, Any] = {}
    planner = _DcpMergeLoadPlanner()
    storage_reader = FileSystemReader(src_dir)

    metadata = storage_reader.load_metadata()
    planner.configure_planner(state_dict, metadata, is_coordinator=True)
    storage_reader.configure_reader(metadata, is_coordinator=True, rank=0)

    local_plan = planner.build_local_plan()
    local_data = storage_reader.optimize_local_plan(local_plan)
    all_data = [local_data]

    all_local_plans = planner.build_global_plan(all_data)
    all_results = storage_reader.optimize_global_plan(all_local_plans)

    final_local_plan = planner.finalize_plan(all_results[0])
    storage_reader.execute_read(final_local_plan, planner)

    return state_dict


def convert_full_checkpoint_to_dcp(
    src_ckpt: str | os.PathLike[str],
    dst_dir: str | os.PathLike[str],
    src_platform: Literal["torch", "huggingface", "mindspore"] = "torch",
) -> None:
    """Load checkpoint weights and write HyperParallel DCP under ``dst_dir``.

    Args:
        src_ckpt: Source path: Hugging Face safetensors directory (full or shard layout), a PyTorch **full checkpoint**
            file, or a MindSpore **full checkpoint** file (``.ckpt`` / ``.safetensors``).
        dst_dir: Output directory for DCP files.
        src_platform: ``huggingface`` (HF directory), ``torch`` or ``mindspore`` (full-checkpoint file via
            :meth:`Platform.load_checkpoint`). For ``torch`` / ``mindspore``, the active runtime must match
            (``HYPER_PARALLEL_PLATFORM``). ``huggingface`` uses either platform for safetensors.

    Returns:
        None
    """
    platform = get_platform()
    if src_platform == "huggingface":
        state_dict = parse_checkpoint_from_huggingface(src_ckpt)
    elif src_platform == "torch":
        if platform.platform_type != PlatformType.PYTORCH:
            raise ValueError(
                "src_platform='torch' requires the PyTorch platform; set HYPER_PARALLEL_PLATFORM=torch."
            )
        if not os.path.isfile(src_ckpt):
            raise ValueError(
                "src_platform='torch' requires a single PyTorch checkpoint file path; "
                f"src_ckpt must be an existing file, got {src_ckpt!r}."
            )
        fmt = _torch_full_checkpoint_format_for_path(str(src_ckpt))
        state_dict = platform.load_checkpoint(str(src_ckpt), ckpt_format=fmt)
    elif src_platform == "mindspore":
        if platform.platform_type != PlatformType.MINDSPORE:
            raise ValueError(
                "src_platform='mindspore' requires the MindSpore platform; set HYPER_PARALLEL_PLATFORM=mindspore."
            )
        if not os.path.isfile(src_ckpt):
            raise ValueError(
                "src_platform='mindspore' requires a single checkpoint file path; "
                f"src_ckpt must be an existing file, got {src_ckpt!r}."
            )
        fmt = _mindspore_full_checkpoint_format_for_path(str(src_ckpt))
        state_dict = platform.load_checkpoint(str(src_ckpt), ckpt_format=fmt)
    else:
        raise ValueError(
            f"Unsupported src_platform={src_platform!r}; expected 'huggingface', 'torch', or 'mindspore'."
        )
    full_state_dict_to_dcp_format(state_dict, dst_dir)
