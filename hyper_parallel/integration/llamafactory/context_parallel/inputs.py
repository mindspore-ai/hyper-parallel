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
"""Context-parallel input helpers for the LlamaFactory integration."""
import torch
import torch.distributed as dist

from hyper_parallel.platform import get_platform

_CP_GROUP_CACHE: dict[tuple[int, ...], dist.ProcessGroup] = {}
_CP_VISION_INPUT_KEYS = {
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "mm_token_type_ids",
}


def _get_cp_dp_ranks(hp_args) -> tuple[int, int]:
    """Map global rank to logical ``(cp_rank, dp_rank)`` for a ``(dp, cp)`` mesh."""
    cp_size = getattr(hp_args, "cp_size", 1)
    rank = get_platform().get_rank()
    if cp_size <= 1:
        return 0, rank

    world_size = get_platform().get_world_size()
    if world_size % cp_size != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by cp_size ({cp_size}).")

    dp_rank = rank // cp_size
    cp_rank = rank % cp_size
    return cp_rank, dp_rank


def get_cp_rank(hp_args) -> int:
    """Return the CP rank for the current process."""
    return _get_cp_dp_ranks(hp_args)[0]


def get_dp_rank(hp_args) -> int:
    """Return the DP/FSDP rank for the current process."""
    return _get_cp_dp_ranks(hp_args)[1]


def get_cp_group(hp_args):
    """Return the CP process group for the current DP column."""
    cp_size = getattr(hp_args, "cp_size", 1)
    if cp_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return None

    _, dp_rank = _get_cp_dp_ranks(hp_args)
    world_size = get_platform().get_world_size()
    target_ranks = tuple(dp_rank * cp_size + cp_idx for cp_idx in range(cp_size))
    cached_group = _CP_GROUP_CACHE.get(target_ranks)
    if cached_group is not None:
        return cached_group

    for current_dp_rank in range(world_size // cp_size):
        ranks = tuple(current_dp_rank * cp_size + cp_idx for cp_idx in range(cp_size))
        if ranks in _CP_GROUP_CACHE:
            continue
        _CP_GROUP_CACHE[ranks] = dist.new_group(ranks=list(ranks))
    return _CP_GROUP_CACHE[target_ranks]


def get_cp_group_ranks(hp_args) -> tuple[int, ...] | None:
    """Return the global ranks that form the current CP group."""
    cp_size = getattr(hp_args, "cp_size", 1)
    if cp_size <= 1:
        return None
    _, dp_rank = _get_cp_dp_ranks(hp_args)
    return tuple(dp_rank * cp_size + cp_idx for cp_idx in range(cp_size))


def _slice_tensor_along_dim(tensor: torch.Tensor, dim: int, start: int, end: int) -> torch.Tensor:
    """Slice a tensor along one dimension."""
    index = [slice(None)] * tensor.dim()
    index[dim] = slice(start, end)
    return tensor[tuple(index)].contiguous()


def _has_active_vision_inputs(inputs: dict) -> bool:
    """Return whether the batch carries multimodal inputs that must stay aligned with full placeholders."""
    for key in _CP_VISION_INPUT_KEYS:
        value = inputs.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return True
    return False


def _add_multimodal_cp_metadata(sharded_inputs: dict, inputs: dict, start: int, end: int, seq_len: int) -> None:
    """Attach full-sequence metadata needed by mask-aware multimodal CP sharding."""
    sharded_inputs["_hp_cp_local_seq_start"] = start
    sharded_inputs["_hp_cp_local_seq_end"] = end
    sharded_inputs["_hp_cp_global_input_ids"] = inputs["input_ids"]
    position_ids = inputs.get("position_ids")
    if isinstance(position_ids, torch.Tensor) and position_ids.dim() >= 2 and position_ids.size(-1) == seq_len:
        sharded_inputs["_hp_cp_global_position_ids"] = position_ids


def _shard_tensor_input(key: str, value: torch.Tensor, seq_len: int, start: int, end: int) -> torch.Tensor:
    """Shard one tensor input while preserving full tensors needed by CP runtime patches."""
    if key in _CP_VISION_INPUT_KEYS:
        return value
    if key == "attention_mask":
        if value.dim() == 4 and value.size(-2) == seq_len:
            return _slice_tensor_along_dim(value, -2, start, end)
        return value
    if key == "position_ids" and value.dim() >= 2 and value.size(-1) == seq_len:
        return _slice_tensor_along_dim(value, -1, start, end)
    if key == "labels" and value.dim() >= 2 and value.size(1) == seq_len:
        return value
    if key in {"input_ids", "inputs_embeds"} and value.dim() >= 2 and value.size(1) == seq_len:
        return value[:, start:end].contiguous()
    if value.dim() >= 2 and value.size(1) == seq_len and key != "loss_mask":
        return value[:, start:end].contiguous()
    return value


def shard_inputs_for_cp(inputs: dict, cp_rank: int, cp_size: int) -> dict:
    """Shard sequence inputs for CP while preserving multimodal alignment metadata."""
    if cp_size <= 1:
        return inputs

    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() < 2:
        return inputs

    seq_len = input_ids.size(1)
    if seq_len % cp_size != 0:
        return inputs

    shard_len = seq_len // cp_size
    start = cp_rank * shard_len
    end = start + shard_len
    sharded_inputs = dict(inputs)
    has_vision_inputs = _has_active_vision_inputs(inputs)
    if has_vision_inputs:
        _add_multimodal_cp_metadata(sharded_inputs, inputs, start, end, seq_len)

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            sharded_inputs[key] = _shard_tensor_input(key, value, seq_len, start, end)

    if "position_ids" not in sharded_inputs and not has_vision_inputs:
        device = input_ids.device
        position_ids = torch.arange(start, end, device=device, dtype=torch.long).unsqueeze(0)
        sharded_inputs["position_ids"] = position_ids.expand(input_ids.size(0), -1)

    return sharded_inputs
