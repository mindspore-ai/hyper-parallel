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
"""Bounded decoding and compact transport for native vLLM router capture."""

import base64
import binascii
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from rl.dataset.contracts import RouterTrace

_NPY_HEADER_LIMIT = 1024


def _is_integer(value: Any) -> bool:
    """Keep boolean settings from impersonating dimensions or expert counts."""
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class RouterCaptureSpec:
    """Checkpoint-derived dimensions and expert bounds for one capture backend."""

    layer_ids: tuple[int, ...]
    top_k: int
    num_experts: int

    def __post_init__(self) -> None:
        """Validate dimensions before using them to bound response allocations."""
        if (not isinstance(self.layer_ids, tuple) or not self.layer_ids
                or any(not _is_integer(layer) or layer < 0 for layer in self.layer_ids)
                or tuple(sorted(set(self.layer_ids))) != self.layer_ids):
            raise ValueError("router capture layer IDs must be ordered unique non-negative integers")
        if (not _is_integer(self.top_k) or not _is_integer(self.num_experts)
                or not 0 < self.top_k <= self.num_experts <= np.iinfo(np.int64).max):
            raise ValueError("router capture requires 0 < top_k <= num_experts <= int64 maximum")

    @classmethod
    def from_checkpoint(cls, weights_path: str) -> "RouterCaptureSpec":
        """Read the local Qwen3-MoE sparse-layer layout without loading a model."""
        with (Path(weights_path) / "config.json").open(encoding="utf-8") as source:
            config = json.load(source)
        if not isinstance(config, dict):
            raise ValueError("router capture checkpoint configuration must be a mapping")
        layers = config.get("num_hidden_layers")
        step = config.get("decoder_sparse_step", 1)
        dense_layers = config.get("mlp_only_layers", [])
        if (not _is_integer(layers) or layers <= 0 or not _is_integer(step) or step <= 0
                or not isinstance(dense_layers, list)
                or any(not _is_integer(layer) or not 0 <= layer < layers for layer in dense_layers)):
            raise ValueError("router capture requires a valid Qwen3-MoE decoder layer layout")
        return cls(tuple(layer for layer in range(layers)
                         if layer not in dense_layers and (layer + 1) % step == 0),
                   config.get("num_experts_per_tok"), config.get("num_experts"))


def _load_router_array(binary: bytes, token_count: int, spec: RouterCaptureSpec) -> np.ndarray:
    """Validate the NPY header and byte count before loading bounded expert IDs."""
    source = io.BytesIO(binary)
    try:
        version = np.lib.format.read_magic(source)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(source, max_header_size=_NPY_HEADER_LIMIT)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(source, max_header_size=_NPY_HEADER_LIMIT)
        else:
            raise ValueError("unsupported router capture NPY version")
    except (ValueError, EOFError, OSError) as error:
        raise ValueError("router capture payload has an invalid NPY header") from error
    if (len(shape) != 3 or shape[0] not in (token_count - 1, token_count)
            or shape[1:] != (len(spec.layer_ids), spec.top_k)):
        raise ValueError("router capture shape must be [S or S-1, checkpoint layers, top_k]")
    if dtype.kind not in "iu" or dtype.itemsize not in (1, 2, 4, 8):
        raise ValueError("router capture NPY must contain integer expert IDs without pickle")
    expected_bytes = int(np.prod(shape, dtype=object)) * dtype.itemsize
    if len(binary) - source.tell() != expected_bytes:
        raise ValueError("router capture payload byte count does not match its NPY shape")
    array = np.load(io.BytesIO(binary), allow_pickle=False, max_header_size=_NPY_HEADER_LIMIT)
    if np.any(array < 0) or np.any(array >= spec.num_experts):
        raise ValueError("router capture expert ID is outside the checkpoint expert range")
    return array


def decode_router_trace(payload: Any, token_count: int, spec: RouterCaptureSpec) -> RouterTrace:
    """Decode one base64 NPY, retaining input positions and a missing final row.

    Args:
        payload: Native vLLM ``choices.routed_experts`` base64 NPY string.
        token_count: Unpadded prompt plus returned response length.
        spec: Expected sparse layer mapping, top-k and number of experts.

    Returns:
        A compact CPU trace, with a false final mask when capture returned S-1 rows.
    """
    if not _is_integer(token_count) or token_count <= 0:
        raise ValueError("router capture token_count must be positive")
    max_bytes = token_count * len(spec.layer_ids) * spec.top_k * 8 + _NPY_HEADER_LIMIT + 16
    if not isinstance(payload, str) or not payload or len(payload) > 4 * ((max_bytes + 2) // 3):
        raise ValueError("router capture requires a bounded base64 NPY payload")
    try:
        binary = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("router capture payload is not valid base64") from error
    array = _load_router_array(binary, token_count, spec)
    compact_dtype = next(dtype for dtype in (np.uint8, np.int16, np.int32, np.int64)
                         if spec.num_experts - 1 <= np.iinfo(dtype).max)
    expert_ids = torch.zeros((token_count, len(spec.layer_ids), spec.top_k),
                             dtype=torch.from_numpy(np.empty(0, dtype=compact_dtype)).dtype)
    captured = array.shape[0]
    expert_ids[:captured] = torch.from_numpy(array.astype(compact_dtype, copy=True))
    valid_mask = torch.arange(token_count) < captured
    return RouterTrace(expert_ids, valid_mask, spec.layer_ids)


def router_trace_from_capture(record: Mapping[str, Any], token_count: int) -> Optional[RouterTrace]:
    """Normalize a gateway's JSON capture at the per-call program boundary."""
    spec = record.get("router_capture_spec")
    if spec is None:
        return None
    if not isinstance(spec, Mapping) or not isinstance(spec.get("layer_ids"), (list, tuple)):
        raise ValueError("Captured router trace requires checkpoint router metadata")
    capture_spec = RouterCaptureSpec(tuple(spec["layer_ids"]), spec.get("top_k"), spec.get("num_experts"))
    return decode_router_trace(record.get("routed_experts"), token_count, capture_spec)


def router_trace_to_wire(trace: RouterTrace) -> dict[str, Any]:
    """Detach compact CPU buffers from their owner for existing TP object transport."""
    return {"shape": tuple(trace.expert_ids.shape), "dtype": str(trace.expert_ids.numpy().dtype),
            "expert_ids": trace.expert_ids.contiguous().numpy().tobytes(),
            "valid_mask": trace.valid_mask.numpy().tobytes(), "layer_ids": trace.layer_ids}


def router_trace_from_wire(payload: Mapping[str, Any]) -> RouterTrace:
    """Restore binary TP buffers without materializing Python expert-ID lists."""
    if not isinstance(payload, Mapping):
        raise ValueError("router trace wire payload must be a mapping")
    shape, dtype_name = payload.get("shape"), payload.get("dtype")
    if (not isinstance(shape, tuple) or len(shape) != 3
            or any(not _is_integer(size) or size <= 0 for size in shape)
            or dtype_name not in ("uint8", "int8", "int16", "int32", "int64")):
        raise ValueError("router trace wire shape or dtype is invalid")
    dtype = np.dtype(dtype_name)
    ids, mask = payload.get("expert_ids"), payload.get("valid_mask")
    if (not isinstance(ids, bytes) or len(ids) != shape[0] * shape[1] * shape[2] * dtype.itemsize
            or not isinstance(mask, bytes) or len(mask) != shape[0]
            or bool((np.frombuffer(mask, dtype=np.uint8) > 1).any())):
        raise ValueError("router trace wire byte count or mask is invalid")
    return RouterTrace(torch.from_numpy(np.frombuffer(ids, dtype=dtype).reshape(shape).copy()),
                       torch.from_numpy(np.frombuffer(mask, dtype=np.bool_).copy()), payload.get("layer_ids"))
