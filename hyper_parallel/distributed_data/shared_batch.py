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
"""CPU tensor trees exchanged through reusable, producer-owned shared storage."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.transport import _decode_model_batch, _encode_model_batch, _tensor_specs


class SharedBatchWriter:
    """Publish one tensor tree until the receiver acknowledges its command.

    Each dtype has one growable shared buffer. The ordered request/reply
    protocol prevents reuse until the receiver has finished copying it.
    """

    def __init__(self, directory: str = "/dev/shm") -> None:
        """Allocate the private directory lazily with no fixed payload limit."""
        self._resources = ExitStack()
        self._directory = self._resources.enter_context(TemporaryDirectory(prefix="hp-data-", dir=directory))
        self._buffers: dict[torch.dtype, tuple[str, torch.Tensor]] = {}
        self._generation = 0

    def write(self, value: Any) -> tuple[Any, list[Any]]:
        """Copy CPU tensor leaves once and return their layout descriptors."""
        schema, tensors = _encode_model_batch(value)
        groups: dict[torch.dtype, list[int]] = {}
        for index, tensor in enumerate(tensors):
            if tensor.device.type != "cpu" or tensor.layout != torch.strided:
                raise ValueError("Process data producers must return dense CPU tensors.")
            groups.setdefault(tensor.dtype, []).append(index)
        descriptors = []
        for dtype, indices in groups.items():
            required = sum(tensors[index].numel() for index in indices)
            cached = self._buffers.get(dtype)
            if cached is None or cached[1].numel() < required:
                self._generation += 1
                capacity = 1 << max(0, required - 1).bit_length()
                path = str(Path(self._directory) / str(self._generation))
                buffer = torch.from_file(path, shared=True, size=capacity, dtype=dtype)
                self._buffers[dtype] = path, buffer
                if cached is not None:
                    Path(cached[0]).unlink()
            path, buffer = self._buffers[dtype]
            offset = 0
            fields = []
            for index in indices:
                tensor = tensors[index]
                count = tensor.numel()
                buffer[offset:offset + count].copy_(tensor.reshape(-1))
                fields.append((index, offset, count, tuple(tensor.shape)))
                offset += count
            descriptors.append((path, buffer.numel(), dtype, required, fields))
        return schema, descriptors

    def close(self) -> None:
        """Release this writer's buffers and unlink only its own files."""
        self._buffers.clear()
        self._resources.close()


class SharedBatchReader:
    """Map shared buffers and give each received batch independent storage."""

    def __init__(self) -> None:
        """Cache mappings by dtype without retaining obsolete generations."""
        self._mappings: dict[torch.dtype, tuple[str, torch.Tensor]] = {}

    def read(self, message: tuple[Any, list[Any]], *, pin_memory: bool = False) -> Any:
        """Copy before acknowledging reuse; optional pinned storage belongs to the caller."""
        schema, descriptors = message
        tensors = [None] * len(_tensor_specs(schema))
        for path, capacity, dtype, used, fields in descriptors:
            cached = self._mappings.get(dtype)
            if cached is None or cached[0] != path:
                cached = path, torch.from_file(path, shared=True, size=capacity, dtype=dtype)
                self._mappings[dtype] = cached
            # A caller may retain an old batch, and its H2D may still be pending.
            # Fresh destination storage makes both lifetimes independent of IPC reuse.
            buffer = torch.empty(used, dtype=dtype, pin_memory=pin_memory)
            buffer.copy_(cached[1][:used])
            for index, offset, count, shape in fields:
                tensors[index] = buffer[offset:offset + count].reshape(shape)
        return _decode_model_batch(schema, tensors)

    def close(self) -> None:
        """Release cached mappings without unlinking the sender's files."""
        self._mappings.clear()
