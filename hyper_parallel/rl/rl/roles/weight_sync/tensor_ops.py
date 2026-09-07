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
"""Shared tensor extraction and bounded packing for weight transports."""

from collections.abc import Mapping
from typing import Any

from rl.roles.weight_sync.layout import TransferBucket


def local_tensor(value: Any) -> Any:
    """Return a DTensor's local shard or the original plain tensor."""
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def pack_direct_bucket(
    state_dict: Mapping[str, Any],
    bucket: TransferBucket,
    device: Any,
) -> Any:
    """Pack one direct route into a bounded byte tensor on ``device``."""
    import torch  # pylint: disable=C0415,forbidden-backend-import

    packed = torch.empty(bucket.total_bytes, dtype=torch.uint8, device=device)
    for entry in bucket.entries:
        value = state_dict.get(entry.source_key)
        if value is None:
            raise ValueError(
                f"Direct reshard source parameter {entry.source_key!r} is missing"
            )
        source_slice = tuple(
            slice(start, start + length)
            for start, length in zip(entry.source_starts, entry.lengths)
        )
        fragment = local_tensor(value)[source_slice].detach()
        if entry.physical_permutation != tuple(range(len(entry.lengths))):
            fragment = fragment.permute(entry.physical_permutation)
        fragment = fragment.contiguous()
        if str(fragment.device) != str(device):
            fragment = fragment.to(device)
        raw = fragment.view(torch.uint8).view(-1)
        if int(raw.numel()) != entry.num_bytes:
            raise ValueError(
                f"Direct reshard source fragment {entry.source_key!r} has "
                f"{raw.numel()} bytes, expected {entry.num_bytes}"
            )
        packed.narrow(0, entry.buffer_offset, entry.num_bytes).copy_(raw)
    return packed


__all__ = ["local_tensor", "pack_direct_bucket"]
