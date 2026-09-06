# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""data.parallel.batch_parallel: CP-aware batch sharding.

``shard_batch_for_cp``: data-pipeline CP sharding (aligned with the THD
contract of the 02 collater); ``_shard_seq_lens_for_cp`` recomputes
seq_lens/seq_lens_padded per CP rank. ``CPBatchSharder`` slices
``input_ids``/``labels`` contiguously along the sequence dimension and
``TPBatchBroadcaster`` moves the CP-local tensors and global sequence
boundaries to the target device, broadcasting metadata and tensor fields
to peer TP ranks (the canonical batch exists only on TP rank zero for the
active CP coordinate).

Split out of components/distributed/cp_utils.py in stage 4e; merged with
components/datasets/parallel/batch_parallel.py in stage 6 (05 §11.2).
"""

from collections.abc import Mapping
from typing import Any, Callable, Optional
import torch
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform
from hyper_parallel.data.parallel.dataloader_parallel import DataLoaderParallelContext


platform = get_platform()


def shard_batch_for_cp(batch: dict[str, Any], cp_mesh: DeviceMesh) -> dict[str, Any]:
    """Shard the sequence-dim tensors of a batch along the CP mesh
    (05 §6.3.4 canonical).

    Contract (aligned with the 02 collater output):
      - input_ids/labels/position_ids: [B, S] int64
      - seq_lens / seq_lens_padded: [B, max_num_packs] int64, padded with the
        -1000 sentinel
      - qkv_format: "thd" (passthrough)

    Sharding strategy: pad to a multiple of 2*cp, then slice the token
    interval [cp_rank*chunk, (cp_rank+1)*chunk); the seq_lens family is
    recomputed separately (_shard_seq_lens_for_cp).
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return batch

    cp_rank = cp_mesh.get_local_rank()
    seq_len = batch["input_ids"].shape[1]
    pad_len = (-seq_len) % (cp_size * 2)
    chunk = (seq_len + pad_len) // cp_size
    lo = cp_rank * chunk
    hi = lo + chunk
    slc = slice(lo, hi)

    pad_values = {"labels": -100, "input_ids": 0, "attention_mask": 0}
    padded = dict(batch)
    if pad_len > 0:
        for k, v in batch.items():
            if k == "qkv_format" or not isinstance(v, torch.Tensor) or v.ndim < 1:
                continue
            if k in ("seq_lens", "seq_lens_padded"):
                continue  # recomputed separately, not padded
            if k == "position_ids":
                # position_ids increment-pad: continue incrementing from the last value
                last = v[..., -1:].to(torch.long)
                inc = torch.arange(1, pad_len + 1, device=v.device,
                                   dtype=v.dtype)
                inc = inc.reshape(*([1] * (v.ndim - 1)), pad_len)
                pad_block = inc.expand(*v.shape[:-1], pad_len) + last
            else:
                shape = list(v.shape)
                shape[-1] = pad_len
                pad_block = torch.full(shape, pad_values.get(k, 0),
                                       dtype=v.dtype, device=v.device)
            padded[k] = torch.cat([v, pad_block], dim=-1)

    out = {}
    for k, v in padded.items():
        if k in ("seq_lens", "seq_lens_padded"):
            continue
        if k == "qkv_format":
            out[k] = v
        elif isinstance(v, torch.Tensor) and v.ndim >= 1:
            out[k] = v[..., slc]
        else:
            out[k] = v

    if "seq_lens" in batch and "seq_lens_padded" in batch:
        out["seq_lens"], out["seq_lens_padded"] = _shard_seq_lens_for_cp(
            batch["seq_lens"], batch["seq_lens_padded"],
            cp_rank=cp_rank, chunk=chunk,
        )
    return out


def _shard_seq_lens_for_cp(seq_lens, seq_lens_padded, *, cp_rank: int, chunk: int):
    """Recompute seq_lens/seq_lens_padded per CP shard (preserving the -1000
    sentinel semantics).

    Walk the cumulative pack offsets of each sample (accumulated by
    seq_lens_padded); for each pack:
    - fully inside [lo, hi): kept as-is;
    - crossing the boundary: truncated to [lo, hi), with the actual /
      padding-inclusive lengths recomputed after truncation;
    - fully outside: skipped.
    The output is shifted to the local coordinate system; when
    max_local_packs=0 it is set to 1 to avoid an empty tensor.
    """
    batch_size = seq_lens.shape[0]
    lo = cp_rank * chunk
    hi = lo + chunk
    device = seq_lens.device
    sentinel = -1000

    local_lens_b, local_lens_padded_b = [], []
    max_local_packs = 0
    for b in range(batch_size):
        row_lens = seq_lens[b].tolist()
        row_padded = seq_lens_padded[b].tolist()
        local_lens, local_padded = [], []
        offset = 0
        for raw_len, raw_pad in zip(row_lens, row_padded):
            if raw_len == sentinel:
                break
            pack_start = offset
            pack_end = offset + raw_pad
            offset = pack_end
            inter_start = max(pack_start, lo)
            inter_end = min(pack_end, hi)
            if inter_start >= inter_end:
                continue
            actual_start = max(pack_start, lo)
            actual_end = min(pack_start + raw_len, hi)
            local_actual = max(actual_end - actual_start, 0)
            local_pad = inter_end - inter_start
            if local_actual > 0 or local_pad > 0:
                local_lens.append(local_actual)
                local_padded.append(local_pad)
        local_lens_b.append(local_lens)
        local_lens_padded_b.append(local_padded)
        max_local_packs = max(max_local_packs, len(local_lens))

    if max_local_packs == 0:
        max_local_packs = 1

    out_lens = torch.full((batch_size, max_local_packs), sentinel,
                          dtype=seq_lens.dtype, device=device)
    out_padded = torch.full((batch_size, max_local_packs), sentinel,
                            dtype=seq_lens_padded.dtype, device=device)
    for b in range(batch_size):
        n = len(local_lens_b[b])
        if n > 0:
            out_lens[b, :n] = torch.tensor(
                local_lens_b[b], dtype=seq_lens.dtype, device=device)
            out_padded[b, :n] = torch.tensor(
                local_lens_padded_b[b], dtype=seq_lens_padded.dtype, device=device)
    return out_lens, out_padded


# ── DataLoader batch distribution (merged from components/datasets/parallel/batch_parallel.py, 05 §11.2) ──


class CPBatchSharder:
    """Select the contiguous token fields owned by the current CP rank."""

    def __init__(
            self,
            parallel_context: DataLoaderParallelContext,
    ) -> None:
        """Initialize context-parallel batch sharding.

        Args:
            parallel_context: DataLoader and TP/CP topology information.
        """
        self.parallel_context = parallel_context

    def shard(
            self,
            canonical_batch: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Shard input IDs and labels along the sequence dim.

        Args:
            canonical_batch: Normalized complete batch on TP rank zero, or ``None`` elsewhere.

        Returns:
            CP-local input IDs and labels on TP rank zero.
        """
        if canonical_batch is None:
            cp_batch = None
            return cp_batch

        input_ids = canonical_batch["input_ids"]
        cp_batch = {
            "input_ids": input_ids,
            "labels": canonical_batch["labels"],
        }

        cp_size = self.parallel_context.cp_world_size
        if cp_size <= 1:
            return cp_batch

        seq_len = input_ids.shape[1]
        if seq_len % cp_size != 0:
            raise ValueError(f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size})")

        cp_rank = self.parallel_context.cp_rank
        for field, value in cp_batch.items():
            local_value = platform.chunk(value, split_dim=1, split_size=cp_size, index=cp_rank)
            cp_batch[field] = local_value.contiguous()

        return cp_batch


class TPBatchBroadcaster:
    """Broadcast CP-local batch fields within one TP group."""

    def __init__(
            self,
            parallel_context: DataLoaderParallelContext,
            device: Any,
    ) -> None:
        """Initialize tensor-parallel batch transport.

        Args:
            parallel_context: DataLoader and TP/CP topology information.
            device: Device receiving the broadcast batch.
        """
        self.parallel_context = parallel_context
        self.device = device

    def broadcast(
            self,
            cp_local_batch: Mapping[str, Any] | None,
            cu_seq_lens: Any,
    ) -> dict[str, Any]:
        """Broadcast CP-local fields and global boundaries across TP ranks.

        Args:
            cp_local_batch: CP-local fields on TP rank zero, or ``None`` elsewhere.
            cu_seq_lens: Global sequence boundaries, unsharded across CP ranks.

        Returns:
            Local batch populated on every TP rank.
        """
        tp_rank = self.parallel_context.tp_rank
        # TP rank zero owns the CP-local DataLoader fields and moves them to
        # the target device before communication.
        if tp_rank == 0:
            parallel_batch = {
                field: value.to(self.device, dtype=platform.tensor_dtype.int64, non_blocking=True)
                for field, value in cp_local_batch.items()
            }
            local_cu_seq_lens = None
            if cu_seq_lens is not None:
                local_cu_seq_lens = cu_seq_lens.to(self.device, dtype=platform.tensor_dtype.int32, non_blocking=True)
        else:
            parallel_batch = None
            local_cu_seq_lens = None

        tp_size = self.parallel_context.tp_world_size
        if tp_size <= 1:
            parallel_batch["cu_seq_lens"] = local_cu_seq_lens
            return parallel_batch

        # Dynamic Online batches do not have a fixed local sequence length.
        # Broadcast compact shape metadata before receivers allocate tensors.
        if tp_rank == 0:
            batch_size, seq_len = parallel_batch["input_ids"].shape
            num_boundaries = 0 if local_cu_seq_lens is None else local_cu_seq_lens.numel()
            batch_meta = platform.tensor(
                [batch_size, seq_len, num_boundaries],
                dtype=platform.tensor_dtype.int64,
                device=self.device,
            )
        else:
            batch_meta = platform.empty((3,), dtype=platform.tensor_dtype.int64, device=self.device)

        tp_group = self.parallel_context.tp_group
        platform.broadcast(batch_meta, group=tp_group, group_src=0)

        # Non-source TP ranks allocate the same device shapes and dtypes.
        if tp_rank != 0:
            batch_size, seq_len, num_boundaries = [int(value) for value in batch_meta.tolist()]
            shape = (batch_size, seq_len)
            parallel_batch = {
                "input_ids": platform.empty(shape, dtype=platform.tensor_dtype.int64, device=self.device),
                "labels": platform.empty(shape, dtype=platform.tensor_dtype.int64, device=self.device),
            }
            if num_boundaries > 0:
                local_cu_seq_lens = platform.empty(
                    (num_boundaries,), dtype=platform.tensor_dtype.int32, device=self.device
                )

        # Broadcast only model fields that cannot be regenerated locally.
        for field in ("input_ids", "labels"):
            platform.broadcast(parallel_batch[field], group=tp_group, group_src=0)

        # Packed boundaries remain global across CP ranks and variable in size.
        if local_cu_seq_lens is not None:
            platform.broadcast(local_cu_seq_lens, group=tp_group, group_src=0)

        parallel_batch["cu_seq_lens"] = local_cu_seq_lens

        return parallel_batch
