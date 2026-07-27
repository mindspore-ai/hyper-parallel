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
"""cp_utils: Context Parallel utilities (05 §4.4.2 / §6.3.4 canonical).

- ``flex_cp_allgather``: K/V all-gather along the CP dim (reuses
  cp_mesh.get_group(); new_group is forbidden);
- ``shard_batch_for_cp``: data-pipeline CP sharding (aligned with the THD
  contract of the 02 collater);
- ``_shard_seq_lens_for_cp``: recompute seq_lens/seq_lens_padded per CP rank.

Note (G5): the seq_len % (2*cp) padding constraint originates from the
zigzag/ring load-balancing scheme; this design uses all-gather K/V +
contiguous chunk (D-01'' rejected ring), so each rank's Q chunk is
equal-length and FLOPs are naturally balanced -- the constraint is redundant
but harmless: the implementation is kept and documented here.
"""

import torch
import torch.distributed as dist


class _AllGatherAlongDim(torch.autograd.Function):
    """all-gather along cp_dim + backward reduce-scatter semantics (sum across
    ranks, then take this rank's chunk)."""

    @staticmethod
    def forward(ctx, t, cp_dim, group, cp_size):
        ctx.cp_dim = cp_dim
        ctx.group = group
        ctx.cp_size = cp_size
        world_t = [torch.empty_like(t) for _ in range(cp_size)]
        dist.all_gather(world_t, t.contiguous(), group=group)
        # cat in cp_rank order: [chunk_rank0, chunk_rank1, ...]
        return torch.cat(world_t, dim=cp_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # reduce-scatter: sum the gradient across ranks, take this rank's chunk
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        rank = dist.get_rank(ctx.group)
        local = torch.chunk(grad, ctx.cp_size, dim=ctx.cp_dim)[rank]
        return local.contiguous(), None, None, None


def flex_cp_allgather(k, v, cp_dim: int, cp_mesh):
    """All-gather K/V along CP dimension for context parallel attention.

    Forward: all-gather K/V along cp_dim (each rank ends up holding the full K/V).
    Backward: reduce-scatter semantics (gradients summed across ranks, then this
      rank's chunk is taken -- implemented explicitly by the _AllGatherAlongDim
      autograd.Function, since plain ``dist.all_gather`` has no autograd kernel).

    Args:
        k, v: [B, N, S_local, H] (cp_dim=2 is the sequence dim).
        cp_dim: the gather dimension.
        cp_mesh: DeviceMesh of the CP dim. The communication group is taken from
            ``cp_mesh.get_group()`` -- already created and cached at DeviceMesh
            construction; **dist.new_group must NOT be called here** (otherwise
            every forward leaks one process group, and the semantics would be
            misaligned).
    """
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return k, v
    group = cp_mesh.get_group()
    return (_AllGatherAlongDim.apply(k, cp_dim, group, cp_size),
            _AllGatherAlongDim.apply(v, cp_dim, group, cp_size))


def shard_batch_for_cp(batch: dict, cp_mesh) -> dict:
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

    _PAD_VALUE = {"labels": -100, "input_ids": 0, "attention_mask": 0}
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
                pad_block = torch.full(shape, _PAD_VALUE.get(k, 0),
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
    B, _K = seq_lens.shape
    lo = cp_rank * chunk
    hi = lo + chunk
    device = seq_lens.device
    SENTINEL = -1000

    local_lens_b, local_lens_padded_b = [], []
    max_local_packs = 0
    for b in range(B):
        row_lens = seq_lens[b].tolist()
        row_padded = seq_lens_padded[b].tolist()
        local_lens, local_padded = [], []
        offset = 0
        for raw_len, raw_pad in zip(row_lens, row_padded):
            if raw_len == SENTINEL:
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

    out_lens = torch.full((B, max_local_packs), SENTINEL,
                          dtype=seq_lens.dtype, device=device)
    out_padded = torch.full((B, max_local_packs), SENTINEL,
                            dtype=seq_lens_padded.dtype, device=device)
    for b in range(B):
        n = len(local_lens_b[b])
        if n > 0:
            out_lens[b, :n] = torch.tensor(
                local_lens_b[b], dtype=seq_lens.dtype, device=device)
            out_padded[b, :n] = torch.tensor(
                local_lens_padded_b[b], dtype=seq_lens_padded.dtype, device=device)
    return out_lens, out_padded


def _cp_offset_causal_mask(q_len: int, kv_len: int, lo: int,
                           device, dtype=torch.bool):
    """D-04: offset-aware causal mask (this rank's Q chunk has global offset lo).

    Attendable positions: j <= lo + i (i is the local Q row index).
    Replaces is_causal=True -- torch SDPA's is_causal is top-left aligned when
    q_len != kv_len (equivalent to assuming Q starts at global position 0), so
    under CP the chunks of rank>0 would be incorrectly masked (G4).
    """
    i = torch.arange(q_len, device=device).view(-1, 1)
    j = torch.arange(kv_len, device=device).view(1, -1)
    return (j <= (lo + i)).to(dtype)
