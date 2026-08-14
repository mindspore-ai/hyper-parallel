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
- ``ulysses_seq_to_head`` / ``ulysses_head_to_seq``: differentiable Pure
  Ulysses all-to-all layout transforms;
- ``shard_batch_for_cp``: data-pipeline CP sharding (aligned with the THD
  contract of the 02 collater);
- ``_shard_seq_lens_for_cp``: recompute seq_lens/seq_lens_padded per CP rank.
- Ulysses collectives and MoME/MLA/DSA model-side adaptations.

Note (G5): the seq_len % (2*cp) padding constraint originates from the
zigzag/ring load-balancing scheme; this design uses all-gather K/V +
contiguous chunk (D-01'' rejected ring), so each rank's Q chunk is
equal-length and FLOPs are naturally balanced -- the constraint is redundant
but harmless: the implementation is kept and documented here.
"""

import contextvars
import functools
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.platform import get_platform


_ULYSSES_WRAPPED_FLAG = "_hyper_ulysses_wrapped"
platform = get_platform()


class _SeqAllToAll(torch.autograd.Function):
    """Autograd-aware exchange from one tensor dimension to another."""

    @staticmethod
    def forward(ctx, group, tensor, scatter_dim, gather_dim):
        """Exchange tensor shards between the sequence and head dimensions."""
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(group)
        if tensor.size(scatter_dim) % world_size:
            raise ValueError(
                f"dimension {scatter_dim} size {tensor.size(scatter_dim)} "
                f"is not divisible by CP size {world_size}")
        inputs = [
            part.contiguous()
            for part in tensor.chunk(world_size, dim=scatter_dim)
        ]
        outputs = [torch.empty_like(inputs[0]) for _ in range(world_size)]
        dist.all_to_all(outputs, inputs, group=group)
        return torch.cat(outputs, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _SeqAllToAll.apply(
            ctx.group, grad_output, ctx.gather_dim, ctx.scatter_dim)
        return None, grad_input, None, None


@dataclass(frozen=True)
class _UlyssesContext:
    """Communication context derived from the framework-owned CP mesh."""

    cp_mesh: Any

    @property
    def group(self):
        return self.cp_mesh.get_group()

    @property
    def size(self):
        return self.cp_mesh.size()

    @property
    def rank(self):
        return self.cp_mesh.get_local_rank()


def _sequence_to_head(tensor, context):
    """[B, S/CP, H, D] -> [B, S, H/CP, D]."""
    return _SeqAllToAll.apply(context.group, tensor, 2, 1)


def _head_to_sequence(tensor, context):
    """[B, S, H/CP, D] -> [B, S/CP, H, D]."""
    return _SeqAllToAll.apply(context.group, tensor, 1, 2)


def _gather_sequence(tensor, context):
    """Gather sequence shards and sum their gradients during backward."""
    return torch.cat(
        differentiable_all_gather(
            tensor.contiguous(), group=context.group), dim=1)


def _slice_sequence(tensor, dim, context):
    if tensor.size(dim) % context.size:
        raise ValueError(
            f"sequence length {tensor.size(dim)} is not divisible by "
            f"CP size {context.size}")
    length = tensor.size(dim) // context.size
    return tensor.narrow(
        dim, context.rank * length, length).contiguous()


def _global_seq_len(actual, length, device):
    if actual is None:
        return [length]
    if isinstance(actual, Tensor):
        return actual.to(device=device, dtype=torch.int32)
    return actual


def _slice_sink(tensor, context):
    if tensor is None or tensor.size(2) == 1:
        return tensor
    if tensor.size(2) % context.size:
        raise ValueError(
            f"parameter-sink heads {tensor.size(2)} are not divisible by "
            f"CP size {context.size}")
    count = tensor.size(2) // context.size
    start = context.rank * count
    return tensor[:, :, start:start + count].contiguous()


@dataclass
class _DSATensorContext:
    query: Tensor
    key: Tensor
    q_pe: Tensor
    k_pe: Tensor


_dsa_tensor_context = contextvars.ContextVar(
    "hyper_dsa_tensor_context", default=None)


def _mome_cp_halo_exchange(attention_module, context):
    """Configure cross-rank halo exchange for MoME convolution."""
    original = getattr(attention_module, "_apply_mome")
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def apply_mome_with_halo(hidden_states, mome_mask, conv, use_fused):
        halo = conv.kernel_size[0] - 1
        if halo == 0:
            return original(hidden_states, mome_mask, conv, use_fused)
        if hidden_states.size(1) < halo:
            raise ValueError(
                f"local sequence {hidden_states.size(1)} is shorter than "
                f"MOME halo {halo}")
        tails = differentiable_all_gather(
            hidden_states[:, -halo:].contiguous(), group=context.group)
        mask_tail = mome_mask[:, -halo:].to(
            hidden_states.dtype).contiguous()
        masks = differentiable_all_gather(mask_tail, group=context.group)
        if context.rank == 0:
            left_states = tails[-1] * 0
            left_mask = torch.zeros_like(mask_tail, dtype=torch.bool)
        else:
            left_states = tails[context.rank - 1]
            left_mask = masks[context.rank - 1].bool()
        output = original(
            torch.cat((left_states, hidden_states), dim=1),
            torch.cat((left_mask, mome_mask.bool()), dim=1), conv, use_fused)
        return output[:, halo:].contiguous()

    setattr(apply_mome_with_halo, _ULYSSES_WRAPPED_FLAG, True)
    setattr(attention_module, "_apply_mome", apply_mome_with_halo)


def _mla_cp_alltoall(attention_functions, context):
    """Configure CP all-to-all around the MLA backend."""
    original = attention_functions["npu_fa_rescale"]
    if getattr(original, _ULYSSES_WRAPPED_FLAG, False):
        return

    @functools.wraps(original)
    def mla_with_sequence_head_exchange(
            module, query, key, value, attention_mask, **kwargs):
        if module.attention_type != "mla":
            return original(
                module, query, key, value, attention_mask, **kwargs)
        if not module.apply_FA_rescale or module.use_fused_sink_fa:
            raise ValueError(
                "MLA CP supports only non-fused npu_fa_rescale")
        local_shape = tuple(query.shape)
        query = _sequence_to_head(query, context)
        key = _sequence_to_head(key, context)
        value = _sequence_to_head(value, context)
        length = query.size(1)
        call_kwargs = kwargs.copy()
        call_kwargs.update(
            seq_length=length,
            n_head=query.size(2),
            actual_q_len=_global_seq_len(
                kwargs.get("actual_q_len"), length, query.device),
            actual_kv_len=_global_seq_len(
                kwargs.get("actual_kv_len"), length, query.device),
            param_sink_key=_slice_sink(
                kwargs.get("param_sink_key"), context),
            param_sink_value=_slice_sink(
                kwargs.get("param_sink_value"), context),
        )
        output = _head_to_sequence(
            original(
                module, query, key, value, attention_mask, **call_kwargs),
            context)
        if output.shape[:3] != torch.Size(local_shape[:3]):
            raise RuntimeError(
                f"MLA CP output {tuple(output.shape)} does not restore "
                f"{local_shape}")
        return output

    setattr(mla_with_sequence_head_exchange, _ULYSSES_WRAPPED_FLAG, True)
    attention_functions["npu_fa_rescale"] = mla_with_sequence_head_exchange


def _dsa_cp_alltoall(attention_module, attention_functions, context):
    """Configure CP all-to-all for the DSA indexer, attention, and KL loss."""
    original_indexer = attention_module.dsa_lightning_indexer_forward
    original_sparse = attention_functions["dsa_sparse_attention"]
    original_kl = attention_module.SparseLightningIndexerKLLossTrainFunction

    if not getattr(original_indexer, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_indexer)
        def index_with_gathered_sequence(
                module, index_query, index_key, merge_weight,
                actual_q_len, actual_kv_len):
            index_query = _sequence_to_head(index_query, context)
            merge_weight = _sequence_to_head(
                merge_weight.unsqueeze(-1), context).squeeze(-1)
            index_key = _gather_sequence(index_key, context)
            length = index_query.size(1)
            return original_indexer(
                module, index_query, index_key, merge_weight,
                _global_seq_len(
                    actual_q_len, length, index_query.device),
                _global_seq_len(
                    actual_kv_len, length, index_query.device))

        setattr(index_with_gathered_sequence, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.dsa_lightning_indexer_forward = (
            index_with_gathered_sequence)

    if not getattr(original_sparse, _ULYSSES_WRAPPED_FLAG, False):
        @functools.wraps(original_sparse)
        def sparse_attention_with_gathered_kv(
                module, query, key, value, attention_mask, **kwargs):
            del attention_mask
            local_shape = tuple(query.shape)
            query = _sequence_to_head(query, context)
            q_pe = _sequence_to_head(kwargs["q_pe"], context)
            key = _gather_sequence(key, context)
            value = _gather_sequence(value, context)
            k_pe = _gather_sequence(kwargs["k_pe"], context)
            length = query.size(1)
            call_kwargs = kwargs.copy()
            call_kwargs.update(
                q_pe=q_pe, k_pe=k_pe, seq_length=length,
                n_head=query.size(2),
                actual_q_len=_global_seq_len(
                    kwargs.get("actual_q_len"), length, query.device),
                actual_kv_len=_global_seq_len(
                    kwargs.get("actual_kv_len"), length, query.device),
            )
            old_heads = module.num_heads
            module.num_heads = query.size(2)
            try:
                output, softmax_max, softmax_sum = original_sparse(
                    module, query, key, value, None, **call_kwargs)
            finally:
                module.num_heads = old_heads
            if module.training and not module.freeze_dsa:
                _dsa_tensor_context.set(_DSATensorContext(
                    query=query, key=key, q_pe=q_pe, k_pe=k_pe))
            output = _head_to_sequence(output, context)
            if output.shape[:3] != torch.Size(local_shape[:3]):
                raise RuntimeError(
                    f"DSA CP output {tuple(output.shape)} does not restore "
                    f"{local_shape}")
            return output, softmax_max, softmax_sum

        setattr(sparse_attention_with_gathered_kv, _ULYSSES_WRAPPED_FLAG, True)
        attention_functions["dsa_sparse_attention"] = (
            sparse_attention_with_gathered_kv)

    if not getattr(original_kl, _ULYSSES_WRAPPED_FLAG, False):
        class CPDSAKLLoss:
            """Proxy the DSA KL loss with CP-transformed attention inputs."""

            @staticmethod
            def apply(index_query, index_key, merge_weight, query, key,
                      topk_indices, softmax_max, softmax_sum, query_rope,
                      key_rope, actual_seq_qlen, actual_seq_klen, scale,
                      loss_coeff):
                """Apply the original KL loss with saved global sequence tensors."""
                saved = _dsa_tensor_context.get()
                if saved is None:
                    return original_kl.apply(
                        index_query, index_key, merge_weight, query, key,
                        topk_indices, softmax_max, softmax_sum, query_rope,
                        key_rope, actual_seq_qlen, actual_seq_klen, scale,
                        loss_coeff)
                _dsa_tensor_context.set(None)
                query_tnd, key_tnd, q_pe_tnd, k_pe_tnd = [
                    tensor.flatten(0, 1) for tensor in
                    (saved.query, saved.key, saved.q_pe, saved.k_pe)]
                length = saved.query.size(1)
                return original_kl.apply(
                    index_query, index_key, merge_weight, query_tnd, key_tnd,
                    topk_indices, softmax_max, softmax_sum, q_pe_tnd,
                    k_pe_tnd,
                    _global_seq_len(
                        actual_seq_qlen, length, saved.query.device),
                    _global_seq_len(
                        actual_seq_klen, length, saved.query.device),
                    scale, loss_coeff)

        setattr(CPDSAKLLoss, _ULYSSES_WRAPPED_FLAG, True)
        attention_module.SparseLightningIndexerKLLossTrainFunction = (
            CPDSAKLLoss)


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize and validate a tensor dimension."""
    normalized = dim + ndim if dim < 0 else dim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dimension {dim} is out of range for a {ndim}D tensor")
    return normalized


def _reconstruct_all_to_all(output: Tensor, concat_dim: int) -> Tensor:
    """Move the source-rank axis next to ``concat_dim`` and merge it."""
    rank_axis_position = concat_dim + 1
    permutation = (
        list(range(1, rank_axis_position))
        + [0]
        + list(range(rank_axis_position, output.dim()))
    )
    reconstructed = output.permute(permutation).contiguous()
    shape = list(reconstructed.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return reconstructed.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


def _ulysses_all_to_all(
    tensor: Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    cp_mesh: DeviceMesh,
) -> Tensor:
    """Split one tensor dimension and gather another through all-to-all."""
    cp_size = cp_mesh.size()
    if cp_size <= 1:
        return tensor

    ndim = tensor.dim()
    scatter_dim = _normalize_dim(scatter_dim, ndim)
    gather_dim = _normalize_dim(gather_dim, ndim)
    if scatter_dim == gather_dim:
        raise ValueError("Ulysses scatter_dim and gather_dim must be different")

    shape = list(tensor.shape)
    scatter_size = shape[scatter_dim]
    if scatter_size % cp_size != 0:
        raise ValueError(
            f"tensor dimension {scatter_dim} size ({scatter_size}) must be "
            f"divisible by Ulysses degree ({cp_size})"
        )

    split_shape = (
        shape[:scatter_dim]
        + [cp_size, scatter_size // cp_size]
        + shape[scatter_dim + 1:]
    )
    split_ndim = ndim + 1
    permutation = (
        [scatter_dim]
        + list(range(scatter_dim))
        + list(range(scatter_dim + 1, split_ndim))
    )
    send = tensor.contiguous().reshape(split_shape).permute(permutation).contiguous()
    received = platform.differentiable_all_to_all(
        send, list(send.shape), cp_mesh.get_group())
    return _reconstruct_all_to_all(received, gather_dim)


def ulysses_seq_to_head(
    tensor: Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> Tensor:
    """Convert a sequence shard into a full-sequence head shard.

    Args:
        tensor: Local Q, K, or V tensor.
        seq_dim: Sequence dimension to gather across the CP group.
        head_dim: Head dimension to split across the CP group.
        cp_mesh: One-dimensional CP DeviceMesh.

    Returns:
        Tensor with the sequence dimension gathered and head dimension sharded.
    """
    return _ulysses_all_to_all(
        tensor,
        scatter_dim=head_dim,
        gather_dim=seq_dim,
        cp_mesh=cp_mesh,
    )


def ulysses_head_to_seq(
    tensor: Tensor,
    seq_dim: int,
    head_dim: int,
    cp_mesh: DeviceMesh,
) -> Tensor:
    """Convert a full-sequence head shard back into a sequence shard.

    Args:
        tensor: Local attention output in Ulysses head-sharded layout.
        seq_dim: Full sequence dimension to split across the CP group.
        head_dim: Head dimension to gather across the CP group.
        cp_mesh: One-dimensional CP DeviceMesh.

    Returns:
        Tensor restored to the original local sequence-sharded layout.
    """
    return _ulysses_all_to_all(
        tensor,
        scatter_dim=seq_dim,
        gather_dim=head_dim,
        cp_mesh=cp_mesh,
    )


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
