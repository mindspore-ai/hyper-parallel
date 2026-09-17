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
"""Head-tail sequence folding for DSA context parallelism (causal load balance).

With colossal CP every rank holds one contiguous slice of the sequence, so under a
causal mask rank ``r`` needs the key prefix ``[0, (r + 1) * S / N)``: every DSA kernel
whose cost grows with the key length (lightning indexer, sparse flash attention and
its gradient, the indexer KL loss) gets linearly more expensive towards the last rank.

Folding cuts the sequence into ``2N`` chunks of ``Sf = S / (2N)`` and gives rank ``r``
chunk ``2r`` (block 0) followed by chunk ``2N - 2r - 1`` (block 1). The caller
(MindFormers' CP input preparation) lays the data out this way once; everything
token-wise runs unchanged on the folded order. Only the DSA kernels need to know:
each is called once per block against that block's own causal key prefix,

    block 0 -> (2r + 1) chunks,   block 1 -> 2 (N - r) chunks,

which sum to ``2N + 1`` chunks on every rank.

The key side is still all-gathered along the sequence, so the gathered full-length
key is in *folded* order: slot ``i`` holds true chunk ``src(i) = i`` if ``i`` is even
else ``2N - i`` (an involution). One ``index_select`` per block both restores natural
order and cuts the causal prefix, because every prefix is a whole number of chunks.

Ported from the static-graph implementation (MindFormers ``dsa_lb_stage1``,
``training_graph/transformer/dsa/utils.py``); the chunk convention is the same one the
generic head-tail flash-attention load balance in ``context_parallel.py`` uses.
"""
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from hyper_parallel.platform import get_platform

platform = get_platform()

__all__ = [
    'check_fold_shapes',
    "set_dsa_cp_fold",
    "dsa_cp_fold_enabled",
    "build_fold_order",
    "balanced_prefix_chunks",
    "build_prefix_order",
    "build_prefix_scatter_order",
    "unfold_prefix_pair",
    "unfold_prefix",
    "fold_prefix_grad",
    "split_half",
    "fold_lightning_indexer",
    "fold_sparse_flash_attention",
    "fold_sparse_indexer_kl_loss",
]

# Folding is a property of how the data was sliced, so it is process-wide rather than
# per call: every DSA kernel on a CP-sharded query sees folded blocks once the input
# is folded. A module flag (instead of a thread-local) also covers activation
# recompute and the backward / overlap threads without re-arming anything.
_FOLD_STATE = {"enabled": False}


def set_dsa_cp_fold(enabled: bool) -> None:
    """Turn head-tail folding of DSA CP kernels on or off for this process."""
    _FOLD_STATE["enabled"] = bool(enabled)


def dsa_cp_fold_enabled() -> bool:
    """Whether DSA CP kernels should run once per folded block."""
    return _FOLD_STATE["enabled"]


# ---------------------------------------------------------------------------
# Chunk bookkeeping (pure Python, shared by forward and gradient paths)
# ---------------------------------------------------------------------------

def build_fold_order(seq_shards: int) -> list:
    """Chunk permutation between folded and natural order (an involution).

    Splitting the sequence into ``2N`` chunks, the folded full sequence holds true
    chunk ``c if c is even else 2N - c`` at slot ``c``; applying the same list again
    maps back, so it serves both directions.
    """
    n = int(seq_shards)
    return [c if c % 2 == 0 else 2 * n - c for c in range(2 * n)]


def balanced_prefix_chunks(seq_shard_id: int, seq_shards: int, block_id: int) -> int:
    """Number of ``Sf`` chunks in one folded block's causal key prefix."""
    if block_id == 0:
        return 2 * seq_shard_id + 1
    return 2 * (seq_shards - seq_shard_id)


def build_prefix_order(seq_shard_id: int, seq_shards: int, block_id: int) -> list:
    """Folded-chunk slots whose natural-order concatenation is one block's causal prefix."""
    return build_fold_order(seq_shards)[: balanced_prefix_chunks(seq_shard_id, seq_shards, block_id)]


def build_prefix_scatter_order(seq_shard_id: int, seq_shards: int, block_id: int) -> list:
    """Inverse of :func:`build_prefix_order` for gradients.

    Entry ``s`` names the prefix chunk that carries folded slot ``s``'s gradient; slots
    the prefix does not cover point at ``m`` (one past the prefix), which the caller
    fills with a zero chunk, so a single ``index_select`` scatters and zero-fills.
    """
    prefix = build_prefix_order(seq_shard_id, seq_shards, block_id)
    m = len(prefix)
    inverse = [m] * (2 * seq_shards)
    for i, slot in enumerate(prefix):
        inverse[slot] = i
    return inverse


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

_INDEX_CACHE = {}


def _index_tensor(values: Sequence[int], ref):
    """Cached int32 index tensor for ``index_select`` on ``ref``'s device."""
    key = tuple(values)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        idx = platform.from_numpy(np.asarray(values, dtype=np.int32))
        _INDEX_CACHE[key] = idx
    if type(ref).__module__.startswith("torch"):
        idx = idx.to(ref.device)
    return idx


def _chunked_shape(shape: tuple, seq_dim: int, chunks: int) -> tuple:
    """``shape`` with ``seq_dim`` split into ``(chunks, shape[seq_dim] // chunks)``."""
    return tuple(shape[:seq_dim]) + (chunks, shape[seq_dim] // chunks) + tuple(shape[seq_dim + 1:])


def _unfold_prefix_select(x, seq_shard_id: int, seq_shards: int, block_id: int, seq_dim: int):
    twon = 2 * seq_shards
    if x.shape[seq_dim] % twon != 0:
        raise ValueError(
            f"DSA CP fold needs the key sequence ({x.shape[seq_dim]}) to be a multiple of 2 * cp ({twon})."
        )
    sf = x.shape[seq_dim] // twon
    prefix = build_prefix_order(seq_shard_id, seq_shards, block_id)
    y = x.reshape(_chunked_shape(x.shape, seq_dim, twon))
    y = y.index_select(seq_dim, _index_tensor(prefix, x))
    out_shape = list(x.shape)
    out_shape[seq_dim] = len(prefix) * sf
    return y.reshape(tuple(out_shape))


class _UnfoldPrefix(platform.Function):
    """``index_select`` forward whose backward is another ``index_select``, not ``index_add``.

    Autograd's own backward for ``index_select`` is ``index_add`` onto a zero tensor of the
    full key length. On Ascend ``aclnnIndexAdd`` has no bf16 kernel, so every call casts the
    whole key gradient bf16->fp32 twice and back once -- regardless of how short the prefix
    is. At 128k (N=128) that was ~490ms/rank per step, more than the fold's gathers
    themselves. The prefix slots are a set of distinct chunks, so the inverse is exactly
    ``fold_prefix_grad``: append one zero chunk and gather, staying in bf16.
    """

    @staticmethod
    def forward(ctx, x, seq_shard_id, seq_shards, block_id, seq_dim):  # pylint: disable=arguments-differ
        ctx.fold_args = (seq_shard_id, seq_shards, block_id, x.shape[seq_dim], seq_dim)
        return _unfold_prefix_select(x, seq_shard_id, seq_shards, block_id, seq_dim)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        seq_shard_id, seq_shards, block_id, full_len, seq_dim = ctx.fold_args
        grad = fold_prefix_grad(grad_output, seq_shard_id, seq_shards, block_id, full_len, seq_dim)
        return grad, None, None, None, None


def unfold_prefix(x, seq_shard_id: int, seq_shards: int, block_id: int, seq_dim: int = 1):
    """Natural-order causal key prefix of one folded block, from a folded full-length key.

    One ``index_select`` over whole chunks does both the un-fold and the prefix cut.
    Differentiable: the gradient goes back onto the folded layout through
    ``fold_prefix_grad`` (see ``_UnfoldPrefix``).
    """
    if x is None:
        return None
    return _UnfoldPrefix.apply(x, seq_shard_id, seq_shards, block_id, seq_dim)


class _UnfoldPrefixPair(platform.Function):
    """Both blocks' key prefixes from one folded key, with a merged backward.

    Each block's prefix starts at natural chunk 0, so the shorter one is the head of the
    longer one. Two independent ``_UnfoldPrefix`` backwards would each concat a zero chunk
    and gather at full key length, then add the two full-length results; here the short
    gradient is added onto the long gradient's head (short length) and one concat + one
    gather scatters the sum. Forward stays two gathers: its cost scales with the output,
    and making the short prefix a view of the long one would alias two autograd outputs.
    """

    @staticmethod
    def forward(ctx, x, seq_shard_id, seq_shards, seq_dim):  # pylint: disable=arguments-differ
        full_len = x.shape[seq_dim]
        sf = full_len // (2 * seq_shards)
        m = (balanced_prefix_chunks(seq_shard_id, seq_shards, 0),
             balanced_prefix_chunks(seq_shard_id, seq_shards, 1))
        ctx.fold_args = (seq_shard_id, seq_shards, full_len, seq_dim, sf, m)
        return (_unfold_prefix_select(x, seq_shard_id, seq_shards, 0, seq_dim),
                _unfold_prefix_select(x, seq_shard_id, seq_shards, 1, seq_dim))

    @staticmethod
    def backward(ctx, grad0, grad1):  # pylint: disable=arguments-differ
        seq_shard_id, seq_shards, full_len, seq_dim, sf, m = ctx.fold_args
        long_id = 0 if m[0] >= m[1] else 1
        grads = (grad0, grad1)
        g_long, g_short = grads[long_id], grads[1 - long_id]
        if g_long is None and g_short is None:
            return None, None, None, None
        if g_long is None:
            return fold_prefix_grad(g_short, seq_shard_id, seq_shards, 1 - long_id, full_len, seq_dim), \
                None, None, None
        parts = []
        short_len = m[1 - long_id] * sf
        long_len = m[long_id] * sf
        if g_short is not None:
            parts.append(g_long.narrow(seq_dim, 0, short_len) + g_short)
            if long_len > short_len:
                parts.append(g_long.narrow(seq_dim, short_len, long_len - short_len))
        else:
            parts.append(g_long)
        zero_shape = list(g_long.shape)
        zero_shape[seq_dim] = sf
        parts.append(platform.zeros(tuple(zero_shape), dtype=g_long.dtype, device=g_long.device))
        g = platform.cat(parts, dim=seq_dim)
        g = g.reshape(_chunked_shape(g.shape, seq_dim, m[long_id] + 1))
        g = g.index_select(seq_dim, _index_tensor(
            build_prefix_scatter_order(seq_shard_id, seq_shards, long_id), g))
        out_shape = list(g_long.shape)
        out_shape[seq_dim] = full_len
        return g.reshape(tuple(out_shape)), None, None, None


def unfold_prefix_pair(x, seq_shard_id: int, seq_shards: int, seq_dim: int = 1):
    """``(unfold_prefix(x, .., 0), unfold_prefix(x, .., 1))`` with one merged backward."""
    if x is None:
        return None, None
    twon = 2 * seq_shards
    if x.shape[seq_dim] % twon != 0:
        raise ValueError(
            f"DSA CP fold needs the key sequence ({x.shape[seq_dim]}) to be a multiple of 2 * cp ({twon})."
        )
    return _UnfoldPrefixPair.apply(x, seq_shard_id, seq_shards, seq_dim)


def fold_prefix_grad(grad, seq_shard_id: int, seq_shards: int, block_id: int, full_len: int, seq_dim: int = 1):
    """Scatter a key-side gradient computed on one block's prefix back onto the folded layout.

    Needed where a fused kernel returns the key gradient itself (the indexer KL loss):
    that gradient is ordered like the prefix, so padding it at the tail -- what the
    unfolded path does -- would land odd chunks on the wrong tokens and zero chunks the
    prefix did cover.
    """
    twon = 2 * seq_shards
    sf = full_len // twon
    m = grad.shape[seq_dim] // sf
    g = grad.reshape(_chunked_shape(grad.shape, seq_dim, m))
    zero_shape = list(g.shape)
    zero_shape[seq_dim] = 1
    g = platform.cat([g, platform.zeros(tuple(zero_shape), dtype=g.dtype, device=g.device)], dim=seq_dim)
    g = g.index_select(seq_dim, _index_tensor(build_prefix_scatter_order(seq_shard_id, seq_shards, block_id), g))
    out_shape = list(grad.shape)
    out_shape[seq_dim] = full_len
    return g.reshape(tuple(out_shape))


def split_half(x, dim: int) -> Tuple[Optional[object], Optional[object]]:
    """Split a query-side tensor into its two folded blocks (views)."""
    if x is None:
        return None, None
    half = x.shape[dim] // 2
    return x.narrow(dim, 0, half), x.narrow(dim, half, x.shape[dim] - half)


def _cat_pair(a, b, dim: int):
    return platform.cat([a, b], dim=dim)


# ---------------------------------------------------------------------------
# Per-kernel twice-call wrappers (BSND). ``func`` is the local kernel callable the
# distributed op would otherwise have called once on the full local query.
# ---------------------------------------------------------------------------

def check_fold_shapes(local_q, key, seq_shards: int, seq_dim: int = 1) -> None:
    """Catch an N mismatch between who sliced the data and who folds it.

    ``seq_shards`` is how many ways the *sequence* is split, which is **not always CP**:
    stage 2 (non-TND) runs the DSA kernels sequence-parallel over TP as well, so the mesh
    handed to these styles is the flattened ``(cp, tp)`` one and N = cp*tp; stage 1 would use
    cp alone. hyper-parallel never learns which it is -- it only reads "my S is split N ways"
    off the DTensor layout -- so the value MindFormers used to slice the input and the value
    read back here must agree, and nothing but this check enforces that.

    A mismatch is otherwise **silent**: the existing ``key_len % (2N) == 0`` guard still passes
    when N is off by a factor of two, ``split_half`` cuts the local query in half regardless,
    and the result is merely a wrong causal prefix -- a loss that is slightly off rather than
    a crash or a NaN.

    The invariant: the local query holds exactly the two folded chunks this rank owns
    (``2 * sf``) while the key holds all ``2 * N`` of them, hence ``local_q_len * N == key_len``.
    """
    if local_q is None or key is None:
        return
    q_len = int(local_q.shape[seq_dim])
    k_len = int(key.shape[seq_dim])
    if q_len * int(seq_shards) != k_len:
        raise ValueError(
            f"DSA CP fold shape mismatch: local query S={q_len} with {seq_shards} sequence "
            f"shards implies a key of {q_len * int(seq_shards)}, but got {k_len}. The sequence "
            f"split used to slice the input disagrees with the one on the DTensor layout "
            f"(cp vs cp*tp?); folding would silently read the wrong causal prefix."
        )


def fold_lightning_indexer(func: Callable, seq_shard_id: int, seq_shards: int, *args, **kwargs):
    """``lightning_indexer(query, key, weights, ...)``: Top-K indices/scores per block.

    The indices of each block address that block's natural-order key prefix, which is
    exactly the key the sparse attention and the KL loss see for the same block.
    """
    check_fold_shapes(args[0], args[1], seq_shards)
    q0, q1 = split_half(args[0], 1)
    w0, w1 = split_half(args[2], 1)
    key = args[1]
    rest = args[3:]
    k0, k1 = unfold_prefix_pair(key, seq_shard_id, seq_shards)
    out0 = func(q0, k0, w0, *rest, **kwargs)
    out1 = func(q1, k1, w1, *rest, **kwargs)
    if not isinstance(out0, (tuple, list)):
        return _cat_pair(out0, out1, 1)
    return type(out0)(_cat_pair(a, b, 1) for a, b in zip(out0, out1))


def fold_sparse_flash_attention(func: Callable, seq_shard_id: int, seq_shards: int, *args, **kwargs):
    """``sparse_flash_attention(query, key, value, sparse_indices, scale, **kw)`` per block.

    ``attention_out`` is stitched on seq dim 1; ``softmax_max/sum`` are
    ``(B, N2, S1, G)`` and stitch on dim 2.
    """
    q0, q1 = split_half(args[0], 1)
    t0, t1 = split_half(args[3], 1)
    key, value = args[1], args[2]
    rest = args[4:]
    qr0, qr1 = split_half(kwargs.get("query_rope"), 1)
    key_rope = kwargs.get("key_rope")

    keys = unfold_prefix_pair(key, seq_shard_id, seq_shards)
    values = unfold_prefix_pair(value, seq_shard_id, seq_shards)
    key_ropes = unfold_prefix_pair(key_rope, seq_shard_id, seq_shards)

    def _call(block_id, q, topk, q_rope):
        kw = dict(kwargs)
        if "query_rope" in kw:
            kw["query_rope"] = q_rope
        if "key_rope" in kw:
            kw["key_rope"] = key_ropes[block_id]
        return func(q, keys[block_id], values[block_id], topk, *rest, **kw)

    out0 = _call(0, q0, t0, qr0)
    out1 = _call(1, q1, t1, qr1)
    if not isinstance(out0, (tuple, list)):
        return _cat_pair(out0, out1, 1)
    stitched = [_cat_pair(out0[0], out1[0], 1)]
    stitched.extend(_cat_pair(a, b, 2) for a, b in zip(out0[1:], out1[1:]))
    return type(out0)(stitched)


def fold_sparse_indexer_kl_loss(func: Callable, seq_shard_id: int, seq_shards: int, *args):
    """Sparse indexer KL loss per block, MindSpore positional form (17 args).

    Layout of ``args``: query, key, query_index, key_index, weights, sparse_indices,
    softmax_max, softmax_sum, scale, query_rope, key_rope, then scalars/options.
    Query-side inputs split on seq dim 1 except the softmax stats ``(B, N2, S1, G)`` on
    dim 2. Returns ``(d_query_index, d_key_index, d_weights, loss)`` where
    ``d_key_index`` is full-length on the folded layout and ``loss`` is the sum of the
    two blocks (the kernel returns an un-normalised sum over its queries).
    """
    q_side_dims = {0: 1, 2: 1, 4: 1, 5: 1, 6: 2, 7: 2, 9: 1}
    key_side = (1, 3, 10)
    full_len = args[3].shape[1]
    halves = {i: split_half(args[i], d) for i, d in q_side_dims.items()}
    prefixes = {i: unfold_prefix_pair(args[i], seq_shard_id, seq_shards) for i in key_side}

    def _block(block_id):
        call = list(args)
        for i in q_side_dims:
            call[i] = halves[i][block_id]
        for i in key_side:
            call[i] = prefixes[i][block_id]
        return func(*call)

    d_qi0, d_ki0, d_w0, loss0 = _block(0)
    d_qi1, d_ki1, d_w1, loss1 = _block(1)
    d_key_index = (fold_prefix_grad(d_ki0, seq_shard_id, seq_shards, 0, full_len)
                   + fold_prefix_grad(d_ki1, seq_shard_id, seq_shards, 1, full_len))
    return _cat_pair(d_qi0, d_qi1, 1), d_key_index, _cat_pair(d_w0, d_w1, 1), loss0 + loss1
