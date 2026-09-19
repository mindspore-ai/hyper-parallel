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
    'dsa_cp_fold_requester',
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
    "natural_prefix",
    "natural_prefix_grad",
    "fold_dense_lightning_indexer_softmax_lse",
    "fold_dense_indexer_kl_loss",
    "tnd_block_seq_lens",
]

# Folding is a property of how the data was sliced, so it is process-wide rather than
# per call: every DSA kernel on a CP-sharded query sees folded blocks once the input
# is folded. A module flag (instead of a thread-local) also covers activation
# recompute and the backward / overlap threads without re-arming anything.
_FOLD_STATE = {"enabled": False, "requester": ""}


def set_dsa_cp_fold(enabled: bool, requester: str = "") -> None:
    """Turn head-tail folding of DSA CP kernels on or off for this process.

    ``requester`` names the style that asked, so a later style asking for the opposite can
    say who it is fighting with. See :func:`dsa_cp_fold_requester`.
    """
    _FOLD_STATE["enabled"] = bool(enabled)
    _FOLD_STATE["requester"] = requester if enabled else ""


def dsa_cp_fold_requester() -> str:
    """Name of the style that last turned folding on, or "" when it is off."""
    return _FOLD_STATE.get("requester", "")


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
# TND: per-block ``actual_seq_len`` recomputation
# ---------------------------------------------------------------------------

# The klen-vs-T check below costs one device->host sync, so do it once per process: the
# property it guards (whether the data pipeline pads the tail) is a property of the run, not
# of the call.
_KLEN_CHECKED = [False]


def _check_klen_covers_full_len(actual_seq_klen, full_len) -> None:
    """Refuse a padded key tail, which the fold's per-block cumulative lengths cannot express.

    ``tnd_block_seq_lens`` leaves ``K_b``'s last entry at ``min(C_last, e_b)``. The kernel
    requires the accumulated key length to equal the key tensor's ``T``, which holds only when
    ``C_last == T``. A padded tail (``eod_pad_length`` rounds the packed sample up) makes
    ``C_last < T``, and the last block then fails inside the kernel with a message about
    sequence lengths that says nothing about folding or padding. Fail here instead, once.
    """
    if _KLEN_CHECKED[0]:
        return
    _KLEN_CHECKED[0] = True
    try:
        last = int(actual_seq_klen[-1])
    except Exception:  # pylint: disable=broad-except
        return          # cannot read it (traced/placeholder tensor) -- leave it to the kernel
    if last != int(full_len):
        raise ValueError(
            f"DSA CP head-tail fold needs the key cumulative lengths to cover the whole key "
            f"tensor, but actual_seq_klen[-1]={last} while the key length is {int(full_len)}. "
            f"A padded tail (eod_pad_length) is not supported by the folded per-block "
            f"cumulative lengths: the kernel checks that the accumulated key length equals T. "
            f"Pack without tail padding, or turn dsa_enable_load_balance off.")


def tnd_block_seq_lens(actual_seq_qlen, actual_seq_klen, full_len: int,
                       seq_shard_id: int, seq_shards: int, block_id: int):
    """Per-block ``(actual_seq_qlen, actual_seq_klen)`` for one folded TND block.

    This is the **only new semantics** the TND fold needs. Under BSND the causal prefix is
    expressed by the key tensor's length alone; under TND the kernel derives each query's
    causal window from the cumulative document ends instead, so a block that holds tokens
    ``[s_b, e_b)`` of the global sequence and is handed the natural-order key prefix
    ``[0, e_b)`` must be given cumulative lengths *restated in that block's coordinates*::

        Q_b[i] = clamp(C_i - s_b, 0, Sf)     # doc i's tokens that fall inside this block
        K_b[i] = clamp(C_i,       0, e_b)    # doc i's tokens from its start up to the prefix end

    where ``C`` are the global cumulative document ends and ``Sf = e_b - s_b``. The kernels
    pair q document ``i`` with k document ``i``, so restating both sides keeps every query on
    its own document and drops nothing: every (query, key) pair the unfolded call would form is
    formed exactly once across the two blocks, verified against a numpy reference and measured
    bit-identical on the real kernels.

    ``K_b``'s last entry comes out as ``e_b`` on its own (the global total is never smaller
    than a prefix end), which is what the kernel requires -- it checks that the accumulated
    key length equals the key tensor's ``T``.

    **The caller must pass the un-adjusted, sequence-global cumulative lengths**, not the
    per-rank ones ``_adjust_tnd_seq_lens`` produces: that helper assumes a *contiguous* CP
    slice (``offset = local_T * cp_rank``), which the fold deliberately breaks. Feeding its
    output in here is silent -- the shapes all still fit and only the causal prefix is wrong.

    Args:
        actual_seq_qlen: Global cumulative query document ends (int32 Tensor).
        actual_seq_klen: Global cumulative key document ends (int32 Tensor).
        full_len: Global sequence length ``T`` (the folded key's length).
        seq_shard_id: This rank's position among the ``seq_shards`` sequence shards.
        seq_shards: ``N``; the sequence is cut into ``2N`` chunks.
        block_id: 0 for the head chunk (``2r``), 1 for the tail chunk (``2N-1-2r``).

    Returns:
        tuple: ``(Q_b, K_b)`` as int32 tensors of the same length as the inputs.
    """
    if actual_seq_qlen is None or actual_seq_klen is None:
        return actual_seq_qlen, actual_seq_klen
    _check_klen_covers_full_len(actual_seq_klen, full_len)
    sf = int(full_len) // (2 * int(seq_shards))
    # The block's causal prefix spans this many Sf chunks, so its own chunk ends the prefix.
    end = balanced_prefix_chunks(seq_shard_id, seq_shards, block_id) * sf
    start = end - sf
    q_block = platform.tensor_type_cast((actual_seq_qlen - start).clamp(0, sf), 'int32')
    k_block = platform.tensor_type_cast(actual_seq_klen.clamp(0, end), 'int32')
    return q_block, k_block


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


def fold_lightning_indexer(func: Callable, seq_shard_id: int, seq_shards: int, *args,
                           fold_layout: str = "BSND", **kwargs):
    """``lightning_indexer(query, key, weights, ...)``: Top-K indices/scores per block.

    The indices of each block address that block's natural-order key prefix, which is
    exactly the key the sparse attention and the KL loss see for the same block -- and
    because the prefix is a natural-order truncation, an index is the same number in the
    prefix as in the whole sequence, so nothing has to be remapped afterwards.

    ``fold_layout='TND'``: the token axis is dim 0 and the causal window comes from
    ``actual_seq_lengths_query/key``, which are restated per block (see
    :func:`tnd_block_seq_lens`). The caller must hand over the **global** cumulative
    lengths, not ``_adjust_tnd_seq_lens``'s contiguous-slice output.
    """
    tnd = fold_layout == "TND"
    seq_dim = 0 if tnd else 1
    check_fold_shapes(args[0], args[1], seq_shards, seq_dim)
    q0, q1 = split_half(args[0], seq_dim)
    w0, w1 = split_half(args[2], seq_dim)
    key = args[1]
    rest = args[3:]
    k0, k1 = unfold_prefix_pair(key, seq_shard_id, seq_shards, seq_dim)

    def _kwargs(block_id):
        if not tnd:
            return kwargs
        q_len, k_len = tnd_block_seq_lens(
            kwargs.get("actual_seq_lengths_query"), kwargs.get("actual_seq_lengths_key"),
            key.shape[0], seq_shard_id, seq_shards, block_id)
        return {**kwargs, "actual_seq_lengths_query": q_len, "actual_seq_lengths_key": k_len}

    out0 = func(q0, k0, w0, *rest, **_kwargs(0))
    out1 = func(q1, k1, w1, *rest, **_kwargs(1))
    if not isinstance(out0, (tuple, list)):
        return _cat_pair(out0, out1, seq_dim)
    return type(out0)(_cat_pair(a, b, seq_dim) for a, b in zip(out0, out1))


def fold_sparse_flash_attention(func: Callable, seq_shard_id: int, seq_shards: int, *args,
                                fold_layout: str = "BSND", **kwargs):
    """``sparse_flash_attention(query, key, value, sparse_indices, scale, **kw)`` per block.

    BSND: ``attention_out`` is stitched on seq dim 1; ``softmax_max/sum`` are
    ``(B, N2, S1, G)`` and stitch on dim 2.

    TND: query-side inputs are on dim 0 and ``attention_out`` is ``(T, N, D)`` so it also
    stitches on dim 0, while ``softmax_max/sum`` are ``(1, T, N)`` and stitch on dim 1.
    The per-block ``actual_seq_lengths_query/kv`` come from :func:`tnd_block_seq_lens`.
    """
    tnd = fold_layout == "TND"
    seq_dim = 0 if tnd else 1
    stats_dim = 1 if tnd else 2
    check_fold_shapes(args[0], args[1], seq_shards, seq_dim)
    q0, q1 = split_half(args[0], seq_dim)
    t0, t1 = split_half(args[3], seq_dim)
    key, value = args[1], args[2]
    rest = args[4:]
    qr0, qr1 = split_half(kwargs.get("query_rope"), seq_dim)
    key_rope = kwargs.get("key_rope")

    keys = unfold_prefix_pair(key, seq_shard_id, seq_shards, seq_dim)
    values = unfold_prefix_pair(value, seq_shard_id, seq_shards, seq_dim)
    key_ropes = unfold_prefix_pair(key_rope, seq_shard_id, seq_shards, seq_dim)

    def _call(block_id, q, topk, q_rope):
        kw = dict(kwargs)
        if "query_rope" in kw:
            kw["query_rope"] = q_rope
        if "key_rope" in kw:
            kw["key_rope"] = key_ropes[block_id]
        if tnd:
            q_len, k_len = tnd_block_seq_lens(
                kw.get("actual_seq_lengths_query"), kw.get("actual_seq_lengths_kv"),
                key.shape[0], seq_shard_id, seq_shards, block_id)
            kw["actual_seq_lengths_query"] = q_len
            kw["actual_seq_lengths_kv"] = k_len
        return func(q, keys[block_id], values[block_id], topk, *rest, **kw)

    out0 = _call(0, q0, t0, qr0)
    out1 = _call(1, q1, t1, qr1)
    if not isinstance(out0, (tuple, list)):
        return _cat_pair(out0, out1, seq_dim)
    stitched = [_cat_pair(out0[0], out1[0], seq_dim)]
    stitched.extend(_cat_pair(a, b, stats_dim) for a, b in zip(out0[1:], out1[1:]))
    return type(out0)(stitched)


# ``args`` positions of the MindSpore positional form of the sparse indexer KL loss:
#   0 query, 1 key, 2 query_index, 3 key_index, 4 weights, 5 sparse_indices,
#   6 softmax_max, 7 softmax_sum, 8 scale, 9 query_rope, 10 key_rope,
#   11 actual_seq_qlen, 12 actual_seq_klen, 13 layout, 14 sparse_mode, 15/16 pre/next tokens.
_KL_ACTUAL_SEQ_QLEN_IDX = 11
_KL_ACTUAL_SEQ_KLEN_IDX = 12


def fold_sparse_indexer_kl_loss(func: Callable, seq_shard_id: int, seq_shards: int, *args,
                                fold_layout: str = "BSND"):
    """Sparse indexer KL loss per block, MindSpore positional form (17 args).

    Query-side inputs split on the sequence axis except the softmax stats, which carry the
    token axis one dim later. Returns ``(d_query_index, d_key_index, d_weights, loss)``
    where ``d_key_index`` is full-length on the folded layout and ``loss`` is the sum of
    the two blocks (the kernel returns an un-normalised sum over its queries).

    Per layout:

    * ``BSND``: query side on dim 1, softmax stats ``(B, N2, S1, G)`` on dim 2.
    * ``TND``: query side on dim 0, softmax stats ``(1, T, N)`` on dim 1, and the
      per-block ``actual_seq_qlen/klen`` replace args 11/12 (:func:`tnd_block_seq_lens`).

    ``d_key_index`` must be scattered with :func:`fold_prefix_grad` rather than tail-padded:
    the kernel returns it ordered like the *prefix*, so padding at the tail would land odd
    chunks on the wrong tokens.
    """
    tnd = fold_layout == "TND"
    seq_dim = 0 if tnd else 1
    stats_dim = 1 if tnd else 2
    check_fold_shapes(args[0], args[1], seq_shards, seq_dim)
    q_side_dims = {0: seq_dim, 2: seq_dim, 4: seq_dim, 5: seq_dim,
                   6: stats_dim, 7: stats_dim, 9: seq_dim}
    key_side = (1, 3, 10)
    full_len = args[3].shape[seq_dim]
    halves = {i: split_half(args[i], d) for i, d in q_side_dims.items()}
    prefixes = {i: unfold_prefix_pair(args[i], seq_shard_id, seq_shards, seq_dim) for i in key_side}

    def _block(block_id):
        call = list(args)
        for i in q_side_dims:
            call[i] = halves[i][block_id]
        for i in key_side:
            call[i] = prefixes[i][block_id]
        if tnd:
            call[_KL_ACTUAL_SEQ_QLEN_IDX], call[_KL_ACTUAL_SEQ_KLEN_IDX] = tnd_block_seq_lens(
                args[_KL_ACTUAL_SEQ_QLEN_IDX], args[_KL_ACTUAL_SEQ_KLEN_IDX],
                full_len, seq_shard_id, seq_shards, block_id)
        return func(*call)

    d_qi0, d_ki0, d_w0, loss0 = _block(0)
    d_qi1, d_ki1, d_w1, loss1 = _block(1)
    d_key_index = (fold_prefix_grad(d_ki0, seq_shard_id, seq_shards, 0, full_len, seq_dim)
                   + fold_prefix_grad(d_ki1, seq_shard_id, seq_shards, 1, full_len, seq_dim))
    return (_cat_pair(d_qi0, d_qi1, seq_dim), d_key_index,
            _cat_pair(d_w0, d_w1, seq_dim), loss0 + loss1)



# ---------------------------------------------------------------------------
# Natural-order key helpers (stage 1). The dense warm-up does NOT fold the trunk --
# only the query-side tensors entering the DSA kernels are folded (mirroring the
# static graph's per-layer ``dsa_fold_q`` / ``dsa_fold_sm``), so the key side arrives
# in natural order and a block's causal prefix is simply its first ``m`` chunks.
# ---------------------------------------------------------------------------

def natural_prefix(x, seq_shard_id: int, seq_shards: int, block_id: int, seq_dim: int = 1):
    """One folded block's causal key prefix out of a natural-order full-length key.

    Folded block ``b`` of rank ``r`` is natural chunk ``2r`` (b=0) or ``2N-2r-1`` (b=1),
    whose causal prefix is a *contiguous* run from the start -- so this is a narrow, not
    the ``index_select`` the folded-key path needs.
    """
    if x is None:
        return None
    twon = 2 * seq_shards
    if x.shape[seq_dim] % twon != 0:
        raise ValueError(
            f"DSA CP fold needs the key sequence ({x.shape[seq_dim]}) to be a multiple of 2 * cp ({twon})."
        )
    sf = x.shape[seq_dim] // twon
    return x.narrow(seq_dim, 0, balanced_prefix_chunks(seq_shard_id, seq_shards, block_id) * sf)


def natural_prefix_grad(grad, full_len: int, seq_dim: int = 1):
    """Pad a prefix-shaped key gradient back to full length with zeros at the tail.

    The natural-order prefix is a leading slice, so unlike ``fold_prefix_grad`` nothing
    has to be scattered: the positions the prefix did not cover are exactly the tail.
    """
    pad_len = full_len - grad.shape[seq_dim]
    if pad_len <= 0:
        return grad
    pad_shape = list(grad.shape)
    pad_shape[seq_dim] = pad_len
    zero = platform.zeros(tuple(pad_shape), dtype=grad.dtype, device=grad.device)
    return platform.cat([grad, zero], dim=seq_dim)


# ``args`` positions of the MindSpore positional form of the dense indexer forward:
#   0 query_index, 1 key_index, 2 weights, 3 actual_seq_qlen, 4 actual_seq_klen,
#   5 layout, 6 sparse_mode, 7/8 pre/next tokens.
_DENSE_LSE_ACTUAL_SEQ_QLEN_IDX = 3
_DENSE_LSE_ACTUAL_SEQ_KLEN_IDX = 4


def fold_dense_lightning_indexer_softmax_lse(func: Callable, seq_shard_id: int, seq_shards: int, *args,
                                             fold_layout: str = "BSND", **kwargs):
    """Dense (stage-1) indexer softmax statistics per block.

    ``args``: query_index, key_index, weights, then scalars/options. The query side holds
    this rank's two folded blocks; the key side is **natural order** (see
    :func:`natural_prefix`), because stage 1 folds only the query-side tensors.

    BSND: query-side inputs split on dim 1 and both outputs ``(B, Nidx2, S1)`` stitch on dim 2.
    TND: the token axis is dim 0, the outputs are ``(Nidx2, T1)`` and stitch on dim 1, and the
    causal window comes from ``actual_seq_qlen/klen`` (positional 3/4), restated per block by
    :func:`tnd_block_seq_lens` -- which needs the **global** cumulative lengths, not
    ``_adjust_tnd_seq_lens``'s contiguous-slice output.
    """
    tnd = fold_layout == "TND"
    seq_dim = 0 if tnd else 1
    stats_dim = 1 if tnd else 2
    if tnd and len(args) <= _DENSE_LSE_ACTUAL_SEQ_KLEN_IDX:
        raise NotImplementedError(
            "DSA CP head-tail fold under TND needs the MindSpore positional signature of the "
            "dense indexer forward, which carries actual_seq_qlen/klen.")
    check_fold_shapes(args[0], args[1], seq_shards, seq_dim)
    q0, q1 = split_half(args[0], seq_dim)
    w0, w1 = split_half(args[2], seq_dim)

    def _call(block_id, query_index, weights):
        rest = list(args[3:])
        if tnd:
            q_len, k_len = tnd_block_seq_lens(
                args[_DENSE_LSE_ACTUAL_SEQ_QLEN_IDX], args[_DENSE_LSE_ACTUAL_SEQ_KLEN_IDX],
                args[1].shape[seq_dim], seq_shard_id, seq_shards, block_id)
            rest[_DENSE_LSE_ACTUAL_SEQ_QLEN_IDX - 3] = q_len
            rest[_DENSE_LSE_ACTUAL_SEQ_KLEN_IDX - 3] = k_len
        key_prefix = natural_prefix(args[1], seq_shard_id, seq_shards, block_id, seq_dim)
        return func(query_index, key_prefix, weights, *rest, **kwargs)

    out0 = _call(0, q0, w0)
    out1 = _call(1, q1, w1)
    return type(out0)(_cat_pair(a, b, stats_dim) for a, b in zip(out0, out1))


# ``args`` positions of the MindSpore positional form of the dense indexer KL loss:
#   0 query, 1 key, 2 query_index, 3 key_index, 4 weights, 5 softmax_max, 6 softmax_sum,
#   7 softmax_max_index, 8 softmax_sum_index, 9 scale, 10 query_rope, 11 key_rope,
#   12 actual_seq_qlen, 13 actual_seq_klen, 14 layout, 15 sparse_mode, 16/17 pre/next tokens.
# Two more query-side statistics than the sparse loss, hence the shifted seq-len indices.
_DENSE_KL_ACTUAL_SEQ_QLEN_IDX = 12
_DENSE_KL_ACTUAL_SEQ_KLEN_IDX = 13


def fold_dense_indexer_kl_loss(func: Callable, seq_shard_id: int, seq_shards: int, *args,
                               fold_layout: str = "BSND"):
    """Dense (stage-1) indexer KL loss per block, MindSpore positional form (18 args).

    Unlike the sparse loss this one has no ``sparse_indices`` and carries two extra
    query-side statistics (the indexer's own softmax max/sum, from
    :func:`fold_dense_lightning_indexer_softmax_lse`).

    BSND: query-side inputs split on dim 1, except the four statistics -- the attention's
    ``(B, N2, S1, G)`` and the indexer's ``(B, Nidx2, S1)`` -- which split on dim 2.
    TND: the token axis is dim 0 and every statistic is ``(N, T1, ...)``, so they split on
    dim 1; the per-block ``actual_seq_qlen/klen`` come from :func:`tnd_block_seq_lens`.

    The key side is **natural order**: stage 1 folds only the query-side tensors, so each
    block's causal prefix is a leading narrow and ``d_key_index`` pads back at the tail.

    Returns ``(d_query_index, d_key_index, d_weights, loss)`` with ``d_key_index`` full
    length and ``loss`` the sum over the two blocks.
    """
    tnd = fold_layout == "TND"
    seq_dim = 0 if tnd else 1
    stats_dim = 1 if tnd else 2
    q_side_dims = {0: seq_dim, 2: seq_dim, 4: seq_dim, 10: seq_dim,
                   5: stats_dim, 6: stats_dim, 7: stats_dim, 8: stats_dim}
    key_side = (1, 3, 11)
    full_len = args[3].shape[seq_dim]
    check_fold_shapes(args[0], args[1], seq_shards, seq_dim)
    halves = {i: split_half(args[i], d) for i, d in q_side_dims.items()}

    def _block(block_id):
        call = list(args)
        for i in q_side_dims:
            call[i] = halves[i][block_id]
        for i in key_side:
            call[i] = natural_prefix(args[i], seq_shard_id, seq_shards, block_id, seq_dim)
        if tnd:
            call[_DENSE_KL_ACTUAL_SEQ_QLEN_IDX], call[_DENSE_KL_ACTUAL_SEQ_KLEN_IDX] = tnd_block_seq_lens(
                args[_DENSE_KL_ACTUAL_SEQ_QLEN_IDX], args[_DENSE_KL_ACTUAL_SEQ_KLEN_IDX],
                full_len, seq_shard_id, seq_shards, block_id)
        return func(*call)

    d_qi0, d_ki0, d_w0, loss0 = _block(0)
    d_qi1, d_ki1, d_w1, loss1 = _block(1)
    d_key_index = (natural_prefix_grad(d_ki0, full_len, seq_dim)
                   + natural_prefix_grad(d_ki1, full_len, seq_dim))
    return (_cat_pair(d_qi0, d_qi1, seq_dim), d_key_index,
            _cat_pair(d_w0, d_w1, seq_dim), loss0 + loss1)
