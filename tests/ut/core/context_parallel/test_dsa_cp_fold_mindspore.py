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
"""UT: head-tail folding of DSA CP kernels (:mod:`hyper_parallel.core.shard.ops.dsa_cp_fold`).

The real kernels need an NPU, so the twice-call wrappers are checked against toy
kernels with the same contract: right-down-aligned causal attention of a query block
against the key it is given. Simulating every CP rank on one host, the folded result
must reproduce a single full-sequence call -- outputs row by row, the key gradient
after summing the ranks, and the loss after summing the ranks.
"""
import os

import numpy as np
import pytest

pytest.importorskip("mindspore")

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"
from tests.ut.platform.mindspore._ensure_mindspore_platform import (  # noqa: E402
    ensure_mindspore_platform_for_context_parallel,
)

ensure_mindspore_platform_for_context_parallel()

import mindspore as ms  # noqa: E402
from mindspore import Tensor, ops  # noqa: E402

from hyper_parallel.core.shard.ops import dsa_cp_fold as fold  # noqa: E402

B, N_HEADS, D = 1, 2, 4


def _chunk_ids(cp_rank, cp_size):
    return 2 * cp_rank, 2 * cp_size - 2 * cp_rank - 1


def _fold_rows(x, cp_rank, cp_size, axis=1):
    """Rank ``cp_rank``'s local (folded) slice of a natural-order full tensor."""
    sf = x.shape[axis] // (2 * cp_size)
    c0, c1 = _chunk_ids(cp_rank, cp_size)
    idx = list(range(c0 * sf, (c0 + 1) * sf)) + list(range(c1 * sf, (c1 + 1) * sf))
    return np.take(x, idx, axis=axis)


def _folded_full(x, cp_size, axis=1):
    """What the CP all-gather reconstructs: the local folded slices, rank-major."""
    return np.concatenate([_fold_rows(x, r, cp_size, axis) for r in range(cp_size)], axis=axis)


def _rank_positions(seq, cp_rank, cp_size):
    sf = seq // (2 * cp_size)
    c0, c1 = _chunk_ids(cp_rank, cp_size)
    return list(range(c0 * sf, (c0 + 1) * sf)) + list(range(c1 * sf, (c1 + 1) * sf))


def _causal_mask(q_len, k_len):
    """Right-down causal: query ``i`` of a block may see keys ``<= k_len - q_len + i``."""
    i = np.arange(q_len)[:, None]
    j = np.arange(k_len)[None, :]
    return j <= (k_len - q_len + i)


def toy_attention(query, key, value, topk, scale, *rest, query_rope=None, key_rope=None, **kwargs):
    """Dense stand-in for sparse_flash_attention: causal softmax(q k^T) v on the given key.

    Shapes follow the BSND kernel: q (B, S1, N, D), k/v (B, S2, 1, D); returns
    (out, softmax_max, softmax_sum) with the stats as (B, 1, S1, N).
    """
    del topk, rest, kwargs
    q = query if query_rope is None else ops.cat([query, query_rope], -1)
    k = key if key_rope is None else ops.cat([key, key_rope], -1)
    s1, s2 = q.shape[1], k.shape[1]
    q_h = ops.transpose(q, (0, 2, 1, 3))                    # (B, N, S1, D)
    k_t = ops.transpose(k, (0, 2, 3, 1))                    # (B, 1, D, S2)
    scores = ops.matmul(q_h, k_t) * scale                   # (B, N, S1, S2)
    mask = Tensor(_causal_mask(s1, s2))
    scores = ops.masked_fill(scores, ~mask, -1e9)
    smax = scores.max(axis=-1, keepdims=True)
    probs = ops.exp(scores - smax)
    ssum = probs.sum(axis=-1, keepdims=True)
    v_h = ops.transpose(value, (0, 2, 1, 3))                # (B, 1, S2, D)
    out = ops.transpose(ops.matmul(probs / ssum, v_h), (0, 2, 1, 3))
    stats = lambda t: ops.transpose(t[..., 0], (0, 2, 1)).expand_dims(1)  # (B, 1, S1, N)
    return out, stats(smax), stats(ssum)


def toy_kl_loss(query, key, q_idx, k_idx, weights, topk, smax, ssum, scale, q_rope, k_rope, *opts):
    """Stand-in for the fused sparse indexer KL loss: returns its own grads plus a loss.

    Per query ``i`` with visible keys ``V(i)``:  d_q_idx[i] = q_idx[i] * sum_{j in V(i)} k_idx[j],
    d_k_idx[j] = sum_{i: j in V(i)} q_idx[i],  d_w = weights,  loss = sum_i sum_{j in V(i)} k_idx[j].
    The key gradient therefore depends on which queries see which keys, which is what the
    fold must preserve.
    """
    del query, key, topk, smax, ssum, scale, q_rope, k_rope, opts
    s1, s2 = q_idx.shape[1], k_idx.shape[1]
    m = Tensor(_causal_mask(s1, s2).astype(np.float32))  # (S1, S2)
    k_tok = k_idx.sum(axis=(2, 3))                         # (B, S2)
    q_tok = q_idx.sum(axis=(2, 3))                         # (B, S1)
    seen = ops.matmul(k_tok, m.T)                          # (B, S1)
    d_q = q_idx * seen[:, :, None, None]
    d_k_tok = ops.matmul(q_tok, m)                         # (B, S2)
    d_k = ops.broadcast_to(d_k_tok[:, :, None, None], k_idx.shape)
    loss = seen.sum().reshape((1,))
    return d_q, d_k, weights * 1.0, loss


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4, 8])
def test_prefix_chunks_balanced(cp_size):
    """Both blocks' prefixes sum to 2N + 1 chunks on every rank; the fold order is an involution."""
    order = fold.build_fold_order(cp_size)
    assert [order[c] for c in order] == list(range(2 * cp_size))
    for r in range(cp_size):
        total = fold.balanced_prefix_chunks(r, cp_size, 0) + fold.balanced_prefix_chunks(r, cp_size, 1)
        assert total == 2 * cp_size + 1


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_unfold_prefix_is_natural_causal_prefix(cp_size):
    """Un-folding the gathered key yields tokens 0..klen-1 in natural order for each block."""
    seq = 2 * cp_size * 3
    tokens = np.arange(seq, dtype=np.float32).reshape(1, seq, 1, 1)
    folded = Tensor(_folded_full(tokens, cp_size))
    sf = seq // (2 * cp_size)
    for r in range(cp_size):
        c0, c1 = _chunk_ids(r, cp_size)
        for block, last_chunk in ((0, c0), (1, c1)):
            got = fold.unfold_prefix(folded, r, cp_size, block).asnumpy().reshape(-1)
            np.testing.assert_array_equal(got, np.arange((last_chunk + 1) * sf))


@pytest.mark.parametrize("cp_size", [2, 4])
def test_fold_prefix_grad_inverts_unfold(cp_size):
    """Scattering a prefix gradient lands each chunk on the folded slot it came from, zero elsewhere."""
    seq = 2 * cp_size * 2
    rng = np.random.default_rng(0)
    for r in range(cp_size):
        for block in (0, 1):
            klen = fold.balanced_prefix_chunks(r, cp_size, block) * (seq // (2 * cp_size))
            grad = rng.standard_normal((1, klen, 1, 3)).astype(np.float32)
            scattered = fold.fold_prefix_grad(Tensor(grad), r, cp_size, block, seq)
            back = fold.unfold_prefix(scattered, r, cp_size, block).asnumpy()
            np.testing.assert_array_equal(back, grad)
            covered = set(fold.build_prefix_order(r, cp_size, block))
            sf = seq // (2 * cp_size)
            arr = scattered.asnumpy()
            for slot in range(2 * cp_size):
                if slot not in covered:
                    assert not arr[:, slot * sf:(slot + 1) * sf].any()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_folded_attention_matches_full_sequence(cp_size):
    """Forward rows and the key gradient (summed over ranks) equal one full-sequence call."""
    seq = 2 * cp_size * 3
    rng = np.random.default_rng(1)
    q = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    k = rng.standard_normal((B, seq, 1, D)).astype(np.float32)
    qr = rng.standard_normal((B, seq, N_HEADS, 2)).astype(np.float32)
    kr = rng.standard_normal((B, seq, 1, 2)).astype(np.float32)
    w_out = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    topk = np.zeros((B, seq, 1, 1), np.int32)
    scale = 0.5

    def ref_loss(k_t, kr_t):
        out, smax, ssum = toy_attention(Tensor(q), k_t, k_t, Tensor(topk), scale,
                                        query_rope=Tensor(qr), key_rope=kr_t)
        return (out * Tensor(w_out)).sum() + smax.sum() + ssum.sum()

    ref_out, ref_smax, ref_ssum = toy_attention(Tensor(q), Tensor(k), Tensor(k), Tensor(topk), scale,
                                                query_rope=Tensor(qr), key_rope=Tensor(kr))
    ref_gk, ref_gkr = ms.grad(ref_loss, grad_position=(0, 1))(Tensor(k), Tensor(kr))

    k_fold, kr_fold = Tensor(_folded_full(k, cp_size)), Tensor(_folded_full(kr, cp_size))

    def rank_call(r, k_t, kr_t):
        return fold.fold_sparse_flash_attention(
            toy_attention, r, cp_size,
            Tensor(_fold_rows(q, r, cp_size)), k_t, k_t, Tensor(_fold_rows(topk, r, cp_size)), scale,
            query_rope=Tensor(_fold_rows(qr, r, cp_size)), key_rope=kr_t)

    def folded_loss(k_t, kr_t):
        total = 0
        for r in range(cp_size):
            out, smax, ssum = rank_call(r, k_t, kr_t)
            total = total + (out * Tensor(_fold_rows(w_out, r, cp_size))).sum() + smax.sum() + ssum.sum()
        return total

    for r in range(cp_size):
        out, smax, ssum = rank_call(r, k_fold, kr_fold)
        pos = _rank_positions(seq, r, cp_size)
        np.testing.assert_allclose(out.asnumpy(), ref_out.asnumpy()[:, pos], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(smax.asnumpy(), ref_smax.asnumpy()[:, :, pos], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ssum.asnumpy(), ref_ssum.asnumpy()[:, :, pos], rtol=1e-5, atol=1e-5)

    gk, gkr = ms.grad(folded_loss, grad_position=(0, 1))(k_fold, kr_fold)
    np.testing.assert_allclose(gk.asnumpy(), _folded_full(ref_gk.asnumpy(), cp_size), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(gkr.asnumpy(), _folded_full(ref_gkr.asnumpy(), cp_size), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_folded_indexer_matches_full_sequence(cp_size):
    """lightning_indexer-shaped outputs stitch back row by row."""
    seq = 2 * cp_size * 2
    rng = np.random.default_rng(2)
    qi = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    ki = rng.standard_normal((B, seq, 1, D)).astype(np.float32)
    w = rng.standard_normal((B, seq, N_HEADS)).astype(np.float32)

    def toy_indexer(q_t, k_t, w_t, **kwargs):
        del kwargs
        scores = ops.matmul(ops.transpose(q_t, (0, 2, 1, 3)), ops.transpose(k_t, (0, 2, 3, 1)))
        scores = (scores * ops.transpose(w_t, (0, 2, 1))[..., None]).sum(axis=1)  # (B, S1, S2)
        mask = Tensor(_causal_mask(q_t.shape[1], k_t.shape[1]))
        visible = ops.masked_fill(scores, ~mask, 0.0).sum(axis=-1)          # (B, S1)
        count = Tensor(_causal_mask(q_t.shape[1], k_t.shape[1]).sum(-1).astype(np.int32))
        return count.reshape((1, -1, 1, 1)), visible.reshape((1, -1, 1, 1))

    ref_cnt, ref_val = toy_indexer(Tensor(qi), Tensor(ki), Tensor(w), sparse_count=4)
    k_fold = Tensor(_folded_full(ki, cp_size))
    for r in range(cp_size):
        cnt, val = fold.fold_lightning_indexer(
            toy_indexer, r, cp_size, Tensor(_fold_rows(qi, r, cp_size)), k_fold,
            Tensor(_fold_rows(w, r, cp_size)), sparse_count=4)
        pos = _rank_positions(seq, r, cp_size)
        # visible-key counts are absolute causal positions + 1: the prefix is the natural one
        np.testing.assert_array_equal(cnt.asnumpy().reshape(-1), np.array(pos) + 1)
        np.testing.assert_allclose(val.asnumpy(), ref_val.asnumpy()[:, pos], rtol=1e-5, atol=1e-5)
        assert int(ref_cnt.asnumpy().reshape(-1)[pos[-1]]) == pos[-1] + 1


@pytest.mark.parametrize("cp_size", [2, 4])
def test_folded_kl_loss_matches_full_sequence(cp_size):
    """Query-side grads by row, key grad and loss summed over ranks, all equal the full call."""
    seq = 2 * cp_size * 3
    rng = np.random.default_rng(3)
    q = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    k = rng.standard_normal((B, seq, 1, D)).astype(np.float32)
    qi = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    ki = rng.standard_normal((B, seq, 1, D)).astype(np.float32)
    w = rng.standard_normal((B, seq, N_HEADS)).astype(np.float32)
    topk = np.zeros((B, seq, 1, 1), np.int32)
    stats = rng.standard_normal((B, 1, seq, N_HEADS)).astype(np.float32)
    qr = rng.standard_normal((B, seq, N_HEADS, 2)).astype(np.float32)
    kr = rng.standard_normal((B, seq, 1, 2)).astype(np.float32)
    opts = (None, None, "BSND", 3, 1, 1)

    ref = toy_kl_loss(Tensor(q), Tensor(k), Tensor(qi), Tensor(ki), Tensor(w), Tensor(topk),
                      Tensor(stats), Tensor(stats), 0.5, Tensor(qr), Tensor(kr), *opts)
    folded_k = [Tensor(_folded_full(x, cp_size)) for x in (k, ki, kr)]
    d_k_sum, loss_sum = 0, 0
    for r in range(cp_size):
        local = lambda x, axis=1: Tensor(_fold_rows(x, r, cp_size, axis))
        d_q, d_k, d_w, loss = fold.fold_sparse_indexer_kl_loss(
            toy_kl_loss, r, cp_size,
            local(q), folded_k[0], local(qi), folded_k[1], local(w), local(topk),
            local(stats, 2), local(stats, 2), 0.5, local(qr), folded_k[2], *opts)
        pos = _rank_positions(seq, r, cp_size)
        np.testing.assert_allclose(d_q.asnumpy(), ref[0].asnumpy()[:, pos], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(d_w.asnumpy(), ref[2].asnumpy()[:, pos], rtol=1e-5, atol=1e-5)
        assert d_k.shape == ki.shape
        d_k_sum = d_k_sum + d_k.asnumpy()
        loss_sum = loss_sum + loss.asnumpy()
    np.testing.assert_allclose(d_k_sum, _folded_full(ref[1].asnumpy(), cp_size), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(loss_sum, ref[3].asnumpy(), rtol=1e-5, atol=1e-4)


def test_fold_flag_roundtrip():
    """The process-wide switch is off by default and toggles cleanly."""
    fold.set_dsa_cp_fold(False)
    assert not fold.dsa_cp_fold_enabled()
    fold.set_dsa_cp_fold(True)
    assert fold.dsa_cp_fold_enabled()
    fold.set_dsa_cp_fold(False)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_unfold_prefix_backward_matches_index_add(cp_size):
    """Custom backward (gather with a zero chunk) equals the autograd ``index_select`` gradient."""
    seq = 2 * cp_size * 3
    sf = seq // (2 * cp_size)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, seq, 1, 5)).astype(np.float32)
    for r in range(cp_size):
        for block in (0, 1):
            prefix = fold.build_prefix_order(r, cp_size, block)
            w = rng.standard_normal((1, len(prefix) * sf, 1, 5)).astype(np.float32)

            def loss(t, w=w, r=r, block=block):
                return (fold.unfold_prefix(t, r, cp_size, block) * Tensor(w)).sum()

            got = ms.grad(loss)(Tensor(x)).asnumpy()
            want = np.zeros_like(x)
            for j, slot in enumerate(prefix):
                want[:, slot * sf:(slot + 1) * sf] += w[:, j * sf:(j + 1) * sf]
            np.testing.assert_allclose(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("used", ["both", "block0", "block1"])
def test_unfold_prefix_pair_matches_single(cp_size, used):
    """Pair forward equals the two single prefixes; merged backward equals their summed gradients."""
    seq = 2 * cp_size * 3
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, seq, 1, 5)).astype(np.float32)
    for r in range(cp_size):
        ws = [rng.standard_normal(tuple(fold.unfold_prefix(Tensor(x), r, cp_size, b).shape)).astype(np.float32)
              for b in (0, 1)]
        use = {"both": (0, 1), "block0": (0,), "block1": (1,)}[used]

        def pair_loss(t, r=r):
            p = fold.unfold_prefix_pair(t, r, cp_size)
            return sum((p[b] * Tensor(ws[b])).sum() for b in use)

        def single_loss(t, r=r):
            return sum((fold.unfold_prefix(t, r, cp_size, b) * Tensor(ws[b])).sum() for b in use)

        p0, p1 = fold.unfold_prefix_pair(Tensor(x), r, cp_size)
        np.testing.assert_array_equal(p0.asnumpy(), fold.unfold_prefix(Tensor(x), r, cp_size, 0).asnumpy())
        np.testing.assert_array_equal(p1.asnumpy(), fold.unfold_prefix(Tensor(x), r, cp_size, 1).asnumpy())
        np.testing.assert_allclose(ms.grad(pair_loss)(Tensor(x)).asnumpy(),
                                   ms.grad(single_loss)(Tensor(x)).asnumpy(), rtol=0, atol=1e-6)


def toy_dense_lse(q_idx, k_idx, weights, *opts):
    """Stand-in for the dense stage-1 softmax_lse forward: per-query stats over visible keys.

    Returns ``(softmax_max_index, softmax_sum_index)`` as ``(B, 1, S1)``, the layout the
    real kernel uses, so the fold has to stitch them on dim 2 rather than dim 1.
    """
    del opts
    s1, s2 = q_idx.shape[1], k_idx.shape[1]
    m = Tensor(_causal_mask(s1, s2).astype(np.float32))
    k_tok = k_idx.sum(axis=(2, 3))                          # (B, S2)
    seen = ops.matmul(k_tok, m.T)                           # (B, S1)
    w_tok = weights.sum(axis=2)                             # (B, S1)
    return (seen + w_tok).expand_dims(1), (seen * 2.0).expand_dims(1)


def toy_dense_kl_loss(query, key, q_idx, k_idx, weights, smax, ssum,
                      smax_idx, ssum_idx, scale, q_rope, k_rope, *opts):
    """Stand-in for the fused dense indexer KL loss (18-arg MindSpore form).

    Same contract as :func:`toy_kl_loss`, plus the two indexer statistics ``(B, 1, S1)``
    folded into ``d_q_idx`` so a wrongly split stat shows up as a row mismatch.
    """
    del query, key, smax, ssum, scale, q_rope, k_rope, opts
    s1, s2 = q_idx.shape[1], k_idx.shape[1]
    m = Tensor(_causal_mask(s1, s2).astype(np.float32))
    k_tok = k_idx.sum(axis=(2, 3))
    q_tok = q_idx.sum(axis=(2, 3))
    seen = ops.matmul(k_tok, m.T)                           # (B, S1)
    stat = smax_idx[:, 0, :] + ssum_idx[:, 0, :]            # (B, S1)
    d_q = q_idx * (seen + stat)[:, :, None, None]
    d_k_tok = ops.matmul(q_tok, m)
    d_k = ops.broadcast_to(d_k_tok[:, :, None, None], k_idx.shape)
    loss = (seen + stat).sum().reshape((1,))
    return d_q, d_k, weights * 1.0, loss


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4, 8])
def test_folded_dense_lse_matches_full_sequence(cp_size):
    """Stage-1 forward: both per-query statistics match the full call row by row."""
    seq = 2 * cp_size * 3
    rng = np.random.default_rng(11)
    qi = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    ki = rng.standard_normal((B, seq, 1, D)).astype(np.float32)
    w = rng.standard_normal((B, seq, N_HEADS)).astype(np.float32)
    opts = (None, None, "BSND", 3, 1, 1)

    ref = toy_dense_lse(Tensor(qi), Tensor(ki), Tensor(w), *opts)
    natural_ki = Tensor(ki)  # 一阶段只折 q 侧，key 是自然序
    for r in range(cp_size):
        local = lambda x, axis=1: Tensor(_fold_rows(x, r, cp_size, axis))
        out = fold.fold_dense_lightning_indexer_softmax_lse(
            toy_dense_lse, r, cp_size, local(qi), natural_ki, local(w), *opts)
        pos = _rank_positions(seq, r, cp_size)
        for got, want in zip(out, ref):
            np.testing.assert_allclose(got.asnumpy(), want.asnumpy()[:, :, pos], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cp_size", [1, 2, 3, 4, 8])
def test_folded_dense_kl_loss_matches_full_sequence(cp_size):
    """Stage-1 loss: query-side grads by row, key grad (natural order) and loss summed over ranks."""
    seq = 2 * cp_size * 3
    rng = np.random.default_rng(12)
    q = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    k = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    qi = rng.standard_normal((B, seq, N_HEADS, D)).astype(np.float32)
    ki = rng.standard_normal((B, seq, 1, D)).astype(np.float32)
    w = rng.standard_normal((B, seq, N_HEADS)).astype(np.float32)
    stats = rng.standard_normal((B, 1, seq, N_HEADS)).astype(np.float32)
    idx_stats = rng.standard_normal((B, 1, seq)).astype(np.float32)
    qr = rng.standard_normal((B, seq, N_HEADS, 2)).astype(np.float32)
    kr = rng.standard_normal((B, seq, N_HEADS, 2)).astype(np.float32)
    opts = (None, None, "BSND", 3, 1, 1)

    ref = toy_dense_kl_loss(Tensor(q), Tensor(k), Tensor(qi), Tensor(ki), Tensor(w),
                            Tensor(stats), Tensor(stats), Tensor(idx_stats), Tensor(idx_stats),
                            0.5, Tensor(qr), Tensor(kr), *opts)
    natural_k = [Tensor(x) for x in (k, ki, kr)]  # 一阶段只折 q 侧，key 是自然序
    d_k_sum, loss_sum = 0, 0
    for r in range(cp_size):
        local = lambda x, axis=1: Tensor(_fold_rows(x, r, cp_size, axis))
        d_q, d_k, d_w, loss = fold.fold_dense_indexer_kl_loss(
            toy_dense_kl_loss, r, cp_size,
            local(q), natural_k[0], local(qi), natural_k[1], local(w),
            local(stats, 2), local(stats, 2), local(idx_stats, 2), local(idx_stats, 2),
            0.5, local(qr), natural_k[2], *opts)
        pos = _rank_positions(seq, r, cp_size)
        np.testing.assert_allclose(d_q.asnumpy(), ref[0].asnumpy()[:, pos], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(d_w.asnumpy(), ref[2].asnumpy()[:, pos], rtol=1e-5, atol=1e-5)
        assert d_k.shape == ki.shape
        d_k_sum = d_k_sum + d_k.asnumpy()
        loss_sum = loss_sum + loss.asnumpy()
    np.testing.assert_allclose(d_k_sum, ref[1].asnumpy(), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(loss_sum, ref[3].asnumpy(), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("wrapper,n_args", [
    (fold.fold_dense_lightning_indexer_softmax_lse, 3),
    (fold.fold_dense_indexer_kl_loss, 13),
])
def test_dense_fold_rejects_n_mismatch(wrapper, n_args):
    """check_fold_shapes must fire when the slicer's N disagrees with the fold's."""
    seq, cp_size = 24, 4
    local_len = seq // cp_size
    q = Tensor(np.zeros((B, local_len, N_HEADS, D), np.float32))
    key = Tensor(np.zeros((B, seq, 1, D), np.float32))
    args = [q, key] + [q] * (n_args - 2)
    if n_args == 3:
        args = [q, key, Tensor(np.zeros((B, local_len, N_HEADS), np.float32))]
    with pytest.raises(ValueError, match="fold shape mismatch"):
        wrapper(lambda *a, **k: None, 0, cp_size // 2, *args)
