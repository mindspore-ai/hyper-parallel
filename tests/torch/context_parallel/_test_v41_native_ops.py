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
"""Real LI V2 and external-teacher KL checks, including CP-local packed geometry."""

import os
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_parallel.components.functional import compressed_indexer_ops, compressed_smla
from hyper_parallel.components.functional.compressed_indexer_ops import (
    _load_indexer_op, _load_kl_ops, _kl_metadata, _stable_prediction_log, fused_compressed_kl_loss,
)
from hyper_parallel.core.activation_memory import checkpoint
from hyper_parallel.distributed.activation_checkpoint import make_selective_checkpoint_context_fn
from hyper_parallel.components.functional.compressed_smla import _attention_metadata, _load_attention_ops
from hyper_parallel.components.functional.compressed_smla import fused_sparse_mla_attention
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
    compressed_causal_topk,
    shared_compressed_indexer_kl_loss,
    reference_sparse_attention,
    build_sliding_window_indices,
)


def _device():
    """Require a real operator installation and allow selecting an idle card."""
    pytest.importorskip("torch_npu")
    # Native runtimes may cache this setting at their first operator call.
    # The launcher isolates this worker from earlier nondeterministic tests.
    torch.use_deterministic_algorithms(True)
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    required_devices = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if torch.npu.device_count() < required_devices:
        pytest.skip(f"requires {required_devices} visible Ascend NPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("V41_TEST_DEVICE", "0")))
    torch.npu.set_device(local_rank)
    _load_indexer_op()
    _load_attention_ops()
    _load_kl_ops()
    return torch.device("npu", torch.npu.current_device())


def test_li_ratio_packed_and_nonlast_cp_rank():
    """Monotonic scores give an unambiguous oracle for masks and global offsets."""
    device = _device()
    for ratio in (1, 2):
        for start, length in ((0, 7), (7, 16), (8, 16)):
            packed = SharedCompressedPackedSequence(torch.tensor([0, 12, 32, 64]), start, length, 64)
            packed = packed.prepare(device, (ratio,))
            q = torch.ones(1, length, 8, 128, device=device, dtype=torch.bfloat16) * .1
            key = torch.arange(1, 64//ratio+1).view(1, -1, 1).expand(1, -1, 128)
            key = (key.float() * .01).to(device=device, dtype=torch.bfloat16).contiguous()
            weights = torch.full((1, length, 8), .1, device=device)
            native = Mock(wraps=_load_indexer_op())
            with patch.object(compressed_indexer_ops, "_load_indexer_op", return_value=native):
                actual = compressed_causal_topk(
                    q, key, weights, compress_ratio=ratio, sparse_count=4, query_offset=start,
                    minimum_key_indices=packed.minimum_key_indices(device, ratio),
                    query_segments=packed.indexer_segments(ratio), use_fused=True,
                )
            expected = compressed_causal_topk(
                q.cpu(), key.cpu(), weights.cpu(), compress_ratio=ratio, sparse_count=4, query_offset=start,
                minimum_key_indices=packed.minimum_key_indices(device, ratio).cpu(),
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            assert native.call_count > 0


    for ratio in (1, 2):
        offset, length, key_length = 1200, 8, 2048
        q = torch.zeros(1, length, 8, 128, dtype=torch.bfloat16, device=device)
        q[..., :2] = 1
        key = torch.zeros(1, key_length, 128, dtype=torch.bfloat16, device=device)
        positions = torch.arange(key_length, device=device)
        key[0, :, 0] = (positions//128*128).to(key.dtype)
        key[0, :, 1] = (positions%128).to(key.dtype)
        weights = torch.ones(1, length, 8, device=device)
        actual = compressed_causal_topk(q, key, weights, compress_ratio=ratio,
                                       sparse_count=512, query_offset=offset, use_fused=True)
        counts = (torch.arange(length)+offset+1)//ratio
        expected = counts[:, None]-512+torch.arange(512)[None, :]
        torch.testing.assert_close(actual.cpu().long(), expected.unsqueeze(0), rtol=0, atol=0)


def _selections(ratio, offset, length, boundaries):
    """Build deterministic valid selections independently of LI kernels."""
    width = min(512, boundaries[-1] // ratio)
    indices = torch.full((1, length, width), -1, dtype=torch.int32)
    segments = []
    for left, right in zip(boundaries, boundaries[1:]):
        first, last = max(offset, left), min(offset+length, right)
        if first >= last:
            continue
        segments.append((first-offset, last-offset, left//ratio, last//ratio, (last-left)%ratio))
        for position in range(first, last):
            visible = (position+1-left)//ratio
            selected = torch.randperm(visible)[:width].sort().values + left//ratio
            indices[0, position-offset, :selected.numel()] = selected
    return indices, tuple(segments)


def _check_native_precision(actual, expected):
    """Apply CANN's SMLA dtype policy, separately from model-training acceptance.

    Source: ops-transformer/attention/sparse_flash_mla/tests/pytest/result_compare_method.py,
    check_result: BF16 rtol=2**-7, atol=1e-4; FP32 rtol=.005, atol=2.5e-5;
    at least 99.5% close. This is not a full-model convergence criterion.
    """
    assert bool(actual.isfinite().all()) and bool(expected.isfinite().all())
    relative, absolute = (2**-7, 1e-4) if actual.dtype == torch.bfloat16 else (.005, 2.5e-5)
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    difference = (actual-expected).abs()
    close = difference <= (absolute + relative * expected.abs())
    normalized = difference / torch.maximum(torch.maximum(actual.abs(), expected.abs()),
                                            actual.new_tensor(2**-14 / .005))
    fraction = float(close.float().mean())
    assert fraction >= .995 and float(normalized.max()) < 10, (fraction, float(normalized.max()))


def test_causal_kl_metadata_and_gradients():
    """Partial, full, packed, odd offsets and empty rows share the same objective."""
    device = _device()
    cases = [(1, 0, 64, [0, 256]), (2, 0, 66, [0, 256]), (2, 95, 32, [0, 256]),
             (2, 96, 33, [0, 256]), (1, 96, 160, [0, 126, 256]),
             (2, 96, 160, [0, 126, 256]), (1, 496, 48, [0, 2048]),
             (2, 1001, 64, [0, 4096]), (2, 0, 1, [0, 256])]
    for ratio, offset, length, boundaries in cases:
        torch.manual_seed(975)
        bank_length = boundaries[-1]//ratio
        indices, segments = _selections(ratio, offset, length, boundaries)
        indices = indices.to(device)
        source = [(torch.randn(1, length, 8, 128)*.2).bfloat16(),
                  (torch.randn(1, bank_length, 128)*.2).bfloat16(), torch.randn(1, length, 8)*.1]
        reference = [value.to(device).detach().requires_grad_() for value in source]
        candidate = [value.to(device).detach().requires_grad_() for value in source]
        aq = (torch.randn(1, 8, length, 512, device=device)*.3).bfloat16().requires_grad_()
        ak = (torch.randn(1, bank_length, 512, device=device)*.3).bfloat16().requires_grad_()
        sink = torch.linspace(-3, 7, 8, device=device).requires_grad_()
        expected = shared_compressed_indexer_kl_loss(
            *reference, aq, ak, indices, sink, attention_scale=512**-.5, loss_coeff=.7, query_chunk_size=32,
        )
        kl_grad, metadata = _load_kl_ops()
        native = Mock(wraps=kl_grad)
        with patch.object(compressed_indexer_ops, "_load_kl_ops", return_value=(native, metadata)):
            actual = shared_compressed_indexer_kl_loss(
                *candidate, aq, ak, indices, sink, attention_scale=512**-.5, loss_coeff=.7, query_chunk_size=32,
                compress_ratio=ratio, query_segments=segments, use_fused=True,
            )
        assert native.call_count == sum(end > start for _, _, start, end, _ in segments)
        seed = torch.tensor(.37, device=device)
        actual.backward(seed)
        expected.backward(seed)
        torch.testing.assert_close(actual, expected)
        for left, right in zip(candidate, reference):
            torch.testing.assert_close(left.grad, right.grad)
        assert aq.grad is None and ak.grad is None and sink.grad is None


def _rounded_attention_reference(query, raw, main, window, selected, sinks):
    """Independent CPU online softmax with the native BF16 multiply boundary.

    Each bank uses blocks of at most 512 keys. Exponentials are rounded before
    the value multiply, but the running normalizer remains FP32. This is the
    precision contract also modeled by ops-transformer's SMLA golden RUN_MODE=1.
    The separate all-FP32 formula remains the gradient oracle.
    """
    dtype = query.dtype
    query, raw, main, sinks = [value.detach().cpu().float() for value in (query, raw, main, sinks)]
    window = window.cpu().long()
    selected = None if selected is None else selected.cpu().long()
    output = torch.empty(query.shape[0], query.shape[2], query.shape[1], query.shape[3])
    for batch in range(query.shape[0]):
        for row in range(query.shape[2]):
            peak = sinks.clone()
            normalizer = torch.ones_like(peak)
            numerator = torch.zeros_like(query[batch, :, row])
            raw_ids = window[batch, row]
            banks = [raw[batch, 0, raw_ids[raw_ids >= 0]]]
            if selected is not None:
                ids = selected[batch, row]
                banks.append(main[batch, ids[ids >= 0]])
            for bank in banks:
                for first in range(0, bank.shape[0], 512):
                    keys = bank[first:first+512]
                    logits = (query[batch, :, row] @ keys.T) * 512**-.5
                    next_peak = torch.maximum(peak, logits.max(-1).values)
                    correction = (peak-next_peak).exp()
                    exponentials = (logits-next_peak[:, None]).exp()
                    numerator = numerator*correction[:, None] + exponentials.to(dtype).float() @ keys
                    normalizer = normalizer*correction + exponentials.sum(-1)
                    peak = next_peak
            output[batch, row] = numerator/normalizer[:, None]
    return output.to(dtype)


def test_native_attention_matches_independent_fp32_formula():
    """Check native output rounding and gradients against independent formulas."""
    device = _device()
    cases = [(0, 0, 32, [0, 64]), (1, 0, 32, [0, 64]), (2, 0, 33, [0, 64]),
             (2, 95, 32, [0, 256]), (1, 96, 80, [0, 126, 256]),
             (2, 96, 80, [0, 126, 256]), (1, 1024, 16, [0, 2048]),
             (2, 2049, 16, [0, 4096]), (2, 0, 1, [0, 64])]
    for ratio, offset, length, boundaries in cases:
        torch.manual_seed(990)
        indices, segments = _selections(max(1, ratio), offset, length, boundaries)
        indices = indices.to(device)
        total_length = boundaries[-1]
        tensors = [(torch.randn(1, 8, length, 512, device=device)*.3).bfloat16(),
                   (torch.randn(1, 1, total_length, 512, device=device)*.3).bfloat16(),
                   (torch.randn(1, total_length//max(1, ratio), 512, device=device)*.3).bfloat16(),
                   torch.linspace(-3, 7, 8, device=device)]
        reference = [value.detach().clone().requires_grad_() for value in tensors]
        candidate = [value.detach().clone().requires_grad_() for value in tensors]
        query, raw, main, sink = reference
        window = build_sliding_window_indices(1, length, 128, device, query_offset=offset, key_length=total_length)
        packed = SharedCompressedPackedSequence(torch.tensor(boundaries), offset, length, total_length)
        window.masked_fill_(window < packed.local_segment_starts(device).unsqueeze(-1), -1)
        bank = torch.cat((raw, main.unsqueeze(1)), 2) if ratio else raw
        sparse = (torch.cat((window, torch.where(indices >= 0, indices.long()+total_length, -1)), -1)
                  if ratio else window)
        expected = reference_sparse_attention(query.float(), bank.float(), sparse, sink, 512**-.5)
        rounded = _rounded_attention_reference(query, raw, main, window, indices if ratio else None, sink)
        query, raw, main, sink = candidate
        _attention_metadata.cache_clear()
        with torch.inference_mode():
            validation_output = fused_sparse_mla_attention(
                query, raw, main if ratio else None, indices if ratio else None,
                sink, 512**-.5, ratio, 128, offset, segments,
            )
        forward, backward, metadata = _load_attention_ops()
        native = Mock(wraps=forward)
        with patch.object(compressed_smla, "_load_attention_ops", return_value=(native, backward, metadata)):
            actual = fused_sparse_mla_attention(query, raw, main if ratio else None, indices if ratio else None,
                                               sink, 512**-.5, ratio, 128, offset, segments)
        torch.testing.assert_close(actual.detach(), validation_output, rtol=0, atol=0)
        grad = torch.randn_like(actual)*.01
        actual.backward(grad)
        expected.backward(grad.float())
        assert native.call_count > 0
        _check_native_precision(actual, rounded.to(device))
        for left, right in zip(candidate, reference):
            if right.grad is None:
                assert left.grad is None
                continue
            # A zero-width compressed prefix has no mathematical contribution.
            left_grad = left.grad if left.grad is not None else torch.zeros_like(left)
            _check_native_precision(left_grad, right.grad)


def _recompute_case(device, ratio, offset, mode):
    """Replay the complete fused region with an intentionally underflowed student."""
    torch.manual_seed(992)
    length = 64
    total = max(offset + length, length)

    def _tensor(shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device=device) * .1).to(dtype).requires_grad_()

    values = [_tensor((1, length, 8, 128)), _tensor((1, total//ratio, 128)),
              _tensor((1, length, 8), torch.float32), _tensor((1, 8, length, 512)),
              _tensor((1, 1, total, 512)), _tensor((1, total//ratio, 512)), _tensor((8,), torch.float32)]
    with torch.no_grad():
        values[2].mul_(10000)
    segments = ((0, length, 0, (offset+length)//ratio, (offset+length)%ratio),)
    entries = []
    underflow_rows = []
    original_log = _stable_prediction_log

    def _record_log(prediction, query, key, weights, indices, chunk_size):
        underflow_rows.append(int(((prediction <= 0) & (indices >= 0)).any(-1).sum()))
        return original_log(prediction, query, key, weights, indices, chunk_size)

    def _region(iq, ik, weight, query, raw, main, sinks):
        entries.append(mode)
        indices = compressed_causal_topk(
            iq, ik, weight, compress_ratio=ratio, sparse_count=512, query_offset=offset,
            query_segments=segments, use_fused=True,
        )
        kl = fused_compressed_kl_loss(
            iq, ik, weight, query, main, indices, sinks, 512**-.5, .001, 256, ratio, segments,
        )
        output = fused_sparse_mla_attention(
            query, raw, main, indices, sinks, 512**-.5, ratio, 128, offset, segments,
        )
        return output.float().square().mean() + kl

    _kl_metadata.cache_clear()
    _attention_metadata.cache_clear()
    with patch.object(compressed_indexer_ops, "_stable_prediction_log", new=_record_log):
        if mode == "off":
            loss = _region(*values)
        else:
            options = {} if mode == "full" else {"context_fn": make_selective_checkpoint_context_fn()}
            loss = checkpoint(_region, *values, use_reentrant=False, **options)
        loss.backward()
    assert max(underflow_rows) > 0, "the regression must exercise valid student-probability underflow"
    assert len(entries) == (1 if mode == "off" else 2), "checkpoint must actually replay the region"
    gradients = [value.grad.detach().cpu() for value in values]
    assert bool(loss.isfinite()) and all(bool(grad.isfinite().all()) for grad in gradients)
    return loss.detach().cpu(), gradients


def test_native_fused_recompute():
    """Full and strict SAC replay preserve all seven input gradients, including underflow."""
    device = _device()
    for ratio in (1, 2):
        for offset in (0, 1985):
            reference_loss, reference_gradients = _recompute_case(device, ratio, offset, "off")
            for mode in ("full", "selective"):
                loss, gradients = _recompute_case(device, ratio, offset, mode)
                torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
                for actual, expected in zip(gradients, reference_gradients):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_fused_operators():
    """Run native formulas in one isolated worker with determinism enabled from the first call."""
    test_li_ratio_packed_and_nonlast_cp_rank()
    test_causal_kl_metadata_and_gradients()
    test_native_attention_matches_independent_fp32_formula()
