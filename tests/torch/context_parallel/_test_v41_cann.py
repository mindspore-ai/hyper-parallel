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

import json
import os
from unittest.mock import patch

import pytest
import torch

from hyper_parallel.components.functional import compressed_cann
from hyper_parallel.components.modules.shared_compressed_dsa_attention import (
    SharedCompressedPackedSequence,
    _SharedCompressedIndexerKLLoss,
    compressed_causal_topk,
    shared_compressed_indexer_kl_loss,
)


def _device():
    """Require a real operator installation and allow selecting an idle card."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    torch.npu.set_device(int(os.environ.get("V41_TEST_DEVICE", "0")))
    if compressed_cann._operators() is None:
        pytest.skip("requires ops-transformer LI V2 and external-teacher KL")
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
            ops = compressed_cann._operators()
            with patch.object(ops, "lightning_indexer", wraps=ops.lightning_indexer) as native:
                actual = compressed_causal_topk(
                    q, key, weights, compress_ratio=ratio, sparse_count=4, query_offset=start,
                    minimum_key_indices=packed.minimum_key_indices(device, ratio),
                    query_segments=packed.indexer_segments(ratio),
                )
            expected = compressed_causal_topk(
                q.cpu(), key.cpu(), weights.cpu(), compress_ratio=ratio, sparse_count=4, query_offset=start,
                minimum_key_indices=packed.minimum_key_indices(device, ratio).cpu(),
            )
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            assert native.call_count > 0


def test_kl_fused_and_partial_rows_share_the_same_global_mean():
    """Full rows fuse, padding falls back, while the original scalar/gradient contract stays intact."""
    device = _device()
    for batch, padding, underflow in ((1, "none", False), (2, "mixed", False),
                                      (1, "all", False), (1, "none", True)):
        torch.manual_seed(1435)
        inputs = [(torch.randn(batch, 16, 8, 128)*.2).bfloat16(),
                  (torch.randn(batch, 64, 128)*.2).bfloat16(), torch.randn(batch, 16, 8)*.1]
        aq = (torch.randn(batch, 8, 16, 512)*.2).bfloat16().to(device)
        ak = (torch.randn(batch, 64, 512)*.2).bfloat16().to(device)
        sink = torch.linspace(-4, 8, 8, device=device)
        if underflow:
            sink.fill_(1000.)
        idx = torch.arange(16).view(1, 1, 16).expand(batch, 16, 16).clone().to(device)
        idx[:, :, 2] = idx[:, :, 0]
        if padding == "mixed":
            idx[:, ::3, ::2] = -1
            idx[:, 0] = -1
        elif padding == "all":
            idx[:, :, ::2] = -1
        reference = [value.to(device).detach().requires_grad_() for value in inputs]
        fused = [value.to(device).detach().requires_grad_() for value in inputs]
        expected = _SharedCompressedIndexerKLLoss.apply(*reference, aq, ak, idx, sink, 512**-.5, .7, 8, None)
        ops = compressed_cann._operators()
        with patch.object(ops, "sparse_lightning_indexer_kl_loss_grad",
                          wraps=ops.sparse_lightning_indexer_kl_loss_grad) as native:
            with torch.autocast("npu", dtype=torch.bfloat16):
                actual = shared_compressed_indexer_kl_loss(
                    *fused, aq, ak, idx, sink, attention_scale=512**-.5, loss_coeff=.7, query_chunk_size=8,
                )
        assert (native.call_count > 0) == (padding != "all")
        expected.backward(torch.tensor(.37, device=device))
        actual.backward(torch.tensor(.37, device=device))
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected)
        errors = {}
        for name, left, right in zip(("dq", "dk", "dw"), fused, reference):
            torch.testing.assert_close(left.grad, right.grad)
            error = (left.grad.float()-right.grad.float()).norm()
            errors[name] = float(error/right.grad.float().norm().clamp_min(1e-20))
        print(json.dumps({"batch": batch, "padding": padding, "underflow": underflow,
                          "native_calls": native.call_count,
                          "loss_abs": float((actual-expected).detach().abs()), "relative_l2": errors}), flush=True)
