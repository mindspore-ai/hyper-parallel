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
"""Exercise the cached native out bridge against allocating unpermute backward."""

import os

import pytest
import torch
import torch_npu

from hyper_parallel.core.multicore.torch import ops
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import write_evidence


def test_moe_token_unpermute_grad_out() -> None:
    """Check changing inputs and reused outputs on default and non-default streams."""
    records = []
    streams = (torch.npu.current_stream(), torch.npu.Stream())
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens, top_k, hidden in ((1, 1, 32), (17, 2, 96), (128, 8, 512)):
            for strided in (False, True):
                for stream_index, stream in enumerate(streams):
                    with torch.npu.stream(stream):
                        source = torch.randn(tokens, hidden, device=baseline.DEVICE, dtype=dtype)
                        ids = torch.randint(0, 16, (tokens, top_k), device=baseline.DEVICE, dtype=torch.int32)
                        permuted, mapping = torch_npu.npu_moe_token_permute(source, ids)
                        probs = torch.rand(tokens, top_k, device=baseline.DEVICE, dtype=torch.float32)
                        grad_permuted = torch.empty_like(permuted)
                        grad_probs = torch.empty_like(probs)
                        pointers = (grad_permuted.data_ptr(), grad_probs.data_ptr())
                    stream.synchronize()
                    for repeat in range(3):
                        with torch.npu.stream(stream):
                            permuted.normal_()
                            probs.uniform_()
                            gradient = torch.randn_like(source)
                            if strided:
                                gradient = gradient.T.contiguous().T
                            expected = torch_npu.npu_moe_token_unpermute_grad(
                                permuted, gradient, mapping, probs,
                            )
                            grad_permuted.fill_(float("nan"))
                            grad_probs.fill_(float("nan"))
                            ops.mega_moe_unpermute_grad_out(
                                permuted, gradient, mapping, probs, grad_permuted, grad_probs,
                            )
                            gradient.fill_(float("nan"))
                        stream.synchronize()
                        torch.testing.assert_close(grad_permuted, expected[0], rtol=0, atol=0)
                        torch.testing.assert_close(grad_probs, expected[1], rtol=0, atol=0)
                        assert pointers == (grad_permuted.data_ptr(), grad_probs.data_ptr())
                        records.append({"dtype": str(dtype), "tokens": tokens, "top_k": top_k,
                                        "hidden": hidden, "strided": strided, "stream": stream_index,
                                        "repeat": repeat, "exact": True})
    meta_tokens = torch.empty(34, 96, device="meta")
    meta_grad = torch.empty(17, 96, device="meta")
    meta_indices = torch.empty(34, dtype=torch.int32, device="meta")
    meta_probs = torch.empty(17, 2, device="meta")
    ops.mega_moe_unpermute_grad_out(
        meta_tokens, meta_grad, meta_indices, meta_probs,
        torch.empty_like(meta_tokens), torch.empty_like(meta_probs),
    )
    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        ops.mega_moe_unpermute_grad_out(permuted, gradient, mapping, probs, permuted, grad_probs)
    write_evidence({"case": "cached_unpermute_gradient_out", "comparisons": records,
                    "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE"),
                    "meta_passed": True, "overlap_rejected": True})
