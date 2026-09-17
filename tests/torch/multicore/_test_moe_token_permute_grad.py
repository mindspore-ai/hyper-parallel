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
"""Compare the metadata-only ACLNN bridge with the existing permutation backward."""

import torch
import torch_npu

from hyper_parallel.core.multicore.torch import ops
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import write_evidence


def test_moe_token_permute_grad() -> None:
    """Cover dtypes, odd sizes, noncontiguous gradients and input storage reuse."""
    records = []
    stream = torch.npu.Stream()
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for tokens, top_k, hidden in ((1, 1, 32), (17, 2, 96), (128, 8, 512)):
            for strided in (False, True):
                with torch.npu.stream(stream):
                    torch.manual_seed(1709 + baseline.RANK)
                    source = torch.randn(tokens, hidden, device=baseline.DEVICE, dtype=dtype)
                    ids = torch.randint(0, 16, (tokens, top_k), device=baseline.DEVICE, dtype=torch.int32)
                    permuted, mapping = torch_npu.npu_moe_token_permute(source, ids)
                    gradient = torch.randn_like(permuted)
                    if strided:
                        gradient = gradient.T.contiguous().T
                    expected = torch_npu.npu_moe_token_permute_grad(source, gradient, ids, mapping)
                    actual = ops.moe_token_permute_grad(gradient, mapping, tokens, top_k)
                    assert actual.untyped_storage().data_ptr() != gradient.untyped_storage().data_ptr()
                    gradient.fill_(float("nan"))
                stream.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                records.append({"dtype": str(dtype), "tokens": tokens, "top_k": top_k,
                                "hidden": hidden, "strided": strided, "exact": True})
    empty = ops.moe_token_permute_grad(
        torch.empty((0, 32), dtype=torch.bfloat16, device=baseline.DEVICE),
        torch.empty(0, dtype=torch.int32, device=baseline.DEVICE), 0, 2,
    )
    assert empty.shape == (0, 32)
    meta = ops.moe_token_permute_grad(
        torch.empty((34, 96), device="meta"), torch.empty(34, dtype=torch.int32, device="meta"), 17, 2,
    )
    assert meta.shape == (17, 96) and meta.device.type == "meta"
    write_evidence({"case": "metadata_permutation_gradient", "comparisons": records,
                    "empty_passed": True, "meta_passed": True,
                    "grad_v2_available": hasattr(torch_npu, "npu_moe_token_permute_grad_v2")})
