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
"""Validate direct-output dropless permutation against the native allocating op."""
import os

import pytest
import torch
import torch_npu

from hyper_parallel.core.multicore.torch import ops
from tests.common.mark_utils import arg_mark
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import write_evidence


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_moe_token_permute_out() -> None:
    """Feature: Token permutation into caller-owned buffers.

    Description: Change routes while reusing outputs and alternating streams.
    Expectation: Values and mappings match the native operator and invalid outputs are rejected.
    """
    ops._load_native()  # pylint: disable=protected-access
    records = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for index_dtype in (torch.int32, torch.int64):
            for tokens, top_k, hidden in ((1, 1, 32), (17, 2, 96), (128, 8, 512)):
                source = torch.empty(tokens, hidden, device=baseline.DEVICE, dtype=dtype)
                ids = torch.empty(tokens, top_k, device=baseline.DEVICE, dtype=index_dtype)
                output = torch.empty(tokens * top_k, hidden, device=baseline.DEVICE, dtype=dtype)
                mapping = torch.empty(tokens * top_k, device=baseline.DEVICE, dtype=torch.int32)
                pointers = output.data_ptr(), mapping.data_ptr()
                previous = torch.npu.current_stream()
                side = torch.npu.Stream()
                for repeat in range(4):
                    stream = side if repeat % 2 else torch.npu.current_stream()
                    stream.wait_stream(previous)
                    with torch.npu.stream(stream):
                        source.normal_()
                        ids.random_(0, 16 if repeat % 2 else 1)
                        expected = torch_npu.npu_moe_token_permute(source, ids)
                        output.fill_(float("nan"))
                        mapping.fill_(-1)
                        actual = torch.ops.hyper_parallel.moe_token_permute_out(source, ids, output, mapping)
                        source.fill_(float("nan"))
                    stream.synchronize()
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual[1], expected[1].flatten(), rtol=0, atol=0)
                    assert pointers == (actual[0].data_ptr(), actual[1].data_ptr())
                    previous = stream
                    records.append({"dtype": str(dtype), "indices": str(index_dtype),
                                    "tokens": tokens, "top_k": top_k, "repeat": repeat, "exact": True})
    empty = torch.empty(0, 32, device=baseline.DEVICE)
    empty_ids = torch.empty(0, 2, device=baseline.DEVICE, dtype=torch.int32)
    empty_mapping = torch.empty(0, device=baseline.DEVICE, dtype=torch.int32)
    torch.ops.hyper_parallel.moe_token_permute_out(empty, empty_ids, torch.empty_like(empty), empty_mapping)
    meta_tokens = torch.empty(17, 96, device="meta")
    meta_ids = torch.empty(17, 2, device="meta", dtype=torch.int32)
    meta_out = torch.empty(34, 96, device="meta")
    meta_mapping = torch.empty(34, device="meta", dtype=torch.int32)
    torch.ops.hyper_parallel.moe_token_permute_out(meta_tokens, meta_ids, meta_out, meta_mapping)
    with pytest.raises(RuntimeError, match="all top-k rows"):
        torch.ops.hyper_parallel.moe_token_permute_out(meta_tokens, meta_ids, meta_out[:1], meta_mapping)
    singleton_ids = torch.zeros(source.size(0), 1, device=source.device, dtype=torch.int32)
    singleton_mapping = torch.empty(source.size(0), device=source.device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        torch.ops.hyper_parallel.moe_token_permute_out(source, singleton_ids, source, singleton_mapping)
    torch.npu.synchronize()
    write_evidence({"case": "permute_out", "comparisons": records, "meta_passed": True,
                    "empty_passed": True, "overlap_rejected": True,
                    "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE")})
