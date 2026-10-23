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
"""Exercise native grouped forward/dgrad/wgrad with runtime expert loads."""

import os

import pytest
import torch
from torch import _check
from torch.fx.experimental.proxy_tensor import make_fx

from hyper_parallel.components.functional.grouped_matmul import grouped_matmul
from hyper_parallel.components.functional.npu_grouped_swiglu import _get_npu_grouped_matmul
from hyper_parallel.distributed.expert_parallel.collectives import ep_split_sizes


def _joint(inputs, weight, counts, implementation):
    """Represent an EP receive length without requiring a compiler or process group."""
    rows = sum(ep_split_sizes(counts))
    _check(rows <= inputs.shape[0])
    received = inputs.narrow(0, 0, rows)
    groups = counts.cumsum(0)
    if implementation == "swiglu":
        output = _get_npu_grouped_matmul().apply(received, weight, groups)
    else:
        output = grouped_matmul(received, weight, group_list=groups)
    gradients = torch.autograd.grad(output.float().square().sum(), (inputs, weight))
    return output.detach(), *gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_grouped_runtime_loads(dtype):
    """Reuse one GMM joint graph as experts become empty, including zero total receive."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    inputs = torch.randn(32, 64, device="npu", dtype=dtype, requires_grad=True)
    weight = torch.randn(4, 64, 64, device="npu", dtype=dtype, requires_grad=True)
    for implementation in ("grouped", "swiglu"):
        counts = torch.tensor([8, 8, 8, 8], device="npu", dtype=torch.int64)
        graph = make_fx(_joint, tracing_mode="symbolic")(inputs, weight, counts, implementation)
        nodes = [str(node.target) for node in graph.graph.nodes]
        gmm_count = sum("npu_grouped_matmul" in target for target in nodes)
        assert gmm_count == 3, f"Expected forward/dgrad/wgrad GMM nodes=3, got {gmm_count}: {nodes}"
        for loads in ([0, 0, 0, 0], [0, 0, 32, 0], [3, 0, 11, 7]):
            counts = counts.new_tensor(loads)
            expected = _joint(inputs, weight, counts, implementation)
            actual = graph(inputs, weight, counts, implementation)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
