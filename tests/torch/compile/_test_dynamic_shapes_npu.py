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
"""NPU regression for symbolic causal-label padding and cross entropy."""

import os
from unittest.mock import patch

import pytest
import torch

from hyper_parallel.compile import GraphCompiler, PassConfig


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dynamic_causal_loss_npu(dtype: torch.dtype) -> None:
    """Compare loss and gradients across symbolic sequence lengths.

    Args:
        dtype: Floating-point parameter dtype for the NPU model.
    """
    # The NPU extension is optional on CPU/GPU test hosts.
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU hardware is required")
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32)).to(device=device, dtype=dtype)

    def causal_loss(module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute shifted-label loss for a variable sequence length.

        Args:
            module: Language model producing vocabulary logits.
            x: Input token ids.
            y: Target token ids.

        Returns:
            Scalar causal cross entropy.
        """
        shifted = torch.nn.functional.pad(y, (0, 1), value=-100)[..., 1:].contiguous()
        return torch.nn.functional.cross_entropy(module(x).float().flatten(0, 1), shifted.flatten())

    compiler = GraphCompiler(model, causal_loss, pass_config=PassConfig(fsdp_enabled=False), dynamic=True)
    with patch.object(compiler, "compile", wraps=compiler.compile) as compile_spy:
        for length in (5, 7, 7, 11):
            x, y = (torch.randint(0, 32, (1, length), device=device) for _ in range(2))
            y[0, -2] = -100
            model.zero_grad()
            actual = compiler.forward_backward(x=x, y=y)
            expected = causal_loss(model, x, y)
            torch.testing.assert_close(actual, expected)
            gradients = torch.autograd.grad(expected, tuple(model.parameters()))
            for parameter, gradient in zip(model.parameters(), gradients):
                torch.testing.assert_close(parameter.grad, gradient)
        assert compile_spy.call_count == 1, f"Expected one graph capture, got {compile_spy.call_count}"


def run_dynamic_causal_loss_npu() -> None:
    """Run both dtypes without a distributed process group or rendezvous port."""
    for dtype in (torch.float32, torch.bfloat16):
        test_dynamic_causal_loss_npu(dtype)
