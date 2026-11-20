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
"""NPU worker checks for full symmetric output from a triangular Triton GEMM."""

from typing import Optional

import torch
import torch_npu

from hyper_parallel.components.functional import symmetric_matmul


def _bits(value: torch.Tensor) -> torch.Tensor:
    """Copy the output before reinterpreting its original bits on CPU."""
    integer_dtype = torch.int32 if value.dtype == torch.float32 else torch.int16
    return value.cpu().view(integer_dtype)


def _check_product(
    a: torch.Tensor, b: torch.Tensor, out_dtype: Optional[torch.dtype] = None, *, bitwise: bool = False,
) -> None:
    """Compare the lower product and require an exact, guarded symmetric result."""
    dtype = a.dtype if out_dtype is None else out_dtype
    shape = (*a.shape[:-2], a.shape[-2], a.shape[-2])
    elements = a.shape[-2] ** 2 * (1 if a.ndim == 2 else a.shape[0])
    storage = torch.full((elements + 2,), 123, dtype=dtype, device=a.device)
    out = storage[1:-1].view(shape)
    result = symmetric_matmul(a, b, out=out, out_dtype=out_dtype)
    if result is not out:
        raise AssertionError(f"Output buffer identity differs: expected={id(out)}, actual={id(result)}")
    reference = a.float() @ b.float() if dtype == torch.float32 else a @ b
    actual_bits = _bits(result)
    mask = torch.ones(shape[-2:], dtype=torch.bool).tril()
    torch.testing.assert_close(actual_bits, actual_bits.mT, rtol=0, atol=0)
    torch.testing.assert_close(storage[[0, -1]].cpu(), torch.full((2,), 123, dtype=dtype), rtol=0, atol=0)
    if bitwise:
        torch.testing.assert_close(actual_bits[..., mask], _bits(reference)[..., mask], rtol=0, atol=0)
    else:
        rtol = 1e-5 if dtype == torch.float32 else (1e-2 if dtype == torch.bfloat16 else 2e-3)
        torch.testing.assert_close(result.cpu().float()[..., mask], reference.cpu().float()[..., mask],
                                   rtol=rtol, atol=1e-6 if dtype == torch.float32 else 1e-5)


@torch.no_grad()
def test_symmetric_matmul_accuracy() -> None:
    """Cover all static tilings, odd tails, batches and retained FP32 accumulators."""
    torch_npu.npu.set_device(0)
    torch.manual_seed(42)
    for dtype in (torch.float16, torch.bfloat16):
        for shape in ((1, 17), (65, 33), (127, 129), (257, 513), (2, 768, 2048)):
            value = torch.randn(shape, dtype=dtype, device="npu") / shape[-1]**0.5
            _check_product(value, value.mT)
        value = torch.randn(2, 257, 513, dtype=dtype, device="npu") / 513**0.5
        _check_product(value, value.mT, torch.float32)
        empty = torch.empty(2, 0, 7, dtype=dtype, device="npu")
        if symmetric_matmul(empty, empty.mT).numel() != 0:
            raise AssertionError(f"Empty result has elements: expected={0}, shape={empty.shape}")
        value = torch.empty(2, 17, 0, dtype=dtype, device="npu")
        torch.testing.assert_close(symmetric_matmul(value, value.mT),
                                   torch.zeros(2, 17, 17, dtype=dtype, device="npu"), rtol=0, atol=0)
    for shape in ((1024, 1024), (5120, 5120)):
        value = torch.randn(shape, dtype=torch.bfloat16, device="npu") / shape[-1]**0.5
        _check_product(value, value.mT, bitwise=True)


@torch.no_grad()
def test_symmetric_matmul_layouts() -> None:
    """Accept dense/transposed operands without copies and preserve output guards."""
    torch_npu.npu.set_device(0)
    torch.manual_seed(1729)
    for dtype in (torch.float16, torch.bfloat16):
        transposed = (torch.randn(2, 129, 65, dtype=dtype, device="npu") / 129**0.5).mT
        _check_product(transposed, transposed.mT)
        square = torch.randn(129, 129, dtype=dtype, device="npu") / 129**0.5
        square = ((square + square.mT) * 0.5).contiguous()
        _check_product(square, square * 2)
        backing = torch.randn(65 * 129 + 2, dtype=dtype, device="npu")
        unaligned = backing[1:-1].view(65, 129)
        _check_product(unaligned, unaligned.mT)


@torch.no_grad()
def test_symmetric_matmul_streams_and_graph() -> None:
    """Retain cross-stream inputs and use the caller's graph capture stream."""
    torch_npu.npu.set_device(0)
    stream = torch_npu.npu.Stream()
    value = torch.full((2, 257, 513), 0.125, dtype=torch.bfloat16, device="npu")
    stream.wait_stream(torch_npu.npu.current_stream())
    with torch_npu.npu.stream(stream):
        output = symmetric_matmul(value, value.mT)
    del value
    torch.empty(2, 257, 513, dtype=torch.bfloat16, device="npu").fill_(float("nan"))
    torch_npu.npu.current_stream().wait_stream(stream)
    expected = torch.tensor(513 * 0.125**2, dtype=torch.bfloat16)
    actual = output.cpu()
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=0)
    value = torch.randn(2, 257, 513, dtype=torch.bfloat16, device="npu") / 513**0.5
    symmetric_matmul(value, value.mT)
    torch_npu.npu.synchronize()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        captured = symmetric_matmul(value, value.mT)
    graph.replay()
    torch_npu.npu.synchronize()
    captured_bits = _bits(captured)
    torch.testing.assert_close(captured_bits, captured_bits.mT, rtol=0, atol=0)
    reference = value @ value.mT
    torch.testing.assert_close(captured.cpu().float(), reference.cpu().float(), rtol=1e-2, atol=1e-5)
