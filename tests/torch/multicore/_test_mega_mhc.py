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
"""NPU worker for cached-plan isolation across concurrent mHC invocations."""

import os

import torch
import torch_npu  # noqa: F401  # pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.mega_mhc.function import _hyper_mega_mhc_forward
from hyper_parallel.core.multicore.modules.mega_mhc_grad.common import MegaMhcGradCache
from hyper_parallel.core.multicore.modules.mega_mhc_grad.function import hyper_mega_mhc_grad
from tests.common.multicore.mega_mhc_golden import torch_mega_mhc
from tests.common.multicore.mega_mhc_grad_golden import torch_mega_mhc_grad


def _make_case(seed: int) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Make two reproducible, distinct cases that share an LRU plan key."""
    torch.manual_seed(seed)
    device = torch.device("npu", torch.npu.current_device())
    tokens, hidden = 256, 512
    inputs = (
        torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16),
        torch.randn(tokens, 4, hidden, device=device, dtype=torch.bfloat16),
        torch.sigmoid(torch.randn(tokens, 4, device=device)),
        2 * torch.sigmoid(torch.randn(tokens, 4, device=device)),
        torch.softmax(torch.randn(tokens, 4, 4, device=device), dim=-1),
        torch.randn(24, 4 * hidden, device=device) / (4 * hidden) ** 0.5,
        torch.ones(3, device=device),
        torch.zeros(24, device=device),
        torch.ones(hidden, device=device, dtype=torch.bfloat16),
    )
    gradients = tuple(torch.randn_like(inputs[index]) * 0.1 for index in (1, 2, 3, 4, 0))
    return inputs, gradients


def _run_case(inputs: tuple[torch.Tensor, ...], gradients: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Run paired forward/backward calls without a host or device synchronize."""
    outputs, cache = _hyper_mega_mhc_forward(*inputs, need_backward=True)
    grad_outputs = hyper_mega_mhc_grad(*gradients, *inputs, cache)
    return (*outputs, *grad_outputs)


def test_same_shape_multistream_forward_backward() -> None:
    """Compare concurrent invocations to serial native executions, not a performance golden."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    with torch.no_grad():
        cases = (_make_case(17), _make_case(23))
        # Warm both LRU plans on the default stream before testing counter isolation.
        expected = tuple(_run_case(*case) for case in cases)
        torch.npu.synchronize()
        default_stream = torch.npu.current_stream()
        streams = (torch.npu.Stream(), torch.npu.Stream())
        for stream in streams:
            stream.wait_stream(default_stream)
        pending = []
        for _ in range(4):
            for index, stream in enumerate(streams):
                with torch.npu.stream(stream):
                    pending.append((index, _run_case(*cases[index])))
        for stream in streams:
            stream.synchronize()
        for index, outputs in pending:
            for actual, serial in zip(outputs, expected[index]):
                torch.testing.assert_close(actual, serial, rtol=3e-3, atol=3e-3)


def _require_semantic_parity(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Use the existing BF16 backward benchmark's relative-L2 and cosine acceptance criteria."""
    actual_fp32 = actual.detach().float().flatten()
    expected_fp32 = expected.detach().float().flatten()
    expected_norm = expected_fp32.norm().item()
    error_norm = (actual_fp32 - expected_fp32).norm().item()
    relative_l2 = error_norm / max(expected_norm, 1e-12)
    cosine = (actual_fp32.dot(expected_fp32).item() / max(actual_fp32.norm().item() * expected_norm, 1e-12)
              if expected_norm > 1e-12 else 1.0)
    print(f"MHC_PARITY={label} relative_l2={relative_l2} cosine={cosine}", flush=True)
    assert torch.isfinite(actual_fp32).all().item(), f"expected finite {label}, actual contains nonfinite values"
    assert relative_l2 <= 2e-2 and cosine >= 0.999, (
        f"expected {label} relative_l2<=0.02 cosine>=0.999, actual relative_l2={relative_l2} cosine={cosine}"
    )


def _require_gradient_parity(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    """Report every gradient before failing, retaining the same per-tensor acceptance criteria."""
    failures = []
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        try:
            _require_semantic_parity(actual_tensor, expected_tensor, f"gradient-{index}")
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, f"expected all gradients within precision criteria, actual failures={failures}"


def _require_rms_cache_parity(cache: MegaMhcGradCache, previous_pre: torch.Tensor) -> None:
    """Validate backward intermediates independently of the normalized public forward output."""
    expected_mixed = (
        cache.new_residual.float() * previous_pre.reshape(*cache.new_residual.shape[:-1], 1).float()
    ).sum(dim=-2).to(cache.mixed_input.dtype)
    _require_semantic_parity(cache.mixed_input, expected_mixed, "cache-mixed-input")
    expected_rstd = torch.rsqrt(cache.mixed_input.float().square().mean(dim=-1, keepdim=True) + 1e-6)
    _require_semantic_parity(cache.rms_rstd, expected_rstd, "cache-rms-rstd")


def _run_reference_case(*, include_backward: bool) -> None:
    """Compare native results to independent Torch math without changing acceptance thresholds."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    inputs, gradients = _make_case(17)
    with torch.no_grad():
        outputs, cache = _hyper_mega_mhc_forward(*inputs, need_backward=True)
        if include_backward:
            actual_gradients = hyper_mega_mhc_grad(*gradients, *inputs, cache)
        expected_outputs = torch_mega_mhc(*inputs)
    if include_backward:
        expected_gradients = torch_mega_mhc_grad(*gradients, *inputs)
    torch.npu.synchronize()
    assert len(outputs) == len(expected_outputs) == 5, (
        f"expected=5 outputs, actual native={len(outputs)} reference={len(expected_outputs)}"
    )
    for index, (actual, expected) in enumerate(zip(outputs, expected_outputs)):
        _require_semantic_parity(actual, expected, f"output-{index}")
    _require_rms_cache_parity(cache, inputs[2])
    if include_backward:
        assert len(actual_gradients) == len(expected_gradients) == 9, (
            f"expected=9 gradients, actual native={len(actual_gradients)} reference={len(expected_gradients)}"
        )
        _require_gradient_parity(actual_gradients, expected_gradients)


def test_reference_forward() -> None:
    """Compare all five outputs without invoking the native or reference backward."""
    _run_reference_case(include_backward=False)


def test_reference_forward_backward() -> None:
    """Compare all five outputs and all nine input/parameter gradients to independent Torch math."""
    _run_reference_case(include_backward=True)
