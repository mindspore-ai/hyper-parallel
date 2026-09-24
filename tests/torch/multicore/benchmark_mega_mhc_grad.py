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
"""Single-NPU precision validation for HyperMegaMhcGrad."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.mega_mhc_grad.function import hyper_mega_mhc_grad
from hyper_parallel.core.multicore.modules.mega_mhc_grad.golden import (
    MegaMhcGradCache,
    MegaMhcGradOutputs,
    make_cann_grad_cache,
    torch_mega_mhc_grad,
)


def _make_case(tokens: int, hidden_size: int) -> tuple[tuple[torch.Tensor, ...], MegaMhcGradCache]:
    """Create deterministic forward inputs, caches, and five upstream gradients."""
    torch.manual_seed(17)
    device = torch.device("npu:0")
    residual = torch.randn(tokens, 4, hidden_size, device=device, dtype=torch.bfloat16)
    previous_output = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
    previous_pre = torch.sigmoid(torch.randn(tokens, 4, device=device))
    previous_post = 2.0 * torch.sigmoid(torch.randn(tokens, 4, device=device))
    previous_residual = torch.softmax(torch.randn(tokens, 4, 4, device=device), dim=-1)
    phi = torch.randn(24, 4 * hidden_size, device=device) * ((4 * hidden_size) ** -0.5)
    alpha = torch.ones(3, device=device)
    bias = torch.zeros(24, device=device)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
    forward_inputs = (
        previous_output,
        residual,
        previous_pre,
        previous_post,
        previous_residual,
        phi,
        alpha,
        bias,
        norm_weight,
    )
    cache = make_cann_grad_cache(*forward_inputs)
    upstream = (
        torch.randn_like(residual) * 0.1,
        torch.randn_like(previous_pre) * 0.1,
        torch.randn_like(previous_post) * 0.1,
        torch.randn_like(previous_residual) * 0.1,
        torch.randn_like(previous_output) * 0.1,
    )
    return (*upstream, *forward_inputs, cache), cache


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Calculate stable absolute, relative-L2, and cosine error statistics."""
    actual_fp32 = actual.float().flatten()
    expected_fp32 = expected.float().flatten()
    difference = actual_fp32 - expected_fp32
    expected_norm = torch.linalg.vector_norm(expected_fp32)
    difference_norm = torch.linalg.vector_norm(difference)
    denominator = max(expected_norm.item(), 1e-12)
    cosine = torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
        actual_fp32, expected_fp32, dim=0, eps=1e-12
    )
    return {
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "relative_l2": difference_norm.item() / denominator,
        "cosine": cosine.item(),
    }


def _compare(actual: MegaMhcGradOutputs, expected: MegaMhcGradOutputs) -> dict[str, Any]:
    """Compare all public input and parameter gradients."""
    names = (
        "grad_previous_output",
        "grad_residual",
        "grad_previous_pre",
        "grad_previous_post",
        "grad_previous_residual",
        "grad_phi",
        "grad_alpha",
        "grad_bias",
        "grad_norm_weight",
    )
    return {name: _metric(value, reference) for name, value, reference in zip(names, actual, expected)}


def _validate_precision(comparison: dict[str, dict[str, float]]) -> None:
    """Require the fused kernel to match the independent semantic reference."""
    for name, metrics in comparison.items():
        if metrics["relative_l2"] > 2e-2 or metrics["cosine"] < 0.999:
            raise AssertionError(f"{name} failed semantic accuracy: {metrics}")


def run(hidden_size: int) -> dict[str, Any]:
    """Validate fused backward against the independent semantic gradient.

    Args:
        hidden_size: Hidden dimension under test.
    """
    precision_inputs, _ = _make_case(128, hidden_size)
    with torch.no_grad():
        fused = hyper_mega_mhc_grad(*precision_inputs)
    semantic = torch_mega_mhc_grad(*precision_inputs)
    torch.npu.synchronize()
    fused_vs_torch = _compare(fused, semantic)
    _validate_precision(fused_vs_torch)
    return {
        "shape": {"tokens": 128, "hidden_size": hidden_size},
        "passed": True,
        "fused_vs_torch_semantic": fused_vs_torch,
    }


def _parse_args() -> argparse.Namespace:
    """Parse benchmark and output controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--output", help="optional benchmark JSON output")
    return parser.parse_args()


def main() -> None:
    """Run validation and print its machine-readable result."""
    args = _parse_args()
    result = run(args.hidden_size)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(serialized + "\n")


if __name__ == "__main__":
    main()
