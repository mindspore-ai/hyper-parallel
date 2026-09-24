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
"""Single-NPU BF16 precision validation for HyperMegaMhc."""

from __future__ import annotations

import argparse
import json

import torch
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.mega_mhc.function import hyper_mega_mhc
from hyper_parallel.core.multicore.modules.mega_mhc.golden import (
    MegaMhcOutputs,
    cann_mega_mhc,
    torch_mega_mhc,
)


def _make_inputs(tokens: int, hidden_size: int) -> tuple[torch.Tensor, ...]:
    """Create one deterministic shifted-mHC boundary on NPU 0."""
    torch.manual_seed(11)
    device = torch.device("npu:0")
    residual = torch.randn(tokens, 4, hidden_size, device=device, dtype=torch.bfloat16)
    previous_output = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
    previous_pre = torch.sigmoid(torch.randn(tokens, 4, device=device))
    previous_post = 2.0 * torch.sigmoid(torch.randn(tokens, 4, device=device))
    previous_res = torch.softmax(torch.randn(tokens, 4, 4, device=device), dim=-1)
    phi = torch.randn(24, 4 * hidden_size, device=device) * ((4 * hidden_size) ** -0.5)
    alpha = torch.ones(3, device=device)
    bias = torch.zeros(24, device=device)
    norm_weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    return (
        previous_output,
        residual,
        previous_pre,
        previous_post,
        previous_res,
        phi,
        alpha,
        bias,
        norm_weight,
    )


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Calculate absolute error and cosine similarity in FP32."""
    actual_fp32 = actual.detach().cpu().float()
    expected_fp32 = expected.detach().cpu().float()
    difference = (actual_fp32 - expected_fp32).abs()
    cosine = torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
        actual_fp32.flatten(), expected_fp32.flatten(), dim=0
    )
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "cosine": cosine.item(),
    }


def _compare(actual: MegaMhcOutputs, expected: MegaMhcOutputs) -> dict[str, dict[str, float]]:
    """Compare all public outputs."""
    names = ("new_residual", "next_pre", "next_post", "next_residual", "block_input")
    return {name: _metric(value, reference) for name, value, reference in zip(names, actual, expected)}


def _validate_precision(comparison: dict[str, dict[str, float]]) -> None:
    """Enforce BF16 acceptance thresholds for every fused output."""
    for name, metrics in comparison.items():
        minimum_cosine = 0.9999 if name == "block_input" else 0.99999
        if metrics["cosine"] < minimum_cosine:
            raise AssertionError(
                f"{name} cosine {metrics['cosine']} is below {minimum_cosine}"
            )
        maximum_mean_abs = 2e-3 if name == "block_input" else 2e-4
        if metrics["mean_abs"] > maximum_mean_abs:
            raise AssertionError(
                f"{name} mean abs {metrics['mean_abs']} exceeds {maximum_mean_abs}"
            )


def run(hidden_size: int, token_tile: int) -> dict[str, object]:
    """Validate the fused operator against CANN and semantic references.

    Args:
        hidden_size: Hidden dimension under test.
        token_tile: Number of tokens assigned to each logical task.
    """
    precision_inputs = _make_inputs(128, hidden_size)

    def fused_function(*inputs: torch.Tensor) -> MegaMhcOutputs:
        """Run the candidate with the selected token tile."""
        return hyper_mega_mhc(*inputs, need_backward=True, token_tile=token_tile)
    with torch.no_grad():
        fused = fused_function(*precision_inputs)
        fused_without_backward = hyper_mega_mhc(
            *precision_inputs, need_backward=False, token_tile=token_tile
        )
        cann = cann_mega_mhc(*precision_inputs)
        semantic = torch_mega_mhc(*precision_inputs)
        torch.npu.synchronize()
        fused_vs_cann = _compare(fused, cann)
        fused_vs_torch = _compare(fused, semantic)
        _validate_precision(fused_vs_cann)
        _validate_precision(fused_vs_torch)
        without_backward_vs_fused = _compare(fused_without_backward, fused)
        _validate_precision(without_backward_vs_fused)
        return {
            "shape": {"tokens": 128, "hidden_size": hidden_size, "token_tile": token_tile},
            "passed": True,
            "fused_vs_cann": fused_vs_cann,
            "fused_vs_torch": fused_vs_torch,
            "without_backward_vs_fused": without_backward_vs_fused,
        }


def _parse_args() -> argparse.Namespace:
    """Parse command-line benchmark controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--token-tile", type=int, default=32)
    parser.add_argument("--output", help="optional JSON result path")
    return parser.parse_args()


def main() -> None:
    """Run and print a machine-readable validation result."""
    args = _parse_args()
    if args.token_tile <= 0:
        raise ValueError(f"token_tile must be positive, got {args.token_tile}")
    result = run(args.hidden_size, args.token_tile)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(serialized + "\n")


if __name__ == "__main__":
    main()
