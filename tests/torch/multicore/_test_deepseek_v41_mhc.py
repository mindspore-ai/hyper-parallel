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
"""Independent V4.1 oracle for fused outputs and every input/parameter gradient."""

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from hyper_parallel.components.modules import PipelinedMhc
from hyper_parallel.models.deepseek_v41.adapter.distributed.mega_mhc import (
    DeepseekV41HyperMegaMhc,
    _reference_shifted_boundary,
)
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41PipelinedHyperConnection


class _Coefficients(nn.Module):
    """Supply source-shaped coefficient state to the actual V4.1 reference module."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(4 * hidden, eps=1e-6, elementwise_affine=False)
        self.fn = nn.Parameter(torch.randn(24, 4 * hidden) / (4 * hidden) ** 0.5)
        self.base = nn.Parameter(torch.randn(24) * 0.02)
        self.scale = nn.Parameter(torch.ones(3))
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = 4, 20, 1e-6


def _metric(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Apply the upstream BF16 semantic relative-L2/cosine acceptance thresholds."""
    actual, expected = actual.detach().float().cpu().flatten(), expected.detach().float().cpu().flatten()
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        raise AssertionError("Non-finite fused mHC value")
    difference = actual - expected
    relative = float(difference.norm()) / max(float(expected.norm()), 1e-12)
    cosine = float(actual.dot(expected)) / max(float(actual.norm() * expected.norm()), 1e-24)
    if expected.norm() == 0 and actual.norm() == 0:
        cosine = 1.0
    result = {"max_abs": float(difference.abs().max()), "relative_l2": relative, "cosine": cosine}
    if relative > 2e-2 or cosine < 0.999:
        raise AssertionError(f"Fused mHC semantic mismatch: {result}")
    return result


def _inputs(tokens: int, hidden: int) -> list[torch.Tensor]:
    """Include non-unit norm weights and nontrivial mappings in the backward oracle."""
    return [
        torch.randn(1, tokens, hidden, dtype=torch.bfloat16),
        torch.randn(1, tokens, 4, hidden, dtype=torch.bfloat16),
        torch.sigmoid(torch.randn(1, tokens, 4)),
        2 * torch.sigmoid(torch.randn(1, tokens, 4)),
        torch.softmax(torch.randn(1, tokens, 4, 4), dim=-1),
        (1 + 0.1 * torch.randn(hidden)).bfloat16(),
    ]


def _run_case(tokens: int, hidden: int, implementation: str) -> dict:
    """Run independent cloned state through CPU eager and the native NPU autograd kernel."""
    torch.manual_seed(42 + tokens + hidden)
    reference = DeepseekV41PipelinedHyperConnection(_Coefficients(hidden))
    source = deepcopy(reference)
    candidate = (PipelinedMhc(module=source, shifted=True, use_fused_ops=True)
                 if implementation == "family" else DeepseekV41HyperMegaMhc(module=source))
    candidate = candidate.to("npu")
    values = _inputs(tokens, hidden)
    cpu = [value.clone().requires_grad_() for value in values]
    npu = [value.clone().to("npu").requires_grad_() for value in values]
    expected = _reference_shifted_boundary(reference, *cpu, 1e-6)
    actual = candidate.advance(*npu)
    output_names = ("residual", "next_pre", "next_post", "next_mix", "block_input")
    outputs = {name: _metric(a, b) for name, a, b in zip(output_names, actual, expected)}
    torch.manual_seed(2026)
    upstream = [torch.randn_like(value) / value.numel() for value in expected]
    sum((value * grad).sum() for value, grad in zip(expected, upstream)).backward()
    sum((value * grad.to("npu")).sum() for value, grad in zip(actual, upstream)).backward()
    input_names = ("previous_output", "residual", "previous_pre", "previous_post", "previous_mix", "norm_weight")
    gradients = {name: _metric(a.grad, b.grad) for name, a, b in zip(input_names, npu, cpu)}
    for name, parameter in reference.named_parameters():
        gradients[name] = _metric(candidate.get_parameter(name).grad, parameter.grad)
    return {"tokens": tokens, "hidden": hidden, "implementation": implementation,
            "outputs": outputs, "gradients": gradients, "grad_fn": type(actual[0].grad_fn).__name__}


def test_fused_mhc_native_parity() -> None:
    """Cover cropped/full hidden sizes, CP-local token count and both supported callers."""
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    results = []
    for tokens, hidden in ((128, 640), (256, 640), (128, 5120)):
        for implementation in ("family", "legacy"):
            result = _run_case(tokens, hidden, implementation)
            results.append(result)
            print(json.dumps(result), flush=True)
    output = os.environ.get("HP_FUSED_MHC_PARITY_REPORT")
    if output:
        Path(output).write_text(json.dumps({"passed": True, "cases": results}, indent=2), encoding="utf-8")
