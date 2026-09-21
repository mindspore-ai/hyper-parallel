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
"""Per-tensor FP32 acceptance for DeepSeek-V4.1 block validation."""

from __future__ import annotations

import torch


def fp32_accuracy(actual: torch.Tensor | None, expected: torch.Tensor | None) -> dict:
    """Check shape, presence, finiteness, relative L2 and peak-normalized error.

    Args:
        actual: Candidate tensor, or None for an absent gradient.
        expected: Independent FP32 reference with the same shape and presence.

    Returns:
        Acceptance and metrics. Zero references require exactly zero candidates;
        no tensor, rank or step averaging is performed.
    """
    if actual is None or expected is None:
        return {"passed": actual is None and expected is None, "gradient_absent": True}
    if actual.shape != expected.shape:
        return {"passed": False, "shape_matches": False}
    actual = actual.detach().float()
    expected = expected.detach().float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    if not finite:
        return {"passed": False, "shape_matches": True, "finite": False}
    difference = (actual - expected).abs()
    maximum = float(expected.abs().max()) if expected.numel() else 0.0
    max_error = float(difference.max()) if difference.numel() else 0.0
    norm = float(expected.norm())
    relative_l2 = float(difference.norm()) / norm if norm else (0.0 if max_error == 0 else float("inf"))
    peak_error = max_error / maximum if maximum else (0.0 if max_error == 0 else float("inf"))
    return {"passed": relative_l2 <= 0.01 and peak_error <= 0.02,
            "shape_matches": True, "finite": True, "relative_l2": relative_l2,
            "peak_normalized_error": peak_error, "max_abs_error": max_error,
            "reference_max_abs": maximum}
