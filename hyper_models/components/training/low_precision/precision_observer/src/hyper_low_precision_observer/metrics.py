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
"""Mergeable error moments for baseline-to-candidate comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


_MOMENT_COUNT = 4
_SCALE_INDEX = _MOMENT_COUNT


@dataclass
class DeviceErrorMetrics:
    """Mergeable error moments that remain on the source device."""

    count: int
    moment_values: torch.Tensor
    count_values: torch.Tensor

    def merge_(self, other: "DeviceErrorMetrics") -> "DeviceErrorMetrics":
        """Merge another device-resident metric into this instance."""
        if (
            self.moment_values.device != other.moment_values.device
            or self.count_values.device != other.count_values.device
        ):
            raise ValueError("Device error moments must be on the same device")
        self_scale = self.moment_values[_SCALE_INDEX]
        other_scale = other.moment_values[_SCALE_INDEX]
        merged_scale = torch.maximum(self_scale, other_scale)
        safe_scale = torch.where(
            merged_scale > 0,
            merged_scale,
            torch.ones_like(merged_scale),
        )
        self_factor = (self_scale / safe_scale).square()
        other_factor = (other_scale / safe_scale).square()
        self.moment_values[:_MOMENT_COUNT].mul_(self_factor).add_(
            other.moment_values[:_MOMENT_COUNT] * other_factor
        )
        self.moment_values[_SCALE_INDEX].copy_(merged_scale)
        self.count += other.count
        self.count_values.add_(other.count_values)
        return self

    def to_host(self) -> "ErrorMetrics":
        """Materialize this metric as host scalar values."""
        return materialize_device_metrics({"metric": self})["metric"]


@dataclass(frozen=True)
class ErrorMetrics:
    count: int
    sq_error_sum: float
    baseline_sq_sum: float
    candidate_sq_sum: float
    dot_sum: float
    nonfinite_count: int
    clipped_count: int = 0
    underflow_count: int = 0
    baseline_nonfinite_count: int = 0
    candidate_nonfinite_count: int = 0

    @classmethod
    def compare(
        cls,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
        *,
        clipped_mask: torch.Tensor | None = None,
        underflow_mask: torch.Tensor | None = None,
        quantized: torch.Tensor | None = None,
        quantized_max: float | None = None,
    ) -> "ErrorMetrics":
        """Compare two tensors and return host scalar error moments."""
        return cls.compare_device(
            baseline,
            candidate,
            clipped_mask=clipped_mask,
            underflow_mask=underflow_mask,
            quantized=quantized,
            quantized_max=quantized_max,
        ).to_host()

    @classmethod
    def compare_device(
        cls,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
        *,
        clipped_mask: torch.Tensor | None = None,
        underflow_mask: torch.Tensor | None = None,
        quantized: torch.Tensor | None = None,
        quantized_max: float | None = None,
    ) -> DeviceErrorMetrics:
        """Compare two tensors while retaining reduced moments on device."""
        baseline_values = baseline.detach().float().reshape(-1)
        candidate_values = candidate.detach().float().reshape(-1)
        if baseline_values.numel() != candidate_values.numel():
            raise ValueError(
                "Baseline and candidate tensors must have the same number of elements"
            )
        baseline_finite = torch.isfinite(baseline_values)
        candidate_finite = torch.isfinite(candidate_values)
        finite = baseline_finite & candidate_finite
        zero_count = torch.zeros(
            (),
            dtype=torch.int64,
            device=baseline_values.device,
        )
        event_count_tensors = _event_counts(
            baseline_values,
            candidate_values,
            finite,
            zero_count,
            clipped_mask=clipped_mask,
            underflow_mask=underflow_mask,
            quantized=quantized,
            quantized_max=quantized_max,
        )

        return DeviceErrorMetrics(
            count=baseline_values.numel(),
            moment_values=_scaled_error_moments(
                baseline_values,
                candidate_values,
                finite,
            ),
            count_values=torch.stack((
                (~finite).sum(dtype=torch.int64),
                *event_count_tensors,
                (~baseline_finite).sum(dtype=torch.int64),
                (~candidate_finite).sum(dtype=torch.int64),
            )),
        )

    def merge(self, other: "ErrorMetrics") -> "ErrorMetrics":
        """Return the sum of two mergeable host metrics."""
        return ErrorMetrics(
            count=self.count + other.count,
            sq_error_sum=self.sq_error_sum + other.sq_error_sum,
            baseline_sq_sum=self.baseline_sq_sum + other.baseline_sq_sum,
            candidate_sq_sum=self.candidate_sq_sum + other.candidate_sq_sum,
            dot_sum=self.dot_sum + other.dot_sum,
            nonfinite_count=self.nonfinite_count + other.nonfinite_count,
            clipped_count=self.clipped_count + other.clipped_count,
            underflow_count=self.underflow_count + other.underflow_count,
            baseline_nonfinite_count=(
                self.baseline_nonfinite_count
                + other.baseline_nonfinite_count
            ),
            candidate_nonfinite_count=(
                self.candidate_nonfinite_count
                + other.candidate_nonfinite_count
            ),
        )

    def to_dict(self) -> dict[str, float | int]:
        """Derive user-facing error metrics from raw moments."""
        epsilon = 1.0e-30
        cosine_denominator = (
            self.baseline_sq_sum * self.candidate_sq_sum
        ) ** 0.5
        finite_count = self.count - self.nonfinite_count
        cosine = (
            self.dot_sum / cosine_denominator
            if cosine_denominator > epsilon
            else float(self.sq_error_sum <= epsilon)
        )
        return {
            "count": self.count,
            "mse": self.sq_error_sum / max(finite_count, 1),
            "nrmse": (
                self.sq_error_sum / max(self.baseline_sq_sum, epsilon)
            ) ** 0.5,
            "cosine": cosine,
            "nonfinite_rate": self.nonfinite_count / max(self.count, 1),
            "clip_rate": self.clipped_count / max(self.count, 1),
            "underflow_rate": self.underflow_count / max(self.count, 1),
        }

    def raw_dict(self) -> dict[str, float | int]:
        """Serialize mergeable raw moments."""
        return asdict(self)

    @classmethod
    def from_raw_dict(cls, value: Mapping[str, Any]) -> "ErrorMetrics":
        """Restore mergeable raw moments from an artifact mapping."""
        return cls(
            count=int(value["count"]),
            sq_error_sum=float(value["sq_error_sum"]),
            baseline_sq_sum=float(value["baseline_sq_sum"]),
            candidate_sq_sum=float(value["candidate_sq_sum"]),
            dot_sum=float(value["dot_sum"]),
            nonfinite_count=int(value["nonfinite_count"]),
            clipped_count=int(value.get("clipped_count", 0)),
            underflow_count=int(value.get("underflow_count", 0)),
            baseline_nonfinite_count=int(
                value.get("baseline_nonfinite_count", 0)
            ),
            candidate_nonfinite_count=int(
                value.get("candidate_nonfinite_count", 0)
            ),
        )


def _scaled_error_moments(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    finite: torch.Tensor,
) -> torch.Tensor:
    """Return normalized moments and their common absolute-value scale."""
    finite_baseline = torch.where(finite, baseline, 0.0)
    finite_candidate = torch.where(finite, candidate, 0.0)
    if baseline.numel() == 0:
        value_scale = baseline.new_zeros(())
    else:
        value_scale = torch.maximum(
            finite_baseline.abs().amax(),
            finite_candidate.abs().amax(),
        )
    safe_scale = torch.where(
        value_scale > 0,
        value_scale,
        torch.ones_like(value_scale),
    )
    scaled_baseline = finite_baseline / safe_scale
    scaled_candidate = finite_candidate / safe_scale
    scaled_diff = scaled_candidate - scaled_baseline
    return torch.stack((
        torch.dot(scaled_diff, scaled_diff),
        torch.dot(scaled_baseline, scaled_baseline),
        torch.dot(scaled_candidate, scaled_candidate),
        torch.dot(scaled_baseline, scaled_candidate),
        value_scale,
    ))


def _event_counts(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    finite: torch.Tensor,
    zero_count: torch.Tensor,
    *,
    clipped_mask: torch.Tensor | None,
    underflow_mask: torch.Tensor | None,
    quantized: torch.Tensor | None,
    quantized_max: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count quantization events while reusing converted error tensors."""
    if quantized is not None:
        if clipped_mask is not None or underflow_mask is not None:
            raise ValueError("quantized events and explicit event masks are mutually exclusive")
        if quantized_max is None:
            raise ValueError("quantized_max is required with quantized events")
        quantized_values = quantized.detach().float().reshape(-1)
        if quantized_values.numel() != baseline.numel():
            raise ValueError("Quantized tensor must match the compared tensor size")
        clipped = (
            (quantized_values.abs() >= quantized_max)
            & (baseline.abs() > candidate.abs())
            & finite
        )
        underflow = (
            (baseline != 0)
            & (quantized_values == 0)
            & finite
        )
        return (
            clipped.sum(dtype=torch.int64),
            underflow.sum(dtype=torch.int64),
        )
    if quantized_max is not None:
        raise ValueError("quantized_max requires a quantized tensor")

    counts = []
    for name, mask in (
        ("clipped", clipped_mask),
        ("underflow", underflow_mask),
    ):
        if mask is None:
            counts.append(zero_count)
            continue
        flattened = mask.detach().bool().reshape(-1)
        if flattened.numel() != baseline.numel():
            raise ValueError(f"{name} mask must match the compared tensor size")
        counts.append((flattened & finite).sum(dtype=torch.int64))
    return counts[0], counts[1]


def materialize_device_metrics(
    metrics: Mapping[str, DeviceErrorMetrics],
) -> dict[str, ErrorMetrics]:
    """Batch all device moments into two host transfers for one sampled step."""
    if not metrics:
        return {}
    items = tuple(metrics.items())
    moment_rows = torch.stack([
        metric.moment_values
        for _, metric in items
    ]).cpu().tolist()
    count_rows = torch.stack([
        metric.count_values
        for _, metric in items
    ]).cpu().tolist()
    return {
        name: _materialize_error_metrics(
            metric.count,
            moment_values,
            count_values,
        )
        for (
            (name, metric),
            moment_values,
            count_values,
        ) in zip(items, moment_rows, count_rows)
    }


def _materialize_error_metrics(
    count: int,
    moment_values: list[float],
    count_values: list[int],
) -> ErrorMetrics:
    """Restore unscaled host moments from one device reduction row."""
    scale_squared = float(moment_values[_SCALE_INDEX]) ** 2
    return ErrorMetrics(
        count=count,
        sq_error_sum=float(moment_values[0]) * scale_squared,
        baseline_sq_sum=float(moment_values[1]) * scale_squared,
        candidate_sq_sum=float(moment_values[2]) * scale_squared,
        dot_sum=float(moment_values[3]) * scale_squared,
        nonfinite_count=int(count_values[0]),
        clipped_count=int(count_values[1]),
        underflow_count=int(count_values[2]),
        baseline_nonfinite_count=int(count_values[3]),
        candidate_nonfinite_count=int(count_values[4]),
    )
