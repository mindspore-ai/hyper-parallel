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
"""Shared peak-memory checks for MindSpore pipeline-swap system tests."""
from collections.abc import Sequence

import mindspore as ms

_MB = 1024 ** 2
_LEAK_ABSOLUTE_TOLERANCE_MB = 8.0
_LEAK_RELATIVE_TOLERANCE = 0.005


def reset_step_peak_memory() -> None:
    """Reset allocator peak statistics immediately before one measured step."""
    ms.runtime.empty_cache()
    ms.runtime.reset_peak_memory_stats()


def current_step_peak_memory_mb() -> float:
    """Synchronize and return the current step's device peak memory in MB."""
    ms.runtime.current_stream().synchronize()
    return ms.runtime.max_memory_allocated() / _MB


def steady_peak_memory_mb(step_peaks_mb: Sequence[float]) -> float:
    """Return the maximum peak after discarding the first warm-up step.

    Args:
        step_peaks_mb: Per-step peaks collected after resetting peak stats.

    Returns:
        The largest post-warm-up step peak.

    Raises:
        ValueError: If fewer than three steps were measured.
    """
    if len(step_peaks_mb) < 3:
        raise ValueError(f"At least three step peaks are required, got {len(step_peaks_mb)}")
    return max(step_peaks_mb[1:])


def assert_step_memory_stable(step_peaks_mb: Sequence[float], scene: str) -> None:
    """Assert that post-warm-up peaks do not grow across repeated steps.

    Resetting the peak counter does not release live tensors. A session or
    storage leak therefore raises the next step's baseline and is visible as
    growth in this sequence.

    Args:
        step_peaks_mb: Per-step peaks collected after resetting peak stats.
        scene: Scenario label used in assertion diagnostics.
    """
    steady_peaks = list(step_peaks_mb[1:])
    if len(steady_peaks) < 2:
        raise ValueError(f"{scene} needs at least two post-warm-up peaks, got {len(steady_peaks)}")
    split_index = max(1, len(steady_peaks) // 2)
    early_peak_mb = max(steady_peaks[:split_index])
    late_peak_mb = max(steady_peaks[split_index:])
    tolerance_mb = max(
        _LEAK_ABSOLUTE_TOLERANCE_MB,
        early_peak_mb * _LEAK_RELATIVE_TOLERANCE,
    )
    assert late_peak_mb <= early_peak_mb + tolerance_mb, (
        f"{scene} step peak grew after warm-up: peaks={step_peaks_mb}, "
        f"allowed_growth={tolerance_mb:.1f} MB"
    )


def format_step_peaks(step_peaks_mb: Sequence[float]) -> str:
    """Format per-step peaks for distributed worker logs."""
    return "[" + ", ".join(f"{peak:.1f}" for peak in step_peaks_mb) + "] MB"
