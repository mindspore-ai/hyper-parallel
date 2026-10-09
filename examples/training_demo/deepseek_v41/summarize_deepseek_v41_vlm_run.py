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
"""Summarize a DeepSeek-V4.1 VLM training log and render its loss curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


_METRIC_LINE = re.compile(r"step=\d+[^\r\n]*training/total_loss=[^\s\r\n]+")
_METRIC = re.compile(r"([A-Za-z0-9_/]+)=([^\s\r\n]+)")
_CSV_FIELDS = (
    "step",
    "epoch",
    "data/consumed_samples",
    "data/consumed_tokens",
    "training/foundation_loss",
    "training/total_loss",
    "training/grad_norm",
    "training/lr",
    "performance/step_time",
    "performance/tokens_per_second",
    "memory/device_max_allocated_gb",
    "memory/device_max_reserved_gb",
)


def _parse_metrics(log_path: Path) -> list[dict[str, Any]]:
    """Parse and deduplicate rank-zero step metrics from a trainer log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    by_step: dict[int, dict[str, Any]] = {}
    for match in _METRIC_LINE.finditer(text):
        values = dict(_METRIC.findall(match.group(0)))
        step = int(values["step"])
        row: dict[str, Any] = {"step": step, "epoch": int(values["epoch"])}
        for field in _CSV_FIELDS[2:]:
            value = values[field]
            row[field] = int(value) if field.startswith("data/") else float(value)
        by_step[step] = row
    return [by_step[step] for step in sorted(by_step)]


def _validate_metrics(rows: list[dict[str, Any]], expected_steps: int | None) -> None:
    """Require a contiguous, finite training curve."""
    if not rows:
        raise ValueError("the log contains no complete training-step metrics")
    expected_indices = list(range(len(rows)))
    actual_indices = [int(row["step"]) for row in rows]
    if actual_indices != expected_indices:
        raise ValueError(f"training steps are not contiguous: {actual_indices[:5]} ... {actual_indices[-5:]}")
    if expected_steps is not None and len(rows) != expected_steps:
        raise ValueError(f"expected {expected_steps} steps, found {len(rows)}")
    for row in rows:
        for field in ("training/foundation_loss", "training/total_loss", "training/grad_norm"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"step {row['step']} has non-finite {field}: {row[field]}")


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write the selected trainer metrics as a stable CSV artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _moving_average(values: list[float], window: int) -> list[float]:
    """Return a trailing moving average with a growing initial window."""
    averages = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        averages.append(running_sum / min(index + 1, window))
    return averages


def _polyline(values: list[float], *, width: int, height: int, margin: int, y_min: float, y_max: float) -> str:
    """Convert a scalar series into SVG polyline coordinates."""
    x_span = width - 2 * margin
    y_span = height - 2 * margin
    denominator = max(len(values) - 1, 1)
    points = []
    for index, value in enumerate(values):
        x_position = margin + index * x_span / denominator
        y_position = height - margin - (value - y_min) * y_span / (y_max - y_min)
        points.append(f"{x_position:.2f},{y_position:.2f}")
    return " ".join(points)


def _write_svg(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Render raw loss and a ten-step moving average as a standalone SVG."""
    losses = [float(row["training/total_loss"]) for row in rows]
    smoothed = _moving_average(losses, window=10)
    width, height, margin = 1000, 520, 70
    raw_min, raw_max = min(losses), max(losses)
    padding = max((raw_max - raw_min) * 0.08, 0.05)
    y_min, y_max = raw_min - padding, raw_max + padding
    raw_points = _polyline(losses, width=width, height=height, margin=margin, y_min=y_min, y_max=y_max)
    smooth_points = _polyline(smoothed, width=width, height=height, margin=margin, y_min=y_min, y_max=y_max)
    grid_lines = []
    for index in range(6):
        y = margin + index * (height - 2 * margin) / 5
        value = y_max - index * (y_max - y_min) / 5
        grid_lines.append(
            f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" y2="{y:.2f}" '
            'stroke="#d9e2ec" stroke-width="1"/>'
            f'<text x="{margin - 10}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="14" fill="#52606d">{value:.3f}</text>'
        )
    x_labels = []
    for index in range(5):
        step = round(index * (len(rows) - 1) / 4)
        x = margin + index * (width - 2 * margin) / 4
        x_labels.append(
            f'<text x="{x:.2f}" y="{height - margin + 28}" text-anchor="middle" '
            f'font-size="14" fill="#52606d">{step}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="34" text-anchor="middle" font-size="22"
      font-family="sans-serif" fill="#102a43">DeepSeek-V4.1-Flash crop: 100-step training loss</text>
{''.join(grid_lines)}
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#243b53"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#243b53"/>
{''.join(x_labels)}
<text x="{width / 2}" y="{height - 15}" text-anchor="middle" font-size="16"
      font-family="sans-serif" fill="#334e68">Training step</text>
<text x="18" y="{height / 2}" text-anchor="middle" font-size="16" font-family="sans-serif"
      fill="#334e68" transform="rotate(-90 18 {height / 2})">Cross-entropy loss</text>
<polyline points="{raw_points}" fill="none" stroke="#829ab1" stroke-width="1.5" opacity="0.7"/>
<polyline points="{smooth_points}" fill="none" stroke="#007d8a" stroke-width="3"/>
<line x1="{width - 275}" y1="58" x2="{width - 235}" y2="58" stroke="#829ab1" stroke-width="2"/>
<text x="{width - 225}" y="63" font-size="14" font-family="sans-serif" fill="#334e68">raw loss</text>
<line x1="{width - 145}" y1="58" x2="{width - 105}" y2="58" stroke="#007d8a" stroke-width="3"/>
<text x="{width - 95}" y="63" font-size="14" font-family="sans-serif" fill="#334e68">10-step MA</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    """Parse arguments and create reproducible curve artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()

    rows = _parse_metrics(args.log)
    _validate_metrics(rows, args.expected_steps)
    _write_csv(rows, args.csv)
    _write_svg(rows, args.svg)
    losses = [float(row["training/total_loss"]) for row in rows]
    step_times = [float(row["performance/step_time"]) for row in rows]
    summary = {
        "steps": len(rows),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_mean_last_10": sum(losses[-10:]) / min(len(losses), 10),
        "step_time_mean_seconds": sum(step_times) / len(step_times),
        "step_time_mean_seconds_excluding_first": sum(step_times[1:]) / max(len(step_times) - 1, 1),
        "max_allocated_gb": max(float(row["memory/device_max_allocated_gb"]) for row in rows),
        "max_reserved_gb": max(float(row["memory/device_max_reserved_gb"]) for row in rows),
        "grad_norm_min": min(float(row["training/grad_norm"]) for row in rows),
        "grad_norm_max": max(float(row["training/grad_norm"]) for row in rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
