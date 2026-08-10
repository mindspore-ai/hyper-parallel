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
"""Extract rank-zero reward metrics from a training log and render an SVG curve."""

import argparse
import csv
import re
from pathlib import Path

_METRIC_LINE = re.compile(r"step=(?P<step>\d+)\s+\|\s+(?P<metrics>.*)")
_METRIC = re.compile(r"(?P<name>[a-zA-Z0-9_/-]+)=(?P<value>[-+0-9.eE]+)")


def parse_rewards(log_path: Path) -> list[tuple[int, float]]:
    """Parse and deduplicate ``reward/mean`` values by optimizer step."""
    by_step: dict[int, float] = {}
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _METRIC_LINE.search(line)
            if match is None:
                continue
            metrics = {
                item.group("name"): float(item.group("value"))
                for item in _METRIC.finditer(match.group("metrics"))
            }
            if "reward/mean" in metrics:
                by_step[int(match.group("step"))] = metrics["reward/mean"]
    if not by_step:
        raise ValueError(f"No reward/mean metrics found in {log_path}")
    return sorted(by_step.items())


def write_csv(points: list[tuple[int, float]], output_path: Path) -> None:
    """Write machine-readable reward points."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "reward_mean"))
        writer.writerows(points)


def _polyline_points(points: list[tuple[int, float]]) -> str:
    """Map step/reward pairs into the fixed chart viewport."""
    left, right = 100.0, 940.0
    top, bottom = 120.0, 500.0
    min_step, max_step = points[0][0], points[-1][0]
    step_span = max(max_step - min_step, 1)
    rendered = []
    for step, reward in points:
        x_coord = left + (step - min_step) / step_span * (right - left)
        y_coord = bottom - max(0.0, min(1.0, reward)) * (bottom - top)
        rendered.append(f"{x_coord:.1f},{y_coord:.1f}")
    return " ".join(rendered)


def write_svg(points: list[tuple[int, float]], output_path: Path) -> None:
    """Render reward values on a fixed zero-to-one y axis."""
    point_text = _polyline_points(points)
    circles = []
    left, right = 100.0, 940.0
    top, bottom = 120.0, 500.0
    min_step, max_step = points[0][0], points[-1][0]
    step_span = max(max_step - min_step, 1)
    for step, reward in points:
        x_coord = left + (step - min_step) / step_span * (right - left)
        y_coord = bottom - max(0.0, min(1.0, reward)) * (bottom - top)
        circles.append(
            f'<circle cx="{x_coord:.1f}" cy="{y_coord:.1f}" r="5" fill="#34d399">'
            f"<title>step {step}: {reward:.6f}</title></circle>"
        )
    grid_lines = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_coord = bottom - tick * (bottom - top)
        grid_lines.append(
            f'<line x1="{left:.0f}" y1="{y_coord:.1f}" x2="{right:.0f}" '
            f'y2="{y_coord:.1f}" stroke="#26354f" stroke-width="1"/>'
            f'<text x="88" y="{y_coord + 5:.1f}" fill="#94a3b8" '
            f'text-anchor="end" font-family="monospace" font-size="13">{tick:.2f}</text>'
        )
    rewards = [reward for _, reward in points]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="620" viewBox="0 0 1040 620">
<rect width="1040" height="620" fill="#0b1220"/>
<text x="70" y="55" fill="#f8fafc" font-family="sans-serif" font-size="27"
      font-weight="700">Hyper-RL Qwen3.5 GRPO reward</text>
<text x="70" y="84" fill="#94a3b8" font-family="sans-serif" font-size="14">GSM8K rule reward · 2-card Ascend FSDP</text>
{''.join(grid_lines)}
<line x1="100" y1="120" x2="100" y2="500" stroke="#64748b" stroke-width="2"/>
<line x1="100" y1="500" x2="940" y2="500" stroke="#64748b" stroke-width="2"/>
<polyline points="{point_text}" fill="none" stroke="#34d399" stroke-width="4"
          stroke-linejoin="round" stroke-linecap="round"/>
{''.join(circles)}
<text x="100" y="530" fill="#94a3b8" font-family="monospace" font-size="13">step {min_step}</text>
<text x="940" y="530" fill="#94a3b8" text-anchor="end" font-family="monospace" font-size="13">step {max_step}</text>
<text x="100" y="570" fill="#cbd5e1" font-family="monospace" font-size="14">
points {len(points)} · min {min(rewards):.4f} · max {max(rewards):.4f} · final {rewards[-1]:.4f}</text>
</svg>
'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    """Parse command-line paths and create CSV/SVG artifacts."""
    parser = argparse.ArgumentParser(description="Render a Hyper-RL reward curve")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("svg_path", type=Path)
    parser.add_argument("--csv-path", type=Path, default=None)
    args = parser.parse_args()
    points = parse_rewards(args.log_path)
    csv_path = args.csv_path or args.svg_path.with_suffix(".csv")
    write_csv(points, csv_path)
    write_svg(points, args.svg_path)
    print(f"wrote {len(points)} reward points to {csv_path} and {args.svg_path}")


if __name__ == "__main__":
    main()
