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
"""Plot HotpotQA training rewards as offline SVG and CSV files."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from xml.sax.saxutils import escape

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP = re.compile(r"rl\.utils\.monitoring \| step=(\d+) \|")
SMOOTH_WINDOW = 20


def read_rewards(path: Path) -> list[tuple[int, float, float, float]]:
    """Keep the latest complete reward record for each optimizer step."""
    records = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STEP.search(line)
        if not match:
            continue
        values = []
        for name in ("reward/mean", "reward/max", "reward/std"):
            found = re.search(rf"(?:^|[, ]){re.escape(name)}=({NUMBER})(?:[, ]|$)", line)
            if not found:
                break
            values.append(float(found[1]))
        if len(values) == 3:
            step = int(match[1])
            records[step] = (step, *values)
    return [records[step] for step in sorted(records)]


def _moving_average(values: list[float]) -> list[float]:
    """Use a trailing twenty-step window, shortened at startup."""
    total = 0.0
    averages = []
    for index, value in enumerate(values):
        total += value
        if index >= SMOOTH_WINDOW:
            total -= values[index - SMOOTH_WINDOW]
        averages.append(total / min(index + 1, SMOOTH_WINDOW))
    return averages


def write_plot(records: list[tuple[int, float, float, float]], path: Path, *,
               metric: str = "mean", only_smoothed: bool = False) -> None:
    """Write raw and smoothed mean or max reward curves to one SVG."""
    if metric not in {"mean", "max"}:
        raise ValueError(f"Unsupported reward metric: {metric}")
    width, height = 900, 520
    left, right, top, bottom = 75, 30, 45, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    first, last = records[0][0], records[-1][0]
    values = [row[1 if metric == "mean" else 2] for row in records]
    ceiling = max([0.1, *values]) * 1.1

    def x(step: int) -> float:
        return left + (step - first) * plot_width / max(1, last - first)

    def y(value: float) -> float:
        return top + plot_height * (1 - value / ceiling)

    label = metric.title()
    series = [
        (f"{label} (raw)", "#127c85" if metric == "mean" else "#d49324", values),
        (f"{label} ({SMOOTH_WINDOW}-step)", "#164e9b", _moving_average(values)),
    ]
    if only_smoothed:
        series = series[1:]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="20" fill="#222">'
        f'HotpotQA Codex GRPO {label} reward by optimizer step</text>',
    ]
    for tick in range(6):
        value = ceiling * tick / 5
        ordinate = y(value)
        lines.extend((
            f'<line x1="{left}" y1="{ordinate:.1f}" x2="{width-right}" y2="{ordinate:.1f}" stroke="#e5e8eb"/>',
            f'<text x="{left-10}" y="{ordinate+4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12" fill="#555">{value:.2f}</text>',
        ))
    for step in sorted({first, last, *range(first, last + 1, 5)}):
        lines.append(
            f'<text x="{x(step):.1f}" y="{height-bottom+24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#555">{step}</text>'
        )
    for index, (name, color, curve) in enumerate(series):
        points = " ".join(f"{x(row[0]):.1f},{y(value):.1f}" for row, value in zip(records, curve))
        legend_x = left + index * 205
        lines.extend((
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>',
            f'<line x1="{legend_x}" y1="{height-18}" x2="{legend_x+28}" y2="{height-18}" '
            f'stroke="{color}" stroke-width="3"/>',
            f'<text x="{legend_x+36}" y="{height-14}" font-family="sans-serif" '
            f'font-size="13" fill="#333">{escape(name)}</text>',
        ))
    path.write_text("\n".join([*lines, "</svg>"]) + "\n", encoding="utf-8")


def main() -> None:
    """Generate reward curves and a machine-readable CSV from one log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only-smoothed", action="store_true")
    args = parser.parse_args()
    records = read_rewards(args.log)
    if not records:
        raise SystemExit("No complete reward metrics found in the log yet")
    if args.only_smoothed:
        outputs = [(args.output or args.log.with_name("reward_curve_10step.svg"), "mean", True)]
    else:
        mean_path = args.output or args.log.with_name("reward_mean.svg")
        max_path = (mean_path.with_name(f"{mean_path.stem}_max{mean_path.suffix}")
                    if args.output else args.log.with_name("reward_max.svg"))
        outputs = [(mean_path, "mean", False), (max_path, "max", False)]
    for output, metric, smoothed in outputs:
        write_plot(records, output, metric=metric, only_smoothed=smoothed)
    csv_path = (outputs[0][0].with_suffix(".csv") if args.only_smoothed or args.output
                else args.log.with_name("reward_curves.csv"))
    mean_smooth = _moving_average([row[1] for row in records])
    max_smooth = _moving_average([row[2] for row in records])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "reward_mean", "reward_max", "reward_std",
                         "reward_mean_10step", "reward_max_10step"))
        writer.writerows((*row, mean_smooth[index], max_smooth[index])
                        for index, row in enumerate(records))
    print(f"Plotted {len(records)} steps: {', '.join(str(path) for path, _, _ in outputs)}; {csv_path}")


if __name__ == "__main__":
    main()
