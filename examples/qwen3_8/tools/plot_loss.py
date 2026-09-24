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
"""Plot raw and smoothed loss curves from one HyperParallel training log.

Example:
    python examples/qwen3_8/tools/plot_loss.py /path/to/train.log \
        --output /path/to/loss_curve.png --smooth-window 5
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed plotting arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot raw and moving-average loss from a HyperParallel log."
    )
    parser.add_argument("log_file", type=Path, help="Training log to parse.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("loss_curve.png"),
        help="Destination PNG path (default: ./loss_curve.png).",
    )
    parser.add_argument(
        "--metric",
        choices=("total_loss", "foundation_loss"),
        default="total_loss",
        help="Structured training loss field to plot.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Centered moving-average window in recorded steps (default: 5).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Figure title; defaults to the input log filename.",
    )
    return parser.parse_args()


def parse_loss(log_file: Path, metric: str) -> List[Tuple[int, float]]:
    """Extract one structured loss value per optimizer step.

    Args:
        log_file: HyperParallel training log.
        metric: Structured metric suffix, such as ``total_loss``.

    Returns:
        Sorted ``(global_step, loss)`` pairs. If a step is logged more than
        once, the last structured record wins.

    Raises:
        ValueError: If the log is missing, contains no matching values, or a
            matching loss is not finite.
    """
    if not log_file.is_file():
        raise ValueError(f"Training log does not exist: {log_file}")

    pattern = re.compile(
        rf"(?:^|\s)step=(?P<step>\d+)\b.*?"
        rf"training/{re.escape(metric)}=(?P<loss>{_FLOAT_PATTERN})(?:\s|$)"
    )
    losses: Dict[int, float] = {}
    text = log_file.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        step = int(match.group("step"))
        loss = float(match.group("loss"))
        if not math.isfinite(loss):
            raise ValueError(f"Non-finite {metric} at step {step}: {loss}")
        losses[step] = loss

    if not losses:
        raise ValueError(
            f"No 'step=... training/{metric}=...' records found in {log_file}"
        )
    return sorted(losses.items())


def moving_average(values: List[float], window: int) -> List[float]:
    """Compute a centered moving average while retaining edge points.

    Args:
        values: Ordered raw loss values.
        window: Number of neighboring recorded steps in the averaging window.

    Returns:
        Smoothed values with the same length as ``values``.

    Raises:
        ValueError: If ``window`` is not positive.
    """
    if window <= 0:
        raise ValueError("--smooth-window must be greater than zero")

    left = (window - 1) // 2
    right = window // 2
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)

    smoothed = []
    for index in range(len(values)):
        start = max(index - left, 0)
        end = min(index + right + 1, len(values))
        smoothed.append((prefix[end] - prefix[start]) / (end - start))
    return smoothed


def plot_loss(
        records: List[Tuple[int, float]],
        output: Path,
        smooth_window: int,
        title: str,
) -> None:
    """Write raw and smoothed loss curves to one PNG.

    Args:
        records: Sorted ``(step, raw_loss)`` pairs.
        output: Destination PNG path.
        smooth_window: Centered moving-average window.
        title: Figure title.
    """
    steps = [step for step, _ in records]
    raw_losses = [loss for _, loss in records]
    smoothed_losses = moving_average(raw_losses, smooth_window)

    figure = Figure(figsize=(12, 6.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()

    axis.plot(
        steps,
        raw_losses,
        color="#4C78A8",
        linewidth=1.0,
        alpha=0.25,
        label="Raw loss",
    )
    axis.plot(
        steps,
        smoothed_losses,
        color="#F58518",
        linewidth=2.4,
        label=f"Centered moving average (window={smooth_window})",
    )
    axis.set_xlabel("Optimizer Step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend()

    figure.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


def main() -> None:
    """Parse the log and generate the loss figure."""
    args = parse_args()
    log_file = args.log_file.expanduser()
    output = args.output.expanduser()
    records = parse_loss(log_file, args.metric)
    plot_loss(
        records,
        output,
        args.smooth_window,
        args.title or log_file.name,
    )
    print(
        f"Wrote {output} from {len(records)} steps: "
        f"first={records[0][1]:.8g}, last={records[-1][1]:.8g}"
    )


if __name__ == "__main__":
    main()
