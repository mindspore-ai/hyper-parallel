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
"""Plan and read back the DeepSeek-V4.1 expert-degree ladder.

The seven points measured on 8 October all sit at expert degree 1, apart from
EP2 shard1 and EP16 shard16.  That is the whole trouble with every V4.1 finding
so far: the cost model prices the expert degree in the wrong direction, and the
ladder that would test a fix has never been run.  Eighteen of the twenty-five
(expert degree, shard width) points at sixteen dies have no measurement at all,
including every point at degree 4 and 8, degree 16 without the optimizer shard,
and degree 2 with one.

This script does not launch anything.  ``repeat_deepseek_v41_runs.sh`` already
takes the points as ``POINTS`` and derives the expert shard the same way ND's
``expert_shard: group`` does, so the round needs no new launcher.  What was
missing is the three things around it.

``points``
    Print the ``POINTS`` string for what has not been measured, and the exact
    command that runs it.  Nothing here is typed by hand on the node.

``summary``
    Read a finished or half-finished round and print what it measured: each
    point's mean step and spread, the ladder by expert degree at each shard
    width, the shard-width order within each degree, and the points that have
    not run.  Standard library only, so it runs under any Python 3 on any node.
    ``--out FILE`` writes the block to send back.

All twenty-five points are legal at sixteen dies, checked against the trainer's
own ``_build_device_mesh_from_accelerator`` under a fake process group with no
device: the dense mesh needs ``dp * cp`` divisible by the shard width and the
expert mesh needs ``world / ep`` divisible by the expert shard, and every power
of two on both axes satisfies both.  So no point in the grid can be refused for
its mesh, and the round needs no pre-flight beyond the one the window already
has.

The two orders it prints are the result, and they are what the cost model is
scored against off the device.  The script ranks nothing against ND: ND is not
installed on the nodes that run this, and a prediction compared on the machine
that produced the measurement is how a reading gets talked into agreeing.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Every (expert degree, FSDP shard width) point at sixteen dies.  Both axes are
# the powers of two that divide the world, which is what the trainer accepts:
# the dense mesh needs ``dp * cp`` divisible by the shard width, and the expert
# mesh needs ``world / ep`` divisible by the expert shard.
DEGREES: Tuple[int, ...] = (1, 2, 4, 8, 16)
WIDTHS: Tuple[int, ...] = (1, 2, 4, 8, 16)

# The round of 8 October, three repeats of each point, no profiler, steps 3 to 6,
# 4096 tokens, global batch 16, micro batch 1, on one 16 die host.  Mean over all
# twelve steps a point, in milliseconds, from that round's steps.csv.  Carried
# here because a node does not have that file, and a new round read beside these
# is the only way to see the ladder whole.
MEASURED_1008: Dict[Tuple[int, int], float] = {
    (2, 1): 7707.3,
    (16, 16): 7990.2,
    (1, 2): 8061.3,
    (1, 1): 8089.2,
    (1, 8): 8184.8,
    (1, 16): 8281.8,
    (1, 4): 8287.1,
}
ROUND_1008 = "8 October round, three repeats a point"

# What one launch costs, measured over the 21 runs of that round: 81 to 85 s
# each, including the model build and the data loader's first touch.
SECONDS_A_RUN = 84


def all_points() -> List[Tuple[int, int]]:
    """Return every (degree, width) point, degree-major."""
    return [(ep, op) for ep in DEGREES for op in WIDTHS]


def unmeasured_points() -> List[Tuple[int, int]]:
    """Return the points no round has measured yet."""
    return [point for point in all_points() if point not in MEASURED_1008]


def points_string(points: Iterable[Tuple[int, int]]) -> str:
    """Format points the way ``repeat_deepseek_v41_runs.sh`` reads ``POINTS``."""
    return " ".join(f"{ep}:{op}" for ep, op in points)


def _duration(seconds: int) -> str:
    """Format a duration the way the repeat script does."""
    hours, minutes = seconds // 3600, (seconds % 3600) // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds % 60:02d}s"


def cmd_points(args: argparse.Namespace) -> int:
    """Print the points to run and the command that runs them."""
    points = all_points() if args.all else unmeasured_points()
    runs = len(points) * args.repeats
    print(f"{len(points)} points, {args.repeats} repeats each, {runs} launches")
    print(f"about {_duration(runs * SECONDS_A_RUN)} of launches at "
          f"{SECONDS_A_RUN} s a run, plus the one-off asset and data preparation")
    if not args.all:
        print(f"{len(MEASURED_1008)} points are already measured ({ROUND_1008}) and are left out")
    print()
    print("POINTS string:")
    print(f'  POINTS="{points_string(points)}"')
    print()
    print("The command, from the root of this checkout:")
    if args.model:
        print(f'  M={args.model}')
    else:
        # A path nobody has read on that machine is a guess, so derive it and
        # fail loudly rather than printing one to be edited by hand.
        print(f"  M=$(find {args.home} -ipath '*deepseek*' -name tokenizer.json 2>/dev/null | head -1)")
        print('  [ -n "$M" ] || echo "NO V4.1 MODEL DIRECTORY FOUND, say so rather than guessing one"')
        print('  M=$(dirname "${M:-.}")')
        # Which directory it found goes in the log: on these nodes everyone logs
        # in as root, so a home directory is not a person and two people's
        # copies can both match.
        print('  echo "model directory: $M"')
    print(f'  POINTS="{points_string(points)}" REPEATS={args.repeats} \\')
    print(f"      RUN_ROOT={args.run_root} \\")
    print('      ./examples/training_demo/deepseek_v41/repeat_deepseek_v41_runs.sh "$M"')
    print()
    print("Then the block to send back:")
    print(f"  python examples/training_demo/deepseek_v41/nd_v41_ladder.py summary {args.run_root} \\")
    print(f"      --out {args.run_root}/ladder_summary.txt")
    print()
    print("The round robins by repeat, so if the window closes early every point")
    print("still has one measurement rather than a third of the points having three.")
    print("Relaunching with the same RUN_ROOT skips what already succeeded.")
    return 0


def _read_steps(run_root: Path) -> Dict[Tuple[int, int], List[float]]:
    """Group a round's per-step times by point.

    Args:
        run_root: The ``RUN_ROOT`` the repeat script wrote.

    Returns:
        Step times in milliseconds, keyed by (degree, width).
    """
    steps_csv = run_root / "steps.csv"
    if not steps_csv.is_file():
        raise SystemExit(f"No steps.csv under {run_root}: is that the RUN_ROOT the round wrote?")
    by_point: Dict[Tuple[int, int], List[float]] = {}
    with steps_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            point = (int(row["ep"]), int(row["op"]))
            by_point.setdefault(point, []).append(float(row["step_time_ms"]))
    return by_point


def _read_runs(run_root: Path) -> List[Dict[str, str]]:
    """Return a round's run rows, or an empty list when it wrote none."""
    runs_csv = run_root / "runs.csv"
    if not runs_csv.is_file():
        return []
    with runs_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _order_lines(means: Dict[Tuple[int, int], float], own: Sequence[Tuple[int, int]]) -> List[str]:
    """Render every point in measured order, fastest first."""
    lines = ["  rank  point        mean ms   vs fastest   source"]
    fastest = min(means.values())
    for rank, (point, mean) in enumerate(sorted(means.items(), key=lambda kv: kv[1]), start=1):
        source = "this round" if point in own else ROUND_1008
        lines.append(f"  {rank:>4}  ep{point[0]:<2} op{point[1]:<2}  "
                     f"{mean:>9.1f}   {(mean / fastest - 1) * 100:>+9.2f}%   {source}")
    return lines


def _ladder_lines(means: Dict[Tuple[int, int], float]) -> List[str]:
    """Render the expert-degree ladder at each shard width, and the reverse."""
    lines = ["  By expert degree, at a fixed shard width (the direction in question):"]
    for op in WIDTHS:
        row = [(ep, means[(ep, op)]) for ep in DEGREES if (ep, op) in means]
        if len(row) < 2:
            continue
        best = min(row, key=lambda pair: pair[1])[0]
        cells = "  ".join(f"ep{ep}={mean:.0f}" for ep, mean in row)
        lines.append(f"    shard {op:<2}: {cells}   fastest degree {best}")
    lines.append("  By shard width, at a fixed expert degree:")
    for ep in DEGREES:
        row = [(op, means[(ep, op)]) for op in WIDTHS if (ep, op) in means]
        if len(row) < 2:
            continue
        best = min(row, key=lambda pair: pair[1])[0]
        cells = "  ".join(f"op{op}={mean:.0f}" for op, mean in row)
        lines.append(f"    degree {ep:<2}: {cells}   fastest width {best}")
    return lines


def _own_round_lines(by_point: Dict[Tuple[int, int], List[float]],
                     runs: Sequence[Dict[str, str]]) -> List[str]:
    """Render this round's own points, with the spread over its repeats."""
    peaks = _peak_by_point(runs)
    lines = ["This round's points, with the spread over its own repeats:",
             "  point        n   mean ms    min      max   spread%   peak alloc GiB"]
    for point in sorted(by_point, key=lambda key: statistics.fmean(by_point[key])):
        times = by_point[point]
        mean = statistics.fmean(times)
        peak = peaks.get(point)
        peak_cell = f"{peak:>12.1f}" if peak is not None else f"{'-':>12}"
        lines.append(f"  ep{point[0]:<2} op{point[1]:<2} {len(times):>4} {mean:>9.1f} "
                     f"{min(times):>8.1f} {max(times):>8.1f} "
                     f"{(max(times) - min(times)) / mean * 100:>8.1f}  {peak_cell}")
    return lines


def _state_lines(own_means: Dict[Tuple[int, int], float],
                 runs: Sequence[Dict[str, str]]) -> List[str]:
    """Say how much of the round has run, and name what failed or is missing."""
    ok = len([row for row in runs if row.get("status") == "ok"])
    lines = [f"{len(own_means)} points measured here, {ok} launches recorded ok"]
    failed = [row["tag"] for row in runs if row.get("status") != "ok"]
    if failed:
        lines.append("FAILED launches, retry by relaunching with the same RUN_ROOT: "
                     + ", ".join(failed))
    missing = [point for point in all_points()
               if point not in own_means and point not in MEASURED_1008]
    if missing:
        lines.append(f"NOT RUN YET, {len(missing)} points: {points_string(missing)}")
    else:
        lines.append("Every point of the grid now has a measurement.")
    return lines


def _agreement_lines(own_means: Dict[Tuple[int, int], float]) -> List[str]:
    """Render the points both rounds measured, which is the check on the nodes."""
    both = sorted(set(MEASURED_1008) & set(own_means))
    if not both:
        return []
    repeated = ", ".join(
        f"ep{ep} op{op}: {MEASURED_1008[(ep, op)]:.1f} then {own_means[(ep, op)]:.1f} ms"
        for ep, op in both)
    return ["", "Points measured in both rounds, which is the check that the nodes agree:",
            f"  {repeated}"]


def _summary_lines(run_root: Path) -> List[str]:
    """Build the whole block a round is reported as."""
    by_point = _read_steps(run_root)
    runs = _read_runs(run_root)
    own_means = {point: statistics.fmean(times) for point, times in by_point.items()}
    merged = dict(MEASURED_1008)
    merged.update(own_means)  # a point measured twice is reported from this round

    lines = [f"DeepSeek-V4.1 expert-degree ladder, round at {run_root}", ""]
    lines.extend(_state_lines(own_means, runs))
    lines.extend(["", *_own_round_lines(by_point, runs)])
    lines.extend(["", f"All {len(merged)} points, this round beside the {ROUND_1008}:"])
    lines.extend(_order_lines(merged, own=list(own_means)))
    lines.extend(_agreement_lines(own_means))
    lines.extend(["", *_ladder_lines(merged)])
    lines.extend(["", "Times compare only within one node and one round; the two rounds are",
                  "reported side by side rather than pooled, and the repeated points say",
                  "how far apart the nodes are."])
    return lines


def _peak_by_point(runs: Sequence[Dict[str, str]]) -> Dict[Tuple[int, int], float]:
    """Return the highest allocated peak recorded for each point, in GiB."""
    peaks: Dict[Tuple[int, int], float] = {}
    for row in runs:
        if row.get("status") != "ok":
            continue
        try:
            point = (int(row["ep"]), int(row["op"]))
            peak = float(row["peak_alloc_gb"])
        except (KeyError, TypeError, ValueError):
            continue
        peaks[point] = max(peaks.get(point, 0.0), peak)
    return peaks


def cmd_summary(args: argparse.Namespace) -> int:
    """Print, and optionally write, the block a round is reported as."""
    text = "\n".join(_summary_lines(Path(args.run_root).resolve())) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nWritten to {Path(args.out).resolve()}, which is the file to send back")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and run the chosen subcommand."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    points = subparsers.add_parser("points", help="the points to run and the command that runs them")
    points.add_argument("--repeats", type=int, default=3, help="repeats of each point, default 3")
    points.add_argument("--all", action="store_true",
                        help="include the points the 8 October round already measured")
    points.add_argument("--model", help="the V4.1 model directory, if it is already known")
    points.add_argument("--home", default="/home",
                        help="where to look for the model directory, default /home; not $HOME, "
                             "which is shared because these nodes are logged into as root")
    points.add_argument("--run-root", dest="run_root", default="output/nd_v41_repeats/ladder",
                        help="where the round writes, and what a relaunch resumes from")
    points.set_defaults(func=cmd_points)

    summary = subparsers.add_parser("summary", help="read a round back and print the orders")
    summary.add_argument("run_root", help="the RUN_ROOT the repeat script wrote")
    summary.add_argument("--out", help="also write the block to this file")
    summary.set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
