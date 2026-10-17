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
"""Read timed rounds back: each layout's measured step, how ND ordered them, and the layout to launch.

    python <this skill's folder>/scripts/readback.py <run folder> [<round folder> ...]

<run folder> is the one ``ask.py`` kept with ND's ranking of the request; each <round folder> holds the
``runs.csv`` (and ``steps.csv``) the template's timer wrote, oldest first. The rounds this skill keeps for the
template on the cluster come first, unless ``--only-given``. A later round is set beside the earlier ones only
when every layout both timed has runs in it that overlap its earlier range; otherwise reading starts again from
that round, since the node or the code moved under it.

The pick is the fastest layout measured, never ND's: ND's order is scored here, not followed. It is firm when its
slowest run beats every other layout's fastest run, so that no choice of statistic changes it. Everything lands in
``<run>/readback/``: ``readback.md``, ``measured.csv`` and ``launch.yaml``, the trainer yaml of the pick.
"""
import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

import answer
import ask
import timing

Round = Tuple[Path, List[timing.Run]]
# Below three layouts a rank correlation can only be +1 or -1, which says nothing about an order.
MIN_ORDERED = 3


@dataclass
class Measured:
    """One layout's launches across the rounds read, and ND's row for it.

    Attributes:
        key: the layout, as its timer names it.
        runs: the launches that finished.
        failed: the launches that did not.
        nd_row: ND's row for the layout, None when ND kept none.
        nd_place: that row's place in ND's ranking, layouts counted once.
    """

    key: timing.Key
    runs: List[timing.Run] = field(default_factory=list)
    failed: List[timing.Run] = field(default_factory=list)
    nd_row: Optional[Dict[str, str]] = None
    nd_place: Optional[int] = None

    @property
    def mean(self) -> float:
        """The mean step over the finished launches, in ms."""
        return statistics.fmean(run.mean_ms for run in self.runs)

    @property
    def fastest(self) -> float:
        """The fastest launch's mean step, in ms."""
        return min(run.mean_ms for run in self.runs)

    @property
    def slowest(self) -> float:
        """The slowest launch's mean step, in ms."""
        return max(run.mean_ms for run in self.runs)

    @property
    def spread_pct(self) -> float:
        """How far apart the launches ran, in percent of the mean."""
        return 100.0 * (self.slowest - self.fastest) / self.mean

    @property
    def peak_gib(self) -> float:
        """The mean over the launches of the largest memory a die allocated, in GiB."""
        return statistics.fmean(run.peak_gib for run in self.runs)


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """The run folder and the rounds to read, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="the run folder ask.py kept")
    parser.add_argument("rounds", nargs="*", help="round folders the timer wrote (its RUN_ROOT), oldest first")
    parser.add_argument("--only-given", dest="only_given", action="store_true",
                        help="read only the rounds given, not those kept with this skill")
    parser.add_argument("--family", default=None, help="the template, when the run folder has no ask.json")
    parser.add_argument("--cluster", default=None, help="the cluster, when the run folder has no ask.json")
    return parser.parse_args(argv)


def _ranges(runs: Sequence[timing.Run]) -> Dict[timing.Key, Tuple[float, float]]:
    """Each layout's fastest and slowest finished launch."""
    ranges: Dict[timing.Key, Tuple[float, float]] = {}
    for run in runs:
        if run.ok:
            low, high = ranges.get(run.key, (run.mean_ms, run.mean_ms))
            ranges[run.key] = (min(low, run.mean_ms), max(high, run.mean_ms))
    return ranges


def pooled(rounds: Sequence[Round], timer: Dict) -> Tuple[List[timing.Run], List[str]]:
    """The launches read together, and how each round was set beside the ones before it.

    Args:
        rounds: each round's folder and launches, oldest first.
        timer: the template's timer record, to name the layouts.

    Returns:
        The launches of the rounds read together, and one line a round saying how it joined.
    """
    kept: List[timing.Run] = list(rounds[0][1])
    notes = [f"{rounds[0][0].as_posix()}: {sum(run.ok for run in rounds[0][1])} launches finished."]
    for folder, runs in rounds[1:]:
        earlier, later = _ranges(kept), _ranges(runs)
        shared = [key for key in later if key in earlier]
        moved = [key for key in shared if later[key][0] > earlier[key][1] or later[key][1] < earlier[key][0]]
        head = f"{folder.as_posix()}: {sum(run.ok for run in runs)} launches finished"
        if not shared:
            notes.append(f"{head}. It times no layout the rounds before it timed, so it cannot be set beside them: "
                         "it is read alone.")
            kept = list(runs)
        elif moved:
            words = "; ".join(f"{timing.name_of(key, timer)} ran {later[key][0]:.1f} to {later[key][1]:.1f} ms "
                              f"against {earlier[key][0]:.1f} to {earlier[key][1]:.1f} before" for key in moved)
            notes.append(f"{head}. The node or the code moved under it ({words}), so it is read alone.")
            kept = list(runs)
        else:
            names = ", ".join(timing.name_of(key, timer) for key in shared)
            notes.append(f"{head}, set beside the rounds before it through {names}, whose runs overlap there.")
            kept += runs
    return kept, notes


def measure(runs: Sequence[timing.Run], rows: Sequence[Dict[str, str]], pool: Sequence[Dict[str, str]],
            timer: Dict) -> List[Measured]:
    """Each layout the launches timed, with ND's row for it, in the order the rounds first timed them."""
    items: Dict[timing.Key, Measured] = {}
    for run in runs:
        item = items.setdefault(run.key, Measured(run.key))
        (item.runs if run.ok else item.failed).append(run)
    places = {id(row): place for place, row in enumerate(rows, start=1)}
    for row in pool:
        item = items.get(timing.key_of(row, timer))
        if item is not None and item.nd_row is None:
            item.nd_row, item.nd_place = row, places[id(row)]
    return list(items.values())


def _runs(count: int) -> str:
    """``1 run``, ``3 runs``."""
    return f"{count} run" if count == 1 else f"{count} runs"


def _ranks(values: Sequence[float]) -> List[int]:
    """Each value's rank from 1, ties broken by position."""
    result = [0] * len(values)
    for place, index in enumerate(sorted(range(len(values)), key=values.__getitem__), start=1):
        result[index] = place
    return result


def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman's rank correlation of two sequences of the same length, at least two long."""
    count = len(first)
    shifts = sum((one - two) ** 2 for one, two in zip(_ranks(first), _ranks(second)))
    return 1.0 - 6.0 * shifts / (count * (count * count - 1))


def verdict(best: Measured, others: Sequence[Measured], timer: Dict) -> str:
    """Whether one more round could change the pick: firm when its slowest run beats every other's fastest."""
    if len(best.runs) < 2:
        return "It ran once: time it again before launching on it."
    if not others:
        return "It is the only layout measured."
    rival = min(others, key=lambda item: item.fastest)
    margin = rival.fastest - best.slowest
    if margin > 0:
        return (f"Among the layouts timed, its slowest run beats every other's fastest run, by {margin:.1f} ms "
                f"({timing.name_of(rival.key, timer)}): no choice of statistic changes the pick among them.")
    close = ", ".join(timing.name_of(item.key, timer) for item in others if item.fastest <= best.slowest)
    return f"Not separated: its runs overlap those of {close}. One more round of these decides."


def _nd_lines(items: Sequence[Measured], best: Measured, rows: Sequence[Dict[str, str]], timer: Dict,
              degrees: Sequence[str]) -> List[str]:
    """How ND fared against the measurements: its order, what following it costs, its memory."""
    priced = [item for item in items if item.nd_row is not None]
    lines = []
    if len(priced) >= MIN_ORDERED:
        order = spearman([float(item.nd_row["score"]) for item in priced], [item.mean for item in priced])
        lines.append(f"ND's order over these {len(priced)} layouts: {order:+.2f}, its rank correlation with the "
                     "measured order (+1 is the same order, -1 the reverse).")
    if priced:
        choice = min(priced, key=lambda item: item.nd_place)
        cost = 100.0 * (choice.mean / best.mean - 1.0)
        lines.append(f"Following ND: its best of them, {timing.name_of(choice.key, timer)} ({choice.nd_place} of "
                     f"{len(rows)}), measured {choice.mean:.1f} ms, {cost:+.2f}% against the pick.")
        errors = [100.0 * (answer.peak_gib(item.nd_row) / item.peak_gib - 1.0) for item in priced]
        lines.append(f"ND's memory came within {min(errors):+.1f}% to {max(errors):+.1f}% of each layout's "
                     "measured peak allocated.")
    if rows and not any(item.nd_row is rows[0] for item in priced):
        lines.append(f"ND's own pick, {answer.layout(rows[0], degrees)}, is not among the layouts timed.")
    unpriced = [timing.name_of(item.key, timer) for item in items if item.nd_row is None]
    if unpriced:
        lines.append(f"Measured, but ND kept no row for them (over its budget, or outside its search): "
                     f"{', '.join(unpriced)}.")
    return lines


def _planned(run: Path, timer: Dict) -> List[timing.Key]:
    """The layouts the run's launch plan named, in its order; none without a plan."""
    path = run / "launch" / "points.txt"
    if not path.is_file():
        return []
    return [tuple(point.split(":")) for point in path.read_text(encoding="utf-8").split()
            if len(point.split(":")) == len(timer["varies"])]


def _table(items: Sequence[Measured], rows: Sequence[Dict[str, str]], timer: Dict) -> List[str]:
    """One line a layout, fastest first."""
    lines = ["| Place | Layout | Step (ms) | Runs | Spread | Peak allocated (GiB) | ND's place | ND's peak (GiB) |",
             "|---|---|---|---|---|---|---|---|"]
    for place, item in enumerate(sorted(items, key=lambda one: one.mean), start=1):
        nd_place = f"{item.nd_place} of {len(rows)}" if item.nd_row is not None else "no row"
        nd_peak = f"{answer.peak_gib(item.nd_row):.1f}" if item.nd_row is not None else ""
        lines.append(f"| {place} | {timing.name_of(item.key, timer)} | {item.mean:.1f} | {len(item.runs)} "
                     f"| {item.spread_pct:.2f}% | {item.peak_gib:.1f} | {nd_place} | {nd_peak} |")
    return lines


def write_csv(path: Path, items: Sequence[Measured], timer: Dict) -> None:
    """``measured.csv``: one row a layout, fastest first, with ND's place, score and peak beside it."""
    names = [name.lower() for name in timer["varies"]]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["place", *names, "runs", "failed", "mean_ms", "fastest_run_ms", "slowest_run_ms",
                         "spread_pct", "peak_alloc_gib", "nd_place", "nd_score", "nd_peak_gib"])
        for place, item in enumerate(sorted(items, key=lambda one: one.mean), start=1):
            nd = item.nd_row or {}
            writer.writerow([place, *item.key, len(item.runs), len(item.failed), f"{item.mean:.1f}",
                             f"{item.fastest:.1f}", f"{item.slowest:.1f}", f"{item.spread_pct:.2f}",
                             f"{item.peak_gib:.3f}", item.nd_place or "", nd.get("score", ""),
                             f"{answer.peak_gib(nd):.3f}" if nd else ""])


@dataclass
class Source:
    """What a read-back reads: the run's records, its yaml and ranking, and the rounds' folders.

    Attributes:
        family: the template the run was priced for.
        cluster: the cluster it was priced for.
        timer: the template's timer record on the cluster.
        trust: the template's trust record on the cluster.
        doc: the trainer yaml ND priced.
        ranking: ND's ranking of the run.
        folders: the rounds to read, oldest first.
    """

    family: str
    cluster: str
    timer: Dict
    trust: Dict
    doc: Dict
    ranking: answer.Ranking
    folders: List[Path]


def source_of(run: Path, args: argparse.Namespace) -> Source:
    """The records, ranking and rounds a read-back of the run reads.

    Raises:
        SystemExit: the template has no timer record on the cluster, or there is no round to read.
    """
    family, cluster = timing.asked(run, args.family, args.cluster)
    timer = timing.timer_entry(family, cluster)
    if timer is None:
        raise SystemExit(f"no timer is recorded for {family} on {cluster}, so its rounds cannot be read")
    folders = ([] if args.only_given else timing.recorded_rounds(timer)) + [Path(folder) for folder in args.rounds]
    if not folders:
        raise SystemExit("no round to read: give the folder the timer wrote")
    return Source(family=family, cluster=cluster, timer=timer, trust=answer.trust_entry(family, cluster),
                  doc=yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8")),
                  ranking=answer.Ranking(str(run / "nd_rank.csv")), folders=folders)


def _head_lines(source: Source, finished: Sequence[Measured], best: Measured, notes: Sequence[str],
                untimed: int) -> List[str]:
    """The record's status, what was read, the pick and whether one more round could change it.

    *untimed* is how many layouts the run's launch plan names that no round read has timed yet.
    """
    timer = source.timer
    lines = [f"ND's record for {source.family} on {source.cluster}: {source.trust['status']}. The pick below is "
             "measured, not ND's.", "",
             f"Read back: {len(finished)} layouts, {sum(len(item.runs) for item in finished)} finished launches.",
             *notes, "",
             f"The pick, measured: {timing.name_of(best.key, timer)}, {best.mean:.1f} ms a step over "
             f"{_runs(len(best.runs))} ({best.fastest:.1f} to {best.slowest:.1f}).",
             verdict(best, [item for item in finished if item is not best], timer)]
    if untimed:
        lines.append(f"The launch plan still has {untimed} layouts no round has timed, and any of them could be "
                     "faster: the pick holds among the layouts timed.")
    lines.append("")
    if best.nd_row is not None:
        fixed = answer.layout(best.nd_row, [name for name in source.ranking.degrees if name not in timer["varies"]])
        lines += [f"Every layout here is {fixed}.", ""]
    return lines


def _closing_lines(run: Path, source: Source, items: Sequence[Measured], best: Measured) -> List[str]:
    """Failed launches, what the launch plan still has to time, and the trainer yaml of the pick."""
    timer = source.timer
    failed = [f"{timing.name_of(item.key, timer)} {len(item.failed)} ({', '.join(one.status for one in item.failed)})"
              for item in items if item.failed]
    lines = [f"Failed launches: {'; '.join(failed)}." if failed else "Failed launches: none."]
    measured = {item.key for item in items if item.runs}
    missing = [timing.name_of(key, timer) for key in _planned(run, timer) if key not in measured]
    if missing:
        lines.append(f"Still to time from the launch plan: {len(missing)} layouts: {', '.join(missing)}.")
    out = run / "readback"
    if best.nd_row is not None:
        text = yaml.safe_dump(ask.launch_yaml(source.doc, best.nd_row), sort_keys=False, allow_unicode=True)
        (out / "launch.yaml").write_text(text, encoding="utf-8")
        lines.append(f"Trainer yaml for the pick: {out / 'launch.yaml'}; set its tokenizer and data paths on the "
                     "cluster.")
    lines.append(f"Read back in {out}")
    return lines


def report(run: Path, args: argparse.Namespace) -> List[str]:
    """Read the rounds back against the run's ranking, write ``<run>/readback``, and return its lines.

    Raises:
        SystemExit: nothing to read, or no launch of the rounds read finished.
    """
    source = source_of(run, args)
    rows = source.ranking.distinct()
    rounds = [(folder, timing.read_round(folder, source.timer["varies"])) for folder in source.folders]
    runs, notes = pooled(rounds, source.timer)
    items = measure(runs, rows, timing.launchable(rows, source.timer, source.doc), source.timer)
    finished = [item for item in items if item.runs]
    if not finished:
        raise SystemExit("no launch of the rounds read finished: nothing to pick from")
    best = min(finished, key=lambda item: item.mean)
    out = run / "readback"
    out.mkdir(exist_ok=True)
    write_csv(out / "measured.csv", finished, source.timer)
    measured = {item.key for item in finished}
    untimed = sum(1 for key in _planned(run, source.timer) if key not in measured)
    lines = _head_lines(source, finished, best, notes, untimed) + _table(finished, rows, source.timer) + [""]
    lines += _nd_lines(finished, best, rows, source.timer, source.ranking.degrees)
    lines += _closing_lines(run, source, items, best)
    (out / "readback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read the rounds back, keep the result in the run folder, print it."""
    args = _arguments(argv)
    sys.stdout.write("\n".join(report(Path(args.run), args)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
