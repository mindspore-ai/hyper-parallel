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
"""Plan the round that times the layouts worth timing: a trainer yaml each, and one command that times them all.

    python <this skill's folder>/scripts/launch.py <run folder>

<run folder> is the one ``ask.py`` kept with ND's ranking of the request. Which layouts are worth timing follows
the trust record, never a choice made by hand: where ND's order was checked against timed runs, the layout to
launch, those one round cannot tell from it, and ND's own pick when it lies outside the checked region; where it
was never checked, every layout that fits, that the template's timer launches and that no round has timed yet,
since ND's order cannot choose among them. ``--layouts`` names them instead. A layout an earlier round timed is
timed again first, so that the new round can be set beside the earlier ones.

Everything lands in ``<run>/launch/``: the trainer yaml of each layout as its timed run reads it, ``points.txt``,
and ``launch.md``, the plan with the command, which this script also prints.
"""
import argparse
import copy
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import yaml

import answer
import ask
import timing

# A layout chosen for the round, with the reason the plan gives for it.
Chosen = List[Tuple[Dict[str, str], str]]
NEVER_TIMED = "never timed"


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """The run folder and the choices the plan leaves open, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="the run folder ask.py kept")
    parser.add_argument("--family", default=None, help="the template, when the run folder has no ask.json")
    parser.add_argument("--cluster", default=None, help="the cluster, when the run folder has no ask.json")
    parser.add_argument("--layouts", default=None,
                        help='the layouts to time instead of the rule\'s, for example "EP=2,OP=4;EP=4,OP=1"')
    parser.add_argument("--repeats", type=int, default=None, help="runs a layout, the timer record's by default")
    return parser.parse_args(argv)


def _key_function(timer: Optional[Dict[str, Any]], degrees: Sequence[str]) -> Callable[[Dict[str, str]], tuple]:
    """How a row of ND's ranking is named: by the timer's varying degrees, or by every degree without a timer."""
    names = list(timer["varies"]) if timer else list(degrees)

    def key(row: Dict[str, str]) -> tuple:
        """The row's values of those degrees."""
        return tuple(row[name] for name in names)

    return key


def worth_timing(pool: Sequence[Dict[str, str]], rows: Sequence[Dict[str, str]], trust: Dict[str, Any],
                 spread: float, timed: set, key: Callable[[Dict[str, str]], tuple]) -> Chosen:
    """The layouts worth timing among those the timer launches, each with its reason, by the trust record's rule.

    Args:
        pool: ND's layouts the timer can launch, in ND's order.
        rows: every layout ND kept, in its order.
        trust: the template's trust record on the cluster.
        spread: the cluster's re-measure spread in percent: layouts closer than it by ND are one round's tie.
        timed: the layouts earlier rounds timed.
        key: how a row is named, to match it with the timed ones.

    Returns:
        The layouts not timed yet, in ND's order, each with why it is worth a run.
    """
    region = trust.get("region")
    if region is None:
        return [(row, NEVER_TIMED) for row in pool if key(row) not in timed]
    target = answer.recommended(rows, trust)
    chosen = [(target, "the layout to launch")]
    chosen += [(row, f"within {spread}% of it by ND") for row in pool if row is not target
               and answer.inside(row, region) and answer.versus(row, target) < spread]
    if rows[0] is not target:
        chosen.append((rows[0], "ND's own pick, outside the checked region"))
    members = {id(row) for row in pool}
    return [(row, why) for row, why in chosen if id(row) in members and key(row) not in timed]


def anchors(pool: Sequence[Dict[str, str]], timed_runs: Sequence[timing.Run],
            key: Callable[[Dict[str, str]], tuple]) -> Chosen:
    """The layouts timed before that a new round times again first: ND's best of them, and the fastest measured.

    Through them the new round is set beside the earlier ones; and when it cannot be, because the node or the
    code moved, the new round alone still holds both, so ND's choice and the measured winner are compared again.
    """
    means: Dict[tuple, List[float]] = {}
    for one in timed_runs:
        means.setdefault(one.key, []).append(one.mean_ms)
    candidates = [row for row in pool if key(row) in means]
    if not candidates:
        return []
    fastest = min(candidates, key=lambda row: statistics.fmean(means[key(row)]))
    chosen = [(candidates[0], "timed before, ND's best of those; timed again first")]
    if fastest is not candidates[0]:
        chosen.append((fastest, "timed before, the fastest measured; timed again first"))
    return chosen


def named(pool: Sequence[Dict[str, str]], specs: str) -> Chosen:
    """The layouts ``--layouts`` names, each the first of ND's launchable layouts that states its degrees.

    Raises:
        SystemExit: a named layout is not among them, so it does not fit or the timer cannot launch it.
    """
    chosen = []
    for spec in [part for part in specs.split(";") if part.strip()]:
        pairs = answer.wanted(spec)
        found = next((row for row in pool if all(row.get(name) == value for name, value in pairs)), None)
        if found is None:
            raise SystemExit(f"{spec.strip()}: no layout that fits and that the timer launches states it")
        chosen.append((found, "named in the request"))
    return chosen


def _drop(doc: Dict[str, Any], dotted: str) -> None:
    """Delete ``doc[a][b]...`` for the field ``a.b...``, when the yaml states it."""
    *parents, last = dotted.split(".")
    for name in parents:
        doc = doc.get(name)
        if not isinstance(doc, dict):
            return
    doc.pop(last, None)


def _lookup(doc: Dict[str, Any], dotted: str) -> Any:
    """The value of the field ``a.b...``, None when the yaml does not state it."""
    for name in dotted.split("."):
        if not isinstance(doc, dict) or name not in doc:
            return None
        doc = doc[name]
    return doc


def _differences(priced: Any, launched: Any, where: str) -> Iterator[str]:
    """Each field where the two yamls disagree, as ``field: ND priced X, the launcher states Y``."""
    if isinstance(priced, dict) and isinstance(launched, dict):
        for name in sorted(set(priced) | set(launched), key=str):
            field = f"{where}.{name}" if where else str(name)
            if name not in launched:
                yield f"{field}: ND priced {priced[name]!r}, the launcher does not state it"
            elif name not in priced:
                yield f"{field}: the launcher states {launched[name]!r}, which ND did not price"
            else:
                yield from _differences(priced[name], launched[name], field)
    elif priced != launched:
        yield f"{where}: ND priced {priced!r}, the launcher states {launched!r}"


def launcher_differences(doc: Dict[str, Any], timer: Dict[str, Any], launcher: Optional[Dict[str, Any]]) -> List[str]:
    """Where the yaml ND priced and the launcher the timer reads disagree, beyond what the timer sets for each run.

    A difference means the round would time another job than the one priced, so no command is written.
    """
    if launcher is None:
        return [f"this checkout has no {timer['launcher']}, the launcher the timer reads"]
    priced, launched = copy.deepcopy(doc), copy.deepcopy(launcher)
    priced.pop("context", None)
    for dotted in list(timer.get("sets", [])) + list(timer.get("node_paths", [])):
        _drop(priced, dotted)
        _drop(launched, dotted)
    return list(_differences(priced, launched, ""))


def missing_scripts(timer: Dict[str, Any]) -> List[str]:
    """The timer's own scripts that this checkout does not have.

    The command names them by path, so a checkout without them would hand over a command that
    fails on the node. Say so here instead, as with a launcher that states another job.
    """
    root = ask.repo_root()
    return [f"this checkout has no {timer[field]}, the {field} the round runs"
            for field in ("script", "preflight")
            if timer.get(field) and not (root / timer[field]).is_file()]


def timed_yaml(doc: Dict[str, Any], row: Dict[str, str], timer: Optional[Dict[str, Any]],
               launcher: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The trainer yaml one timed run reads: the run's own with the layout's degrees, as ``ask.py`` writes it.

    With a timer, also its step count, the profiler off, and the node's files where the launcher names them,
    which the timer replaces with the node's own on each launch.
    """
    out = ask.launch_yaml(doc, row)
    if timer is None:
        return out
    out.setdefault("training", {})["train_iters"] = timer["train_iters"]
    out.setdefault("profiling", {})["enabled"] = False
    for dotted in timer.get("node_paths", []):
        value = _lookup(launcher or {}, dotted)
        if value is not None:
            *parents, last = dotted.split(".")
            target = out
            for name in parents:
                target = target.setdefault(name, {})
            target[last] = value
    return out


def _tag(row: Dict[str, str], timer: Optional[Dict[str, Any]], degrees: Sequence[str]) -> str:
    """A file name for a layout: ``ep2_op1`` as the timer tags its runs, or every degree without a timer."""
    if timer:
        return "_".join(f"{name.lower()}{row[name]}" for name in timer["varies"])
    return answer.layout(row, degrees).replace(" ", "_")


def _duration(seconds: float) -> str:
    """A duration as ``1h20m`` or ``45m``."""
    minutes = int(round(seconds / 60.0))
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def command_lines(timer: Dict[str, Any], points: str, repeats: int, root: str, doc: Dict[str, Any]) -> List[str]:
    """The lines to paste on a node, from the root of a checkout of this branch: find the model, check the node,
    then time the round detached from the shell, so that a dropped connection does not end it."""
    training = doc.get("training") or {}
    log = f"{root}.log"
    lines = [
        # A path nobody has read on that node is a guess, so derive it, show it, and let the timer refuse a wrong
        # one. Homes there can be links to another disk, hence -L.
        f"M=$(find -L /home -maxdepth 5 -ipath '{timer.get('model_glob', '*')}' -name tokenizer.json 2>/dev/null "
        "| head -1)",
        '[ -n "$M" ] || echo "NO MODEL DIRECTORY FOUND under /home: set M to it by hand"',
        'M=$(dirname "${M:-.}"); echo "model directory: $M"',
    ]
    # Through bash rather than ./, since an archive can lose a script's executable bit.
    if timer.get("preflight"):
        lines.append(f'bash {timer["preflight"]} "$M"')
    lines += [
        f"mkdir -p {PurePosixPath(root).parent}",
        f'nohup env POINTS="{points}" REPEATS={repeats} GLOBAL_BATCH={training.get("global_batch_size")} '
        f'MICRO_BATCH={training.get("micro_batch_size")} RUN_ROOT={root} \\',
        f'    bash {timer["script"]} "$M" > {log} 2>&1 &',
        f"tail -f {log}",
    ]
    return lines


def _points(rows: Sequence[Dict[str, str]], timer: Dict[str, Any]) -> str:
    """The layouts as the timer's POINTS names them, for example ``1:1 2:4``."""
    return " ".join(":".join(row[name] for name in timer["varies"]) for row in rows)


def _table(chosen: Chosen, rows: Sequence[Dict[str, str]], files: Dict[int, Path], degrees: Sequence[str],
           timer: Optional[Dict[str, Any]]) -> List[str]:
    """One line a layout of the round: why it is timed, ND's place and peak, and its trainer yaml.

    With a timer, the degrees it fixes are said once above the table and each line names the ones it varies.
    """
    places = {id(row): place for place, row in enumerate(rows, start=1)}
    lines = []
    if timer:
        fixed = answer.layout(chosen[0][0], [name for name in degrees if name not in timer["varies"]])
        lines += [f"Every layout of the round is {fixed}; they differ in {' and '.join(timer['varies'])}.", ""]
    lines += ["| Layout | Why | ND's place | ND's peak per die (GiB) | Trainer yaml |", "|---|---|---|---|---|"]
    for row, why in chosen:
        name = timing.name_of(timing.key_of(row, timer), timer) if timer else answer.layout(row, degrees)
        lines.append(f"| {name} | {why} | {places[id(row)]} of {len(rows)} | {answer.peak_gib(row):.1f} "
                     f"| {files[id(row)].name} |")
    return lines


def _left_out(pool: Sequence[Dict[str, str]], rows: Sequence[Dict[str, str]], round_rows: Chosen,
              timed: set, key: Callable[[Dict[str, str]], tuple]) -> List[str]:
    """What the round leaves out, and why: layouts timed before, and layouts the timer cannot launch."""
    taken = {id(row) for row, _ in round_rows}
    again = sum(1 for row in pool if id(row) not in taken and key(row) in timed)
    lines = []
    if again:
        lines.append(f"Left out, timed before: {again} layouts.")
    other = len(rows) - len(pool)
    if other:
        lines.append(f"Left out, not launchable by this template's timer: {other} of ND's {len(rows)} layouts "
                     "that fit (other degrees than those it times).")
    return lines


@dataclass
class Records:
    """The records that decide a round for a template on a cluster.

    Attributes:
        family: the template.
        cluster_name: the cluster's key in calibration/clusters.json.
        trust: how far ND's order was checked there (references/trust.json).
        cluster: the cluster's own record, its re-measure spread among it.
        timer: how its layouts are timed there (references/timers.json), None when nothing times them.
    """

    family: str
    cluster_name: str
    trust: Dict[str, Any]
    cluster: Dict[str, Any]
    timer: Optional[Dict[str, Any]]

    @classmethod
    def of(cls, run: Path, args: argparse.Namespace) -> "Records":
        """The records of the template and cluster the run was priced for."""
        family, cluster_name = timing.asked(run, args.family, args.cluster)
        trust = answer.trust_entry(family, cluster_name)
        return cls(family=family, cluster_name=cluster_name, trust=trust,
                   cluster=ask.cluster_record(cluster_name, trust, None),
                   timer=timing.timer_entry(family, cluster_name))


class Plan:
    """The round a run's ranking asks for: what to time, why, and how."""

    def __init__(self, run: Path, args: argparse.Namespace) -> None:
        """Read the run folder and the records that decide the round.

        Args:
            run: the run folder ask.py kept.
            args: the command line's choices.

        Raises:
            SystemExit: the run holds no layout, or nothing decides which layouts are worth timing.
        """
        self.run = run
        self.records = Records.of(run, args)
        timer = self.records.timer
        self.doc = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
        self.ranking = answer.Ranking(str(run / "nd_rank.csv"))
        self.rows = self.ranking.distinct()
        if not self.rows:
            raise SystemExit("ND kept no layout for this run: none fits its budget, so there is nothing to time")
        self.key = _key_function(timer, self.ranking.degrees)
        self.pool = timing.launchable(self.rows, timer, self.doc) if timer else list(self.rows)
        rounds = timing.recorded_rounds(timer) if timer else []
        self.timed_runs = [one for folder in rounds for one in timing.read_round(folder, timer["varies"]) if one.ok]
        if args.layouts:
            self.chosen = named(self.pool, args.layouts)
        elif self.records.trust.get("region") is None and not timer:
            raise SystemExit(f"ND's order of {self.records.family} on {self.records.cluster_name} was never checked "
                             "against timed runs and no timer is recorded for it there: name the layouts to time "
                             "with --layouts")
        else:
            spread = float(self.records.cluster.get("remeasure_spread_pct") or 0.0)
            self.chosen = worth_timing(self.pool, self.rows, self.records.trust, spread, self.timed, self.key)
        self.anchors = anchors(self.pool, self.timed_runs, self.key) if self.chosen else []
        self.repeats = args.repeats or (timer or {}).get("repeats", 1)
        path = ask.repo_root() / timer["launcher"] if timer else None
        self.launcher = (yaml.safe_load(path.read_text(encoding="utf-8")) if path is not None and path.is_file()
                         else None)

    @property
    def timed(self) -> set:
        """The layouts earlier rounds timed."""
        return {one.key for one in self.timed_runs}

    @property
    def timer(self) -> Optional[Dict[str, Any]]:
        """How the template's layouts are timed on the cluster, None when nothing times them."""
        return self.records.timer

    def round_rows(self) -> Chosen:
        """The layouts the round launches, those timed before first, so that a window closing early has them."""
        return list(self.anchors) + list(self.chosen)

    def write(self) -> List[str]:
        """Write the trainer yamls, ``points.txt`` and ``launch.md`` into ``<run>/launch``: the plan's lines."""
        folder = self.run / "launch"
        folder.mkdir(exist_ok=True)
        for old in folder.glob("*.yaml"):  # a plan written again replaces its own yamls, never keeps stale ones
            old.unlink()
        round_rows = self.round_rows()
        files = {}
        for row, _ in round_rows:
            path = folder / f"{_tag(row, self.timer, self.ranking.degrees)}.yaml"
            text = yaml.safe_dump(timed_yaml(self.doc, row, self.timer, self.launcher), sort_keys=False,
                                  allow_unicode=True)
            path.write_text(text, encoding="utf-8")
            files[id(row)] = path
        lines = self._lines(round_rows, files, folder)
        (folder / "launch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines

    def _rule(self) -> str:
        """Why these layouts, in one sentence."""
        if self.records.trust.get("region") is None:
            return ("ND's order of this template was never checked against timed runs here, so every layout that "
                    "fits, that the timer launches and that no round has timed is worth a run.")
        return ("ND's order was checked inside a region: the layout to launch and those one round cannot tell from "
                "it are worth a run, with ND's own pick when it lies outside.")

    def _lines(self, round_rows: Chosen, files: Dict[int, Path], folder: Path) -> List[str]:
        """The plan in words: the round, why, what it leaves out, the time it takes and the command."""
        trust = self.records.trust
        lines = [f"Status: {trust['status']}. {trust['may_say']}", ""]
        if not self.chosen:
            return lines + ["Nothing is left worth timing: every layout the rule names was timed. Read the rounds "
                            f"back with readback.py {self.run}."]
        again = f", and {len(self.anchors)} timed before, timed again first" if self.anchors else ""
        lines += [f"Layouts worth timing: {len(self.chosen)}{again}. {self._rule()}", ""]
        lines += _table(round_rows, self.rows, files, self.ranking.degrees, self.timer)
        lines += [""] + _left_out(self.pool, self.rows, round_rows, self.timed, self.key)
        if self.anchors:
            words = []
            for row, _ in self.anchors:
                earlier = [one.mean_ms for one in self.timed_runs if one.key == self.key(row)]
                words.append(f"{timing.name_of(self.key(row), self.timer)} {min(earlier):.1f} to {max(earlier):.1f} "
                             f"ms over {len(earlier)} runs")
            lines.append(f"Timed again first, as earlier rounds measured them: {'; '.join(words)}. readback.py sets "
                         "the new round beside the earlier ones only where its runs of these overlap those ranges, "
                         "and otherwise reads it alone, with both still in it.")
        lines += ["", f"Each trainer yaml in {folder} is what its timed run reads."]
        if self.timer is None:
            return lines + [f"No timer is recorded for {self.records.family} on {self.records.cluster_name}: "
                            "launch the yamls with the cluster's own tools, one run each at least, and keep each "
                            "run's step times."]
        return lines + self._command_lines(round_rows, folder)

    def _command_lines(self, round_rows: Chosen, folder: Path) -> List[str]:
        """How long the round takes, the command that times it, and what to send back."""
        problems = missing_scripts(self.timer) + launcher_differences(self.doc, self.timer, self.launcher)
        dies = (self.doc.get("context") or {}).get("device_num")
        if dies is not None and int(dies) != int(self.timer["dies"]):
            problems.append(f"the run is priced on {dies} dies and the timer launches {self.timer['dies']}")
        if problems:
            return ["", "No command: this checkout cannot time the job ND priced, because"] + [
                f"- {problem}" for problem in problems]
        points = _points([row for row, _ in round_rows], self.timer)
        (folder / "points.txt").write_text(points + "\n", encoding="utf-8")
        launches = len(round_rows) * self.repeats
        root = f"output/sapp-parallel-strategy/rounds/{self.run.name}"
        return [
            f"The round: {len(round_rows)} layouts, {self.repeats} runs each, {launches} launches, about "
            f"{_duration(launches * self.timer['seconds_a_run'])} at the {self.timer['seconds_a_run']} s a launch "
            "earlier rounds measured, plus a one-off preparation of the model's assets and data. It goes round "
            "the layouts once before it repeats any, so a window that closes early still times each one once. "
            "It runs detached from the shell, so a dropped connection does not end it, and tail only follows its "
            "log; if it stops, run the nohup line again: with the same RUN_ROOT it skips what already succeeded.",
            "",
            f"On one node of {self.timer['dies']} dies, from the root of a checkout of this branch, with the "
            "model's files on the node and PYTHON_BIN set to the Python that has the checkout's dependencies when "
            "it is not the first one on PATH. The model directory it prints must be the one meant, and the "
            "preflight must end with no FAIL:",
            "",
            "```bash",
            *command_lines(self.timer, points, self.repeats, root, self.doc),
            "```",
            "",
            f"Then bring {root}/runs.csv and steps.csv back into one folder here, and read them with",
            f"`python {Path(__file__).resolve().parent / 'readback.py'} {self.run} <that folder>`.",
        ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Plan the round, write it into the run folder, print it."""
    args = _arguments(argv)
    lines = Plan(Path(args.run), args).write()
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
