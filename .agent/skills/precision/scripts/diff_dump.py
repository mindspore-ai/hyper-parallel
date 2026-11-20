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
"""Compare two msprobe dumps and report the first diverging API / module.

Runs ``msprobe compare`` and scans its table in execution order. msprobe is
invoked as an external tool, never imported, so this script loads and
lint-checks without it installed.

Three properties of the real comparison table drive the parsing, each of
which silently breaks a naive reader:

* **The metric columns differ by dump task.** A ``statistics`` dump compares
  as ``Max diff`` / ``Min diff`` / ``Mean diff`` / ``L2norm diff`` plus
  ``*RelativeErr``; ``Cosine`` exists **only** for a ``tensor`` dump. Looking
  for ``Cosine`` in a statistics table finds nothing on every row.
* **``MaxRelativeErr`` is a percent string in statistics mode**
  (``'1.1049723756906076%'``) and a bare ratio in tensor mode
  (``'0.013333333656'``). A plain ``float()`` raises on the former, and
  treating that as "not comparable" skips every diverging row - which reads
  out as "no divergence" on a run that genuinely diverged.
* **msprobe's own ``Result`` column is not a verdict.** It is observed as
  ``'pass'`` on a row whose ``MaxRelativeErr`` is 9368. It is reported here
  for context and never gated on.

Judgement defaults to **bitwise**: any non-zero difference is a divergence,
so the row reported is the first cause rather than a downstream symptom. A
band (``--max-rel`` / ``--cosine-min``) is opt-in, because a band large
enough to ignore noise also hides the cause: on a known single-layer
perturbation the perturbed weight sits at ``MaxRelativeErr`` 0.013 and
``Cosine`` 0.999998, inside a 0.02 / 0.99 band, while its propagated output
sits outside it - so the band reports the symptom and drops the cause.

``rules/precision-acceptance.md`` is the acceptance bar, and it is about the
**1000-step loss trajectory** (mean absolute error <= 0.02 or mean relative
error <= 2%). Those numbers are not a per-tensor band and are deliberately
not reused as one here.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

# Column names as the installed msprobe writes them.
_NAME_KEYS = ("NPU Name", "Bench Name", "Op Name", "name")
_COSINE_KEY = "Cosine"
_STAT_DIFF_KEYS = ("Max diff", "Min diff", "Mean diff", "L2norm diff")
_TENSOR_DIFF_KEYS = ("MaxAbsErr", "EucDist")
_MAXREL_KEY = "MaxRelativeErr"
_RESULT_KEY = "Result"
# Cells msprobe writes when a row carries no comparable tensor - observed as
# "unsupported" for a None bias, and as an empty cell. Not a divergence, and
# not an error either. `nan` is deliberately **not** in this set: a non-finite
# metric is the failure this skill exists to find, so it is reported rather
# than skipped.
_NON_COMPARABLE = {"", "unsupported", "N/A", "None", "-"}


def _cell(row: dict, key: str) -> str | None:
    """Trimmed cell value, or ``None`` when absent / non-comparable."""
    raw = row.get(key)
    if raw is None:
        return None
    text = raw.strip()
    return None if text in _NON_COMPARABLE else text


def to_float(value: str | None) -> float | None:
    """Parse a metric cell, accepting the percent form used in statistics mode.

    ``'1.1049723756906076%'`` becomes ``0.011049723756906076`` so both dump
    tasks are judged on the same scale.
    """
    if value is None:
        return None
    text = value.strip()
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    except ValueError:
        return None


def table_mode(fieldnames: list[str]) -> str:
    """``"tensor"`` when the table carries ``Cosine``, else ``"statistics"``."""
    return "tensor" if _COSINE_KEY in (fieldnames or []) else "statistics"


def step_dir(path: Path, step: str | None) -> Path:
    """Resolve a dump root to the step directory ``msprobe compare`` wants.

    ``compare`` rejects a dump root outright (its path check raises on a
    directory that is not a step / rank level), so a caller passing
    ``.../dump`` gets a bare "invalid file" error. Accept either and resolve.
    """
    if (path / "dump.json").is_file():
        return path
    candidates = sorted(p for p in path.iterdir()
                        if p.is_dir() and (p / "dump.json").is_file()) \
        if path.is_dir() else []
    if step:
        wanted = step if step.startswith("step") else f"step{step}"
        for candidate in candidates:
            if candidate.name == wanted:
                return candidate
        raise FileNotFoundError(f"{wanted} not found under {path}; "
                                f"have {[c.name for c in candidates]}")
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"no step directory with a dump.json under {path} - the run "
            f"produced no dump (see collect_dump.py)")
    raise ValueError(f"{path} holds {len(candidates)} steps "
                     f"({[c.name for c in candidates]}); pass --step")


def run_compare(target: Path, golden: Path, out_dir: Path) -> Path:
    """Invoke ``msprobe compare`` and return the newest result table.

    The real flag surface is ``-tp`` / ``-gp`` / ``-o``; there is no ``-f``
    framework flag and no ``-np`` / ``-bp``. Output is CSV by default. The
    file name carries the pid and a timestamp, so the newest is picked by
    modification time rather than by name order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    before = {p: p.stat().st_mtime for p in out_dir.glob("**/*.csv")}
    command = ["msprobe", "compare", "-tp", str(target), "-gp", str(golden),
               "-o", str(out_dir)]
    print(f"[diff_dump] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    produced = [p for p in out_dir.glob("**/*.csv")
                if p not in before or p.stat().st_mtime > before[p]]
    pool = produced or list(out_dir.glob("**/*.csv"))
    if not pool:
        raise FileNotFoundError(f"no comparison CSV produced in {out_dir}")
    return max(pool, key=lambda p: p.stat().st_mtime)


def row_metrics(row: dict, mode: str) -> tuple[dict, float | None, float | None]:
    """Return ``(diffs, max_rel, cosine)`` for one row of the given mode.

    The diff columns differ by dump task, and ``Cosine`` exists only for a
    tensor dump, so the column set is picked from the mode rather than
    probed per row.
    """
    diff_keys = _STAT_DIFF_KEYS if mode == "statistics" else _TENSOR_DIFF_KEYS
    diffs = {k: to_float(_cell(row, k)) for k in diff_keys}
    return ({k: v for k, v in diffs.items() if v is not None},
            to_float(_cell(row, _MAXREL_KEY)),
            to_float(_cell(row, _COSINE_KEY)) if mode == "tensor" else None)


def _nonfinite(diffs: dict, rel: float | None, cosine: float | None) -> dict:
    """Metrics that are nan / inf, which no band may excuse."""
    everything = {**diffs, _MAXREL_KEY: rel, _COSINE_KEY: cosine}
    return {k: v for k, v in everything.items()
            if v is not None and not math.isfinite(v)}


def _bitwise_hit(diffs: dict, rel: float | None) -> dict:
    """Non-zero differences, so the row reported is the first cause."""
    nonzero = {k: v for k, v in diffs.items() if v != 0.0}
    if nonzero:
        return nonzero
    if not diffs and rel:
        return {_MAXREL_KEY: rel}
    return {}


def _band_hit(rel: float | None, cosine: float | None, max_rel: float | None,
              cosine_min: float | None) -> dict:
    """Metrics outside an explicitly requested band."""
    hit: dict = {}
    if max_rel is not None and rel is not None and rel > max_rel:
        hit[_MAXREL_KEY] = rel
    if cosine_min is not None and cosine is not None and cosine < cosine_min:
        hit[_COSINE_KEY] = cosine
    return hit


def row_divergence(row: dict, mode: str, cosine_min: float | None,
                   max_rel: float | None) -> dict | None:
    """Return the metrics that make this row a divergence, or ``None``.

    With no band given the test is bitwise: any non-zero difference counts.
    A non-finite metric is a divergence whatever band is in force.
    """
    diffs, rel, cosine = row_metrics(row, mode)
    if not diffs and rel is None and cosine is None:
        return None  # nothing comparable on this row
    nonfinite = _nonfinite(diffs, rel, cosine)
    if nonfinite:
        return nonfinite
    banded = max_rel is not None or cosine_min is not None
    hit = (_band_hit(rel, cosine, max_rel, cosine_min) if banded
           else _bitwise_hit(diffs, rel))
    if not hit:
        return None
    for key, value in ((_MAXREL_KEY, rel), (_COSINE_KEY, cosine)):
        if value is not None:
            hit.setdefault(key, value)
    return hit


def scan(csv_path: Path, cosine_min: float | None,
         max_rel: float | None) -> dict:
    """Scan the table in execution order; summarise the divergence."""
    with csv_path.open("r", encoding="utf-8", errors="replace",
                       newline="") as handle:
        reader = csv.DictReader(handle)
        mode = table_mode(reader.fieldnames)
        rows = list(reader)
    first: dict | None = None
    diverging = 0
    comparable = 0
    for index, row in enumerate(rows):
        hit = row_divergence(row, mode, cosine_min, max_rel)
        if hit is None:
            if _cell(row, _MAXREL_KEY) is not None or _cell(row, _COSINE_KEY):
                comparable += 1
            continue
        comparable += 1
        diverging += 1
        if first is None:
            first = {
                "order": index,
                "name": next((_cell(row, k) for k in _NAME_KEYS
                              if _cell(row, k)), "<unknown>"),
                "metrics": hit,
                "msprobe_result": row.get(_RESULT_KEY, ""),
            }
    return {"mode": mode, "rows": len(rows), "comparable": comparable,
            "diverging": diverging, "first": first, "table": csv_path}


def main(argv: list[str] | None = None) -> int:
    """Compare candidate vs baseline dumps and print the first divergence."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/precision/scripts/diff_dump.py",
        description="Compare two msprobe dumps; report the first divergence.")
    parser.add_argument("--candidate", required=True,
                        help="candidate dump root or step directory")
    parser.add_argument("--baseline", required=True,
                        help="baseline dump root or step directory")
    parser.add_argument("--step", default=None,
                        help="step to compare when a dump holds several")
    parser.add_argument("--out-dir", default="precision_compare")
    parser.add_argument("--cosine-min", type=float, default=None,
                        help="opt-in band: flag rows with Cosine below this "
                             "(tensor dumps only). Default is bitwise.")
    parser.add_argument("--max-rel", type=float, default=None,
                        help="opt-in band: flag rows whose MaxRelativeErr "
                             "exceeds this ratio. Default is bitwise.")
    args = parser.parse_args(argv)

    candidate = step_dir(Path(args.candidate).resolve(), args.step)
    baseline = step_dir(Path(args.baseline).resolve(), args.step)
    table = run_compare(candidate, baseline, Path(args.out_dir).resolve())
    summary = scan(table, args.cosine_min, args.max_rel)

    band = ("bitwise" if args.cosine_min is None and args.max_rel is None
            else f"cosine>={args.cosine_min} / max_rel<={args.max_rel}")
    print(f"[diff_dump] task={summary['mode']} rows={summary['rows']} "
          f"comparable={summary['comparable']} band={band}", flush=True)
    first = summary["first"]
    if first is None:
        print(f"[diff_dump] no divergence; table={summary['table']}", flush=True)
        return 0
    metrics = ", ".join(f"{k}={v}" for k, v in first["metrics"].items())
    print(f"[diff_dump] FIRST DIVERGENCE at exec#{first['order']} "
          f"{first['name']}: {metrics}", flush=True)
    print(f"[diff_dump] {summary['diverging']} of {summary['comparable']} "
          f"comparable rows diverge; msprobe's own Result column said "
          f"{first['msprobe_result']!r} on that row (not a verdict)",
          flush=True)
    print(f"[diff_dump] table={summary['table']}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
