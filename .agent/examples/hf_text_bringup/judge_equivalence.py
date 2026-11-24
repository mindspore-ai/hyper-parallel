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
"""Judge the S3 equivalence matrix, and say at what precision.

Two evidence sources are accepted, and every verdict line names the one it
used, because they support different claims:

* **Structured runtime evidence (use this).** Run each case with
  ``--model_integration.mode=runtime`` and its own
  ``HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR``. Rank 0 then writes
  ``cases/runtime/metrics.jsonl`` holding every step's loss, pre- and
  post-clip gradient norm, learning rate and ``global_input_sha256`` at full
  float precision. Pass that file, or the output directory, as a series.
  Each step is compared by the framework's own ``compare_scalar_curves``
  (``hyper_parallel.tools.model_integration.case_compare``, the comparison
  ``hf-model-precision-validation`` uses): exact equality, and a step whose
  logical inputs differ is BLOCKED rather than graded, because different
  inputs make a numerical verdict meaningless. The input hash covers each
  sample separately and is order-free, so it stays comparable across data
  parallel degrees.
* **Trainer text logs (fallback).** The logging callback prints six
  significant digits. Agreement there is agreement to about 1e-6 relative,
  not bitwise agreement, and nothing checks that the two runs saw the same
  inputs. Verdicts from this source carry that label on every line.

Checks (identical = exact at the precision of the evidence):

* ``--identical A,B`` (alias ``--bitwise``) every step identical: recompute
  on/off, deterministic rerun;
* ``--band A,B,TOL`` per-step relative loss difference within ``TOL``;
* ``--early-identical A,B,N`` (alias ``--early-bitwise``) the cross-layout
  criterion: the first ``N`` steps identical, and the step where the series
  first differ, the difference there and at the last common step, are
  reported rather than graded. Late-step drift is accumulation amplified by
  training dynamics; a fixed band over it measures the optimizer's
  amplification rate, not the framework. The inputs must be controlled
  first (released weights and a token-uniform corpus, see the skill's
  cross-layout section); structured evidence verifies that per step.

Example:

    python judge_equivalence.py \
        --series dp8=runs/dp8/model_integration \
        --series rerun=runs/rerun/model_integration \
        --series dp4=runs/dp4/model_integration \
        --identical dp8,rerun --early-identical dp8,dp4,3

Exit codes: 0 every check passed; 2 at least one failed or was blocked;
1 usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_STEP = re.compile(r"\| step=(\d+) .*?training/grad_norm=([0-9.eE+-]+)"
                   r" .*?training/total_loss=([0-9.eE+-]+)")

PRINTED = "printed precision ~6 significant digits, inputs unchecked"
EXACT = "full precision, inputs checked"


@dataclass
class Series:
    """One run's per-step evidence and where it came from."""

    source: str
    rows: list

    @property
    def label(self) -> str:
        """Precision label every verdict on this series carries."""
        return EXACT if self.source == "structured" else PRINTED


def read_series(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(step, loss, grad_norm)`` per logged step, printed precision."""
    series = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = _STEP.search(line)
            if match:
                series.append((int(match.group(1)), match.group(3),
                               match.group(2)))
    return series


def find_runtime_metrics(path: Path) -> Path | None:
    """Locate the runtime's ``metrics.jsonl`` from a file or output directory."""
    if path.is_file():
        return path if path.suffix == ".jsonl" else None
    for candidate in (path / "cases" / "runtime" / "metrics.jsonl",
                      path / "metrics.jsonl"):
        if candidate.is_file():
            return candidate
    return None


def load_series(path: Path) -> Series:
    """Load structured runtime evidence when present, else a text log."""
    metrics = find_runtime_metrics(path)
    if metrics is not None:
        # Imported here so text-log judging keeps working without the package.
        from hyper_parallel.tools.model_integration.case_compare import (  # pylint: disable=import-outside-toplevel
            load_jsonl,
        )
        return Series("structured", load_jsonl(metrics))
    if path.is_dir():
        raise ValueError(f"no cases/runtime/metrics.jsonl under {path}")
    return Series("printed", read_series(path))


def _compare_rows(rows_a: list, rows_b: list) -> dict:
    """Per-step exact comparison by the framework's scalar comparator."""
    from hyper_parallel.tools.model_integration.case_compare import (  # pylint: disable=import-outside-toplevel
        compare_scalar_curves,
    )
    return compare_scalar_curves(rows_a, rows_b, {})


def _difference(row: dict) -> str:
    """Name what differed in one compared step."""
    return (f"loss rel {row['loss_max_rel']:.3e}, "
            f"grad-norm rel {row['norm_max_rel']:.3e}")


def early_bitwise(series_a: list, series_b: list, steps: int):
    """Judge the first ``steps`` of two printed-precision series.

    Returns ``(passed, detail)``. The detail always states where the printed
    values first differ and how far apart they are by the last common step.
    A printed value can hide a difference below its last digit, so the step
    named here is where the difference became visible, not where it began.
    """
    if not series_a or not series_b:
        return False, "no steps to compare"
    common = min(len(series_a), len(series_b))
    if common < steps:
        return False, f"only {common} common steps, need {steps}"

    early = [a == b for a, b in zip(series_a[:steps], series_b[:steps])]
    onset = next((index + 1 for index, (a, b)
                  in enumerate(zip(series_a, series_b)) if a != b), None)
    if onset is None:
        return all(early), (f"first {steps} steps identical; "
                            f"no divergence through step {common}")

    def relative(index):
        first, second = float(series_a[index][1]), float(series_b[index][1])
        return abs(first - second) / max(abs(first), 1e-12)

    detail = (f"first {sum(early)}/{steps} steps identical; divergence from "
              f"step {onset} (rel {relative(onset - 1):.3e}), reaching "
              f"{relative(common - 1):.3e} by step {common}")
    return all(early), detail


def early_exact(rows_a: list, rows_b: list, steps: int) -> tuple[str, str]:
    """Judge the first ``steps`` of two structured series at full precision.

    Returns ``(status, detail)`` with status ``PASS``, ``FAIL`` or
    ``BLOCKED``. Blocked means the runs consumed different logical inputs
    within the judged window, so the comparison says nothing about the
    layouts and must not be read as either verdict.
    """
    if not rows_a or not rows_b:
        return "BLOCKED", "structured evidence is missing or empty"
    common = min(len(rows_a), len(rows_b))
    if common < steps:
        return "FAIL", f"only {common} common steps, need {steps}"
    result = _compare_rows(rows_a[:common], rows_b[:common])
    if "steps" not in result:
        return "FAIL", result.get("reason", "rows could not be aligned")
    rows = result["steps"]
    early = rows[:steps]
    blocked = [row["step"] for row in early if row["status"] == "BLOCKED"]
    if blocked:
        return "BLOCKED", (f"logical inputs differ at steps {blocked}; the "
                           "comparison says nothing about the layouts")
    exact = sum(row["status"] == "PASS" for row in early)
    onset = next((row for row in rows if row["status"] != "PASS"), None)
    if onset is None:
        return "PASS", (f"first {steps} steps exact; no difference through "
                        f"step {rows[-1]['step']}")
    if onset["status"] == "BLOCKED":
        tail = (f"inputs diverge from step {onset['step']}; later steps "
                "are not comparable")
    else:
        tail = (f"first difference at step {onset['step']} "
                f"({_difference(onset)}), reaching loss rel "
                f"{rows[-1]['loss_max_rel']:.3e} by step {rows[-1]['step']}")
    status = "PASS" if exact == steps else "FAIL"
    return status, f"first {exact}/{steps} steps exact; {tail}"


def exact_all(rows_a: list, rows_b: list) -> tuple[str, str]:
    """Every step of two structured series exact, inputs identical."""
    result = _compare_rows(rows_a, rows_b)
    if "steps" not in result:
        return result["status"], result.get("reason", "")
    rows = result["steps"]
    exact = sum(row["status"] == "PASS" for row in rows)
    if result["status"] == "PASS":
        return "PASS", f"{exact}/{len(rows)} steps exact"
    blocked = [row["step"] for row in rows if row["status"] == "BLOCKED"]
    failed = next((row for row in rows if row["status"] == "FAIL"), None)
    if failed is None:
        return "BLOCKED", f"logical inputs differ at steps {blocked}"
    return "FAIL", (f"{exact}/{len(rows)} steps exact; first difference at "
                    f"step {failed['step']} ({_difference(failed)})")


def _printed_identical(sa: list, sb: list) -> tuple[str, str]:
    same = [x == y for x, y in zip(sa, sb)]
    ok = bool(same) and all(same) and len(sa) == len(sb)
    return ("PASS" if ok else "FAIL"), (f"{sum(same)}/{max(len(sa), len(sb))} "
                                       "steps identical")


def _band(a: Series, b: Series, tol: float) -> tuple[str, str]:
    if a.source == "structured":
        result = _compare_rows(a.rows, b.rows)
        if "steps" not in result:
            return "FAIL", result.get("reason", "rows could not be aligned")
        rows = result["steps"]
        if any(row["status"] == "BLOCKED" for row in rows):
            return "BLOCKED", "logical inputs differ; a band over them means nothing"
        rel = [row["loss_max_rel"] for row in rows]
    else:
        rel = [abs(float(x[1]) - float(y[1])) / max(abs(float(x[1])), 1e-12)
               for x, y in zip(a.rows, b.rows)]
    if not rel:
        return "FAIL", "no overlapping steps"
    return ("PASS" if max(rel) <= tol else "FAIL"), \
        f"loss rel diff max={max(rel):.3e} tol={tol:.1e}"


def judge(series: dict, identical: list, band: list,
          early: list | None = None) -> list[tuple[str, str, str, str]]:
    """Evaluate every requested check.

    Returns ``(name, status, precision label, detail)`` rows. Two series of
    different provenance are never compared: a printed value against a full
    precision one would grade the formatting, not the runs.
    """
    mixed = ("FAIL", "mixed evidence",
             "one series is a text log, the other structured")
    rows = []
    for name_a, name_b, steps in (early or []):
        name = f"early-identical {name_a}=={name_b} (first {steps})"
        a, b = series[name_a], series[name_b]
        if a.source != b.source:
            rows.append((name, *mixed))
            continue
        if a.source == "structured":
            status, detail = early_exact(a.rows, b.rows, steps)
        else:
            passed, detail = early_bitwise(a.rows, b.rows, steps)
            status = "PASS" if passed else "FAIL"
        rows.append((name, status, a.label, detail))
    for name_a, name_b in identical:
        name = f"identical {name_a}=={name_b}"
        a, b = series[name_a], series[name_b]
        if a.source != b.source:
            rows.append((name, *mixed))
            continue
        if a.source == "structured":
            status, detail = exact_all(a.rows, b.rows)
        else:
            status, detail = _printed_identical(a.rows, b.rows)
        rows.append((name, status, a.label, detail))
    for name_a, name_b, tol in band:
        name = f"band {name_a}~{name_b}"
        a, b = series[name_a], series[name_b]
        if a.source != b.source:
            rows.append((name, *mixed))
            continue
        status, detail = _band(a, b, tol)
        rows.append((name, status, a.label, detail))
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Judge S3 equivalence from runtime evidence or trainer logs.")
    parser.add_argument("--series", action="append", default=[],
                        metavar="NAME=PATH", required=True,
                        help="metrics.jsonl, a model-integration output "
                             "directory, or a trainer log")
    parser.add_argument("--identical", "--bitwise", dest="identical",
                        action="append", default=[], metavar="A,B")
    parser.add_argument("--band", action="append", default=[],
                        metavar="A,B,TOL")
    parser.add_argument("--early-identical", "--early-bitwise", dest="early",
                        action="append", default=[], metavar="A,B,N")
    args = parser.parse_args(argv)

    series = {}
    for item in args.series:
        name, _, path = item.partition("=")
        if not path:
            parser.error(f"--series needs NAME=PATH, got {item!r}")
        try:
            series[name] = load_series(Path(path))
        except ValueError as err:
            parser.error(str(err))
        loaded = series[name]
        print(f"[judge] {name}: {len(loaded.rows)} steps from {path} "
              f"({loaded.label})")
    try:
        identical = [tuple(spec.split(",")) for spec in args.identical]
        band = [(a, b, float(t)) for spec in args.band
                for a, b, t in [spec.split(",")]]
        early = [(a, b, int(n)) for spec in args.early
                 for a, b, n in [spec.split(",")]]
    except ValueError as err:
        parser.error(str(err))

    failed = 0
    for name, status, label, detail in judge(series, identical, band, early):
        print(f"[judge] {status} | {name} [{label}] | {detail}")
        failed += 0 if status == "PASS" else 1
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
