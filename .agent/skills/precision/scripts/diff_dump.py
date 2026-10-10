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
"""Compare a candidate dump against a baseline and report the first divergence.

This is the public-toolchain skeleton: it invokes ``msprobe compare`` to
produce the per-API/module comparison table, then scans it in execution
order for the first row that falls below the similarity threshold or above
the error threshold. The exact thresholds and the tie-noise rule are aligned
with ``rules/precision-acceptance.md``; any internal-specific judging stays
``[待对齐]``.

msprobe is invoked as an external tool, never imported, so this script loads
and lint-checks without it installed. The comparison CSV it parses is the
public msprobe output shape; column names are resolved defensively.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# Defaults aligned with the acceptance rule's known-noise boundary; override
# via flags. [待对齐] with the internal precision-align口径 when available.
_DEFAULT_COSINE = 0.99
_DEFAULT_MAX_REL = 0.02
# Candidate column names in the public msprobe compare CSV (resolved loosely).
_COSINE_KEYS = ("Cosine", "cosine_similarity", "余弦相似度")
_MAXREL_KEYS = ("MaxRelativeErr", "Max_Relative_Error", "最大相对误差")
_NAME_KEYS = ("NPU Name", "api_name", "Op Name", "name")


def _run_msprobe_compare(candidate: Path, baseline: Path, out_dir: Path) -> Path:
    """Invoke ``msprobe compare``; return the comparison CSV path.

    Argv list by contract. The flag surface is the public msprobe one; adjust
    here if the installed msprobe version differs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "msprobe", "-f", "pytorch", "compare",
        "-np", str(candidate), "-bp", str(baseline),
        "-o", str(out_dir),
    ]
    print(f"[diff_dump] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)
    csvs = sorted(out_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no comparison CSV produced in {out_dir}")
    return csvs[-1]


def _resolve(row: dict, keys: tuple[str, ...]) -> str | None:
    """Return the first present column value among ``keys``."""
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    return None


def _first_divergence(csv_path: Path, cosine_min: float, max_rel: float) -> dict | None:
    """Scan the comparison table in order; return the first row past the band.

    A row diverges when its cosine similarity is below ``cosine_min`` or its
    max relative error exceeds ``max_rel``. Rows whose metrics are absent
    (non-comparable) are skipped, not treated as divergence.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            cosine = _to_float(_resolve(row, _COSINE_KEYS))
            max_rel_err = _to_float(_resolve(row, _MAXREL_KEYS))
            if cosine is None and max_rel_err is None:
                continue
            diverged = ((cosine is not None and cosine < cosine_min)
                        or (max_rel_err is not None and max_rel_err > max_rel))
            if diverged:
                return {
                    "order": index,
                    "name": _resolve(row, _NAME_KEYS) or "<unknown>",
                    "cosine": cosine,
                    "max_rel": max_rel_err,
                }
    return None


def _to_float(value: str | None) -> float | None:
    """Best-effort float parse; ``None`` for absent / non-numeric cells."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    """Compare candidate vs baseline dumps and print the first divergence."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/precision/scripts/diff_dump.py",
        description="Compare two msprobe dumps and report the first diverging API/module.",
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out-dir", default="precision_compare")
    parser.add_argument("--cosine-min", type=float, default=_DEFAULT_COSINE)
    parser.add_argument("--max-rel", type=float, default=_DEFAULT_MAX_REL)
    args = parser.parse_args(argv)

    csv_path = _run_msprobe_compare(
        Path(args.candidate).resolve(), Path(args.baseline).resolve(),
        Path(args.out_dir).resolve())
    first = _first_divergence(csv_path, args.cosine_min, args.max_rel)
    if first is None:
        print(f"[diff_dump] no divergence beyond band (cosine>={args.cosine_min}, "
              f"max_rel<={args.max_rel}); table={csv_path}", flush=True)
        return 0
    print(f"[diff_dump] FIRST DIVERGENCE at exec#{first['order']} {first['name']}: "
          f"cosine={first['cosine']} max_rel={first['max_rel']}; table={csv_path}",
          flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
