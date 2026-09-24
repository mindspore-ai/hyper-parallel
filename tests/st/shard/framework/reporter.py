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
"""Per-rank jsonl reporter and cross-rank summariser."""
import glob
import json
import os
from typing import Dict, List


class Reporter:
    """Append-only jsonl writer used inside the launcher child process.

    One file per rank avoids any cross-rank locking. The summary path is
    rank-0 only; all other ranks just emit their own jsonl.
    """

    def __init__(self, report_dir: str, rank: int) -> None:
        """Open the per-rank jsonl file inside ``report_dir`` for append."""
        os.makedirs(report_dir, exist_ok=True)
        self.path = os.path.join(report_dir, f"rank{rank}.jsonl")
        self._fh = open(self.path, "a", encoding="utf-8")  # pylint: disable=R1732
        self.rank = rank

    def _write(self, record: dict) -> None:
        record["rank"] = self.rank
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def pass_(self, name: str, seconds: float) -> None:
        """Record a passing case with wall-clock duration in seconds."""
        self._write({"name": name, "status": "PASS", "ms": int(seconds * 1000)})

    def fail(self, name: str, err: BaseException, stack: str) -> None:
        """Record a failing case with the exception summary and stack."""
        self._write({
            "name": name, "status": "FAIL",
            "err": f"{type(err).__name__}: {err}", "stack": stack,
        })

    def skip(self, name: str, reason: str) -> None:
        """Record a skipped case (e.g. group broken after a comm failure)."""
        self._write({"name": name, "status": "SKIP", "reason": reason})

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()


def summarize(report_dir: str) -> str:
    """Merge all rank jsonl files; per-case status = worst of all ranks
    (FAIL > SKIP > PASS). Returns a human-readable table string.
    """
    files = sorted(glob.glob(os.path.join(report_dir, "rank*.jsonl")))
    rank_per_case: Dict[str, List[dict]] = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rank_per_case.setdefault(rec["name"], []).append(rec)

    order = {"PASS": 0, "SKIP": 1, "FAIL": 2}
    merged: List[dict] = []
    for recs in rank_per_case.values():
        worst = max(recs, key=lambda r: order.get(r["status"], 0))
        merged.append(worst)

    lines = []
    lines.append("=" * 72)
    lines.append(f"SHARD OPS SUITE REPORT  ({report_dir})")
    lines.append("=" * 72)
    n_pass = n_fail = n_skip = 0
    for rec in merged:
        st = rec["status"]
        if st == "PASS":
            ms = rec.get("ms", 0)
            lines.append(f"PASS  {rec['name']:<50s}  {ms:>6d} ms")
            n_pass += 1
        elif st == "SKIP":
            lines.append(f"SKIP  {rec['name']:<50s}  -- ({rec.get('reason','')})")
            n_skip += 1
        else:
            lines.append(f"FAIL  {rec['name']:<50s}")
            lines.append(f"        {rec.get('err','')}")
            n_fail += 1
    lines.append("-" * 72)
    lines.append(f"Total: {len(merged)}   Pass: {n_pass}   Fail: {n_fail}   Skip: {n_skip}")
    lines.append("=" * 72)
    return "\n".join(lines)
