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
"""What timing a template's layouts on a cluster needs: its timer record, the layouts it launches, its rounds.

Shared by ``launch.py``, which plans a round, and ``readback.py``, which reads rounds back. A round is the folder
a timer writes: ``runs.csv``, one row a launch, with ``steps.csv`` beside it. The rounds a template already has on
a cluster are kept under ``references/``, named by its record in ``references/timers.json``.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REFERENCES = Path(__file__).resolve().parents[1] / "references"
# A layout as a timer names it: its values of the timer's varying degrees, in the record's order.
Key = Tuple[str, ...]


@dataclass(frozen=True)
class Run:
    """One launch of a round, as its timer recorded it.

    Attributes:
        key: the layout, its values of the timer's varying degrees.
        repeat: which repeat of the layout the launch was.
        ok: whether the launch finished with its steps measured.
        mean_ms: the mean step over the steps the timer kept, in ms.
        peak_gib: the largest memory any die allocated, in GiB.
        reserved_gib: the largest memory any die reserved, in GiB.
        status: the status the timer wrote, ``ok`` or how the launch failed.
        source: the round's folder.
    """

    key: Key
    repeat: str
    ok: bool
    mean_ms: float
    peak_gib: float
    reserved_gib: float
    status: str
    source: str


def asked(run: Path, family: Optional[str], cluster: Optional[str]) -> Tuple[str, str]:
    """The template and the cluster a run of ``ask.py`` was priced for: the flags when given, else its ``ask.json``.

    Args:
        run: the run folder ``ask.py`` kept.
        family: the template named on the command line, or None.
        cluster: the cluster named on the command line, or None.

    Returns:
        The template and the cluster.

    Raises:
        SystemExit: neither the flags nor the run folder name a template.
    """
    record: Dict[str, Any] = {}
    path = run / "ask.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
    family = family or record.get("family")
    cluster = cluster or record.get("cluster") or "a3"
    if not family:
        raise SystemExit(f"{run} names no template: give --family")
    return family, cluster


def timer_entry(family: str, cluster: str) -> Optional[Dict[str, Any]]:
    """How the template's layouts are timed on the cluster, or None when nothing times them there."""
    records = json.loads((REFERENCES / "timers.json").read_text(encoding="utf-8"))
    return records.get(f"{family}/{cluster}")


def key_of(row: Dict[str, str], timer: Dict[str, Any]) -> Key:
    """A row of ND's ranking as the layout its timer names."""
    return tuple(row[name] for name in timer["varies"])


def name_of(key: Key, timer: Dict[str, Any]) -> str:
    """A layout as people write it, for example ``EP2 OP1``; ND's CSV calls tensor parallelism MP."""
    return " ".join(f"{'TP' if name == 'MP' else name}{value}" for name, value in zip(timer["varies"], key))


def launchable(rows: Sequence[Dict[str, str]], timer: Dict[str, Any], doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """The rows of ND's ranking the timer can launch for this run, in ND's order.

    Those state the timer's fixed degrees, the data parallel degree its dies leave, and the run's micro-batch and
    recompute mode, which the timer leaves as the yaml states them.
    """
    wanted = dict(timer["fixed"])
    split = 1
    for name in ("MP", "CP", "PP"):
        split *= int(wanted.get(name, "1"))
    wanted["DP"] = str(timer["dies"] // split)
    micro = (doc.get("training") or {}).get("micro_batch_size")
    if micro is not None:
        wanted["MBS"] = str(micro)
    mode = (doc.get("activation_checkpoint") or {}).get("mode")
    if mode:
        wanted["recompute"] = str(mode)
    return [row for row in rows if all(row.get(name, value) == value for name, value in wanted.items())]


def _number(text: Optional[str]) -> float:
    """A CSV cell as a number, 0 when it is empty."""
    return float(text) if text not in (None, "") else 0.0


def read_round(folder: Path, varies: Sequence[str]) -> List[Run]:
    """Every launch a round's ``runs.csv`` records, the failed ones included.

    Args:
        folder: the round's folder, the ``RUN_ROOT`` its timer wrote.
        varies: the timer's varying degrees, which name its columns in lower case.

    Returns:
        The round's launches, in the order it ran them.

    Raises:
        SystemExit: the folder holds no ``runs.csv``, or one this template's timer did not write.
    """
    path = folder / "runs.csv"
    if not path.is_file():
        raise SystemExit(f"no runs.csv in {folder}: give the folder the timer wrote as RUN_ROOT")
    runs = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                key = tuple(row[name.lower()] for name in varies)
            except KeyError as exc:
                raise SystemExit(f"{path} has no column {exc}, so this template's timer did not write it") from exc
            runs.append(Run(key=key, repeat=row.get("repeat", ""), ok=row.get("status") == "ok",
                            mean_ms=_number(row.get("mean_ms")), peak_gib=_number(row.get("peak_alloc_gb")),
                            reserved_gib=_number(row.get("peak_reserved_gb")), status=row.get("status", ""),
                            source=folder.as_posix()))
    return runs


def recorded_rounds(timer: Dict[str, Any]) -> List[Path]:
    """The folders of the rounds the template already has on the cluster, oldest first."""
    return [REFERENCES / folder for folder in timer.get("timed", [])]
