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
"""Read explicit communication metrics; unknown quantities remain null.

Volumes use decimal MB (1e6 bytes), bandwidth uses decimal Gbit/s.
Summed operation durations do not establish uncovered communication or compute.
"""

import json
import math
import re
import sqlite3
from contextlib import closing
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from _moe_log_records import log_records


class CommunicationAnalyzer:
    """Extract logged and supported profiler communication facts."""

    def __init__(self, steps: Optional[int] = None, ranks: Optional[list] = None):
        self.steps = steps
        self.ranks = ranks

    def analyze(self, log_files: List[str], profile_files: List[str]) -> Dict[str, Any]:
        """Return timing, volume and observed step fractions without fake defaults."""
        data, ranks_seen = self._collect_log_events(log_files)
        profile_events = self._collect_profile_events(profile_files, data)
        if not data["events"]:
            data["events"] = profile_events
        if self.ranks is not None:
            data["events"] = [event for event in data["events"] if event.get("rank") in self.ranks]
        if self.steps is not None:
            available = sorted({event["step"] for event in data["events"] if event.get("step") is not None})
            selected = set(available[:self.steps])
            data["events"] = [event for event in data["events"] if event.get("step") in selected]
        events = data["events"]
        ranks_seen.update(event["rank"] for event in events if event.get("rank") is not None)
        timing, volumes, durations = self._group_events(events, data.get("ep_size"))
        volume_rows, breakdown = self._summarize_steps(volumes, durations, data)
        return {"num_ranks": len(ranks_seen) or None, "world_size": data.get("world_size", data.get("num_ranks")),
                "ep_size": data.get("ep_size"), "hidden_size": data.get("hidden_size"),
                "dtype_bytes": data.get("dtype_bytes"), "timing": timing, "volume": volume_rows,
                "breakdown": breakdown, "warnings": data["warnings"],
                "notes": ["Operation-time sum / step time can exceed 1 due to overlap; compute is unknown",
                          "tokens_exchanged is an estimate from dispatch payload / (hidden_size * dtype_bytes)"]}

    def _collect_log_events(self, log_files: List[str]) -> tuple:
        data = {"events": [], "step_times": {}, "other_comm": {}, "warnings": []}
        pattern = re.compile(r"\[MoE\]\s+Layer\s+(\d+)\s+(dispatch|combine):\s*([\d.]+)ms,\s*([\d.]+)MB")
        ranks_seen = set()
        for line, step, rank in log_records(log_files, self.steps, self.ranks):
            if rank is not None:
                ranks_seen.add(rank)
            for field in ("world_size", "num_ranks", "ep_size", "hidden_size", "dtype_bytes"):
                match = re.search(r"\b" + field + r"\s*[=:]\s*(\d+)", line)
                if match:
                    data[field] = int(match.group(1))
            match = pattern.search(line)
            if match:
                layer, direction, duration, volume = match.groups()
                data["events"].append({"step": step, "rank": rank, "layer_id": int(layer),
                                       "direction": direction, "duration_ms": float(duration),
                                       "volume_mb": float(volume)})
            match = re.search(r"step_time\s*[=:]\s*([\d.]+)(ms|s)\b", line)
            if match and step is not None:
                data["step_times"][(rank, step)] = float(match.group(1)) * (1000 if match.group(2) == "s" else 1)
            match = re.search(r"other_comm\s*[=:]\s*([\d.]+)ms", line)
            if match and step is not None:
                data["other_comm"][(rank, step)] = float(match.group(1))
        return data, ranks_seen

    def _collect_profile_events(self, profile_files: List[str], data: dict) -> list:
        profile_events = []
        for filename in profile_files:
            try:
                if Path(filename).suffix == ".db":
                    events = self._read_sqlite(filename)
                else:
                    with open(filename, encoding="utf-8") as handle:
                        capture = json.load(handle)
                    events = self._read_json(capture)
                    if isinstance(capture, dict):
                        for key in ("world_size", "ep_size", "hidden_size", "dtype_bytes"):
                            if capture.get(key) is not None:
                                data.setdefault(key, capture[key])
                if not events:
                    data["warnings"].append(f"No supported AllToAll events with explicit units: {filename}")
                profile_events.extend(events)
            except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
                data["warnings"].append(f"{filename}: {exc}")
        return profile_events

    @staticmethod
    def _group_events(events: list, ep_size: Optional[int]) -> tuple:
        timing = []
        volumes = defaultdict(lambda: {"dispatch": [], "combine": []})
        durations = defaultdict(list)
        for event in events:
            duration, volume = event["duration_ms"], event["volume_mb"]
            bandwidth = volume * 8 / duration if volume is not None and duration is not None and duration > 0 else None
            timing.append([event.get("step"), event.get("layer_id"), event.get("direction"),
                           duration, volume, bandwidth, ep_size, event.get("rank")])
            if event.get("layer_id") is not None and event.get("direction") in ("dispatch", "combine"):
                volumes[event["layer_id"]][event["direction"]].append(volume)
            if event.get("step") is not None and duration is not None:
                durations[(event.get("rank"), event["step"])].append(duration)
        return timing, volumes, durations

    @staticmethod
    def _summarize_steps(volumes: dict, durations: dict, data: dict) -> tuple:
        volume_rows = []
        token_size = data.get("hidden_size", 0) * data.get("dtype_bytes", 0)
        for layer, directions in sorted(volumes.items()):
            dispatch = directions["dispatch"]
            combine = directions["combine"]
            dispatch_mb = sum(dispatch) if dispatch and all(value is not None for value in dispatch) else None
            combine_mb = sum(combine) if combine and all(value is not None for value in combine) else None
            tokens = dispatch_mb * 1e6 / token_size if dispatch_mb is not None and token_size else None
            volume_rows.append([layer, dispatch_mb, combine_mb, tokens, token_size or None])
        breakdown = []
        for (rank, step), values in sorted(durations.items(), key=lambda pair: str(pair[0])):
            total = data["step_times"].get((rank, step))
            ratio = sum(values) / total if total is not None and total > 0 else None
            breakdown.append([step, None, sum(values), data["other_comm"].get((rank, step)), ratio, total, rank])
        return volume_rows, breakdown

    @staticmethod
    def _read_json(capture: Any) -> list:
        if not isinstance(capture, dict):
            return []
        events = []
        operations = capture.get("ops", [])
        if not isinstance(operations, list) or any(not isinstance(op, dict) for op in operations):
            raise ValueError("Profiler ops must be a list of mappings")
        for op in operations:
            name = str(op.get("type", "")) + str(op.get("name", ""))
            if "alltoall" not in name.lower().replace("_", ""):
                continue
            duration = None
            for key, scale in (("duration_ns", 1e-6), ("duration_us", 1e-3), ("duration_ms", 1)):
                if op.get(key) is not None:
                    duration = float(op[key]) * scale
                    break
            size = float(op["bytes"]) / 1e6 if op.get("bytes") is not None else None
            if any(value is not None and (not math.isfinite(value) or value < 0) for value in (duration, size)):
                raise ValueError("Negative/nonfinite communication measurements")
            if duration is None and size is None:
                continue
            CommunicationAnalyzer._validate_position(op)
            events.append({"step": op.get("step"), "layer_id": op.get("layer_id"), "rank": op.get("rank"),
                           "direction": op.get("direction"), "duration_ms": duration, "volume_mb": size})
        return events

    @staticmethod
    def _validate_position(op: dict) -> None:
        for key in ("rank", "step", "layer_id"):
            value = op.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"Profiler {key} must be a nonnegative integer")

    @classmethod
    def _read_sqlite(cls, filename: str) -> list:
        events = []
        uri = Path(filename).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                table = '"' + row[0].replace('"', '""') + '"'
                columns = {item[1].lower(): item[1] for item in connection.execute(f"PRAGMA table_info({table})")}
                name = next((columns[key] for key in ("name", "op_name", "type") if key in columns), None)
                units = {key for key in columns if key in ("duration_ns", "duration_us", "duration_ms")}
                if not name or not units:
                    continue
                for item in connection.execute(f"SELECT * FROM {table}"):
                    op = {key.lower(): value for key, value in dict(item).items()}
                    op["name"] = item[name]
                    events.extend(cls._read_json({"ops": [op]}))
        return events
