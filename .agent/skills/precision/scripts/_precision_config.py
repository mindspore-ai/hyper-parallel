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
"""Shared configuration contract for the precision launcher and training hook."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

DEFAULT_PATH = "precision_dump"
DEFAULT_STEPS = "0-1"
DEFAULT_TASK = "statistics"
DEFAULT_LEVEL = "L1"
TASKS = ("statistics", "tensor")
LEVELS = ("L0", "L1", "mix")
# Reject the characters disallowed by msprobe before launching a workload.
_PATH_FORBIDDEN = set("[]{}(),;:*?\"'<>|&$`!\n\t")


def parse_steps(spec: str) -> list[int]:
    """Expand ranges and comma-separated IDs; an empty string means all IDs."""
    steps = []
    for part in spec.strip().split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            first, _, last = chunk.partition("-")
            low, high = int(first), int(last)
            if low < 0 or low > high:
                raise ValueError(f"invalid nonnegative step range: {chunk!r}")
            steps.extend(range(low, high + 1))
        else:
            steps.append(int(chunk))
    if any(step < 0 for step in steps):
        raise ValueError("step/rank IDs must be nonnegative")
    return sorted(set(steps))


def check_dump_path(dump_path: Path) -> None:
    """Fail early on a path msprobe's own validator would reject."""
    bad = sorted(_PATH_FORBIDDEN.intersection(str(dump_path)))
    if bad:
        raise ValueError(f"dump path {dump_path} contains characters msprobe rejects: {''.join(bad)!r}")


def validate_config(config: dict) -> None:
    """Validate the shared fields; leave additional msprobe options intact."""
    if not isinstance(config, dict):
        raise ValueError("precision config must be a JSON object")
    if config.get("task") not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if config.get("level") not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    dump = config.get("dump_path")
    if not isinstance(dump, str) or not dump or not Path(dump).is_absolute():
        raise ValueError("dump_path must be an absolute path")
    check_dump_path(Path(dump))
    for key in ("step", "rank"):
        values = config.get(key)
        if not isinstance(values, list) or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError(f"{key} must be a list of nonnegative integers")
    if not isinstance(config.get(config["task"]), dict):
        raise ValueError(f"missing options object for task {config['task']!r}")


def build_config(dump_path: Path, steps: list[int], task: str, level: str,
                 ranks: list[int] | None = None) -> dict:
    """Build validated msprobe configuration with per-task nested options."""
    config = {
        "task": task, "dump_path": str(dump_path.resolve()),
        "rank": list(ranks or []), "step": list(steps), "level": level,
        task: {"scope": [], "list": [], "data_mode": ["all"]},
    }
    if task == "statistics":
        config[task]["summary_mode"] = "statistics"
    validate_config(config)
    return config


def read_config(config_path: Path) -> dict:
    """Read a supplied configuration; never fall back on missing/invalid files."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def write_config(config: dict) -> Path:
    """Close a uniquely named file before publishing its path to readers.

    Files remain beside the dump for audit. Unique names also isolate direct
    hook callers across ranks, hosts, and independent process namespaces.
    """
    validate_config(config)
    dump_path = Path(config["dump_path"])
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dump_path.parent,
                                     prefix=f"{dump_path.name}.", suffix=".config.json", delete=False) as handle:
        json.dump(config, handle, indent=2)
        config_path = Path(handle.name)
    return config_path
