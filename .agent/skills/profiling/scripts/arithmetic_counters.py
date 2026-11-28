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
"""Extract physical-core counters from CANN ArithmeticUtilization SQLite exports."""

import argparse
from collections.abc import Sequence
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from typing import TypedDict


COLUMNS = ("coreid", "r49", "r4a", "r4b", "r4c", "r4d", "r4e", "r4f")
EVENT_QUERY = f"SELECT {', '.join(COLUMNS)} FROM EventCount ORDER BY coreid"


class _EngineCounters(TypedDict):
    """Source database and its physical-core events."""

    database: str
    sha256: str
    events: list[tuple[int, ...]]


def _read_engine(path: Path, expected_cores: int) -> _EngineCounters:
    """Read one finalized engine export and identify its source bytes."""
    if not path.is_file():
        raise ValueError(f"Missing counter export: {path}")
    try:
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
            events = connection.execute(EVENT_QUERY).fetchall()
    except sqlite3.Error as error:
        raise ValueError(f"Cannot read EventCount from {path}: {error}") from error
    if any(type(value) is not int or value < 0 for row in events for value in row):
        raise ValueError(f"Expected nonnegative integer counter values and physical core IDs in {path}")
    if len(events) != expected_cores or len({row[0] for row in events}) != expected_cores:
        raise ValueError(f"Expected {expected_cores} distinct physical cores in {path}; got {len(events)} rows")

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return {"database": path.name, "sha256": digest.hexdigest(), "events": events}


def read_arithmetic_counters(
    profile: Path,
    device_ids: Sequence[int],
    cube_cores: int,
    vector_cores: int,
) -> dict[str, object]:
    """Collect raw register totals after verifying the declared physical scope.

    Args:
        profile: Finalized PROF directory containing the device directories.
        device_ids: Physical device IDs from the capture inventory.
        cube_cores: Expected CUBE core count per declared device.
        vector_cores: Expected Vector core count per declared device.

    Returns:
        Raw columns and per-device physical-core rows, with source file hashes.

    Raises:
        ValueError: The declared scope or the exported data is incomplete or invalid.
    """
    if not device_ids or any(type(device) is not int or device < 0 for device in device_ids):
        raise ValueError("Provide nonnegative physical device IDs from the capture inventory")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("Each physical device ID must be supplied once")
    if any(type(count) is not int or count <= 0 for count in (cube_cores, vector_cores)):
        raise ValueError("Expected per-device core counts must be positive integers")

    devices = []
    for device in sorted(device_ids):
        database_dir = profile / f"device_{device}" / "sqlite"
        cube = _read_engine(database_dir / "aicore.db", cube_cores)
        vector = _read_engine(database_dir / "ai_vector_core.db", vector_cores)
        cube_ids = {row[0] for row in cube["events"]}
        vector_ids = {row[0] for row in vector["events"]}
        if cube_ids & vector_ids:
            raise ValueError(f"CUBE and Vector exports share physical core IDs for device {device}")
        devices.append({"device_id": device, "tables": [cube, vector]})
    return {"columns": list(COLUMNS), "devices": devices}


def main(argv: list[str] | None = None) -> int:
    """Validate the capture and write a complete raw-counter receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--devices", nargs="+", type=int, required=True)
    parser.add_argument("--cube-cores-per-die", type=int, required=True)
    parser.add_argument("--vector-cores-per-die", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = read_arithmetic_counters(args.profile, args.devices, args.cube_cores_per_die, args.vector_cores_per_die)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    payload = json.dumps(evidence, separators=(",", ":")) + "\n"
    with NamedTemporaryFile(mode="w", dir=args.out.parent, encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
