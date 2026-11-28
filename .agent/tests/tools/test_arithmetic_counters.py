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
"""Check coverage of CANN physical-core counter exports with SQLite fixtures."""

from contextlib import closing, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import runpy
import sqlite3
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[3] / ".agent/skills/profiling/scripts/arithmetic_counters.py"
COUNTERS = runpy.run_path(str(SCRIPT))


def _export(root: Path, device: int, database: str, core_ids: list[int]) -> Path:
    """Write one physical engine export with known register values."""
    path = root / f"device_{device}" / "sqlite" / database
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        columns = ", ".join(f"{column} INTEGER" for column in COUNTERS["COLUMNS"])
        connection.execute(f"CREATE TABLE EventCount ({columns})")
        rows = [(core, 2, 3, 4, 5, 6, 7, 8) for core in core_ids]
        connection.executemany("INSERT INTO EventCount VALUES (?,?,?,?,?,?,?,?)", rows)
        connection.commit()
    return path


def _command(root: Path, output: Path) -> list[str]:
    """Request the full scope of the two-device fixture."""
    return [str(root), "--devices", "0", "1", "--cube-cores-per-die", "1", "--vector-cores-per-die", "2",
            "--out", str(output)]


class TestArithmeticCounters(unittest.TestCase):
    """Verify complete exports and reject partial or duplicate core coverage."""

    def test_offline_export_can_supply_a_missing_device(self):
        """Reject incomplete output and then accept the finalized re-export."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "counters.json"
            for device in (0, 1):
                _export(root, device, "aicore.db", [0])
            initial = _export(root, 0, "ai_vector_core.db", [4, 2])
            command = _command(root, output)
            with redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as caught:
                COUNTERS["main"](command)
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("Missing counter export", error.getvalue())
            self.assertFalse(output.exists())

            restored = _export(root, 1, "ai_vector_core.db", [4, 2])
            self.assertEqual(COUNTERS["main"](command), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([entry["device_id"] for entry in result["devices"]], [0, 1])
            expected = [[2, 2, 3, 4, 5, 6, 7, 8], [4, 2, 3, 4, 5, 6, 7, 8]]
            for device, path in zip(result["devices"], (initial, restored)):
                vector = device["tables"][1]
                self.assertEqual(vector["events"], expected)
                self.assertEqual(vector["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_duplicate_physical_core_is_rejected(self):
        """Equal row counts cannot hide duplicate physical core identities."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _export(root, 0, "aicore.db", [0])
            _export(root, 0, "ai_vector_core.db", [4, 4])
            with self.assertRaisesRegex(ValueError, "2 distinct physical cores"):
                COUNTERS["read_arithmetic_counters"](root, [0], 1, 2)

    def test_aggregate_core_id_is_rejected(self):
        """A profiler rollup row cannot be counted as a physical core."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _export(root, 0, "aicore.db", [-1])
            _export(root, 0, "ai_vector_core.db", [2, 4])
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                COUNTERS["read_arithmetic_counters"](root, [0], 1, 2)
