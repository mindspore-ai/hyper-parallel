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
"""CPU regression checks for passive SHMEM loader diagnostics."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.common import shmem_loader_evidence as evidence


# Exercise private parsers and caches without loading native libraries in unit tests.
# pylint: disable=protected-access
class TestShmemLoaderEvidence(unittest.TestCase):
    """Keep diagnostics observable without changing native loading or test failures."""

    def setUp(self) -> None:
        """Isolate rank snapshots and library metadata across test cases."""
        self.directory = Path(tempfile.mkdtemp(prefix="hp-shmem-loader-ut-"))
        self.addCleanup(shutil.rmtree, self.directory)
        for cache in (evidence._SNAPSHOTS, evidence._LIBRARIES, evidence._IDENTITIES):
            cache.clear()
            self.addCleanup(cache.clear)

    def test_maps_keep_paths_inodes_and_deleted_mappings(self) -> None:
        """Feature: passive library identity.
        Description: Parse repeated mappings, spaces and a deleted library.
        Expectation: Retain the actual path and inode without loading a library.
        """
        path = "/opt/CANN with spaces/libascendcl.so"
        maps = (
            f"1000-2000 r-xp 000000 00:01 42 {path}\n"
            f"2000-3000 r--p 001000 00:01 42 {path} (deleted)\n"
            "3000-4000 rw-p 000000 00:00 0 [heap]\n"
            "4000-5000 r-xp 000000 00:01 43 /lib/libunrelated.so\n"
        )
        result = evidence._mapped_libraries(maps)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["path"], result[0]["inode"]), (path, 42))
        self.assertEqual(len(result[0]["mappings"]), 2)

    def test_dynsym_distinguishes_defined_and_undefined_versioned_symbols(self) -> None:
        """Feature: Runtime export evidence.
        Description: Parse defined and undefined occurrences of one Runtime interface.
        Expectation: Preserve definition state, version, binding and visibility.
        """
        result = evidence._symbol_definitions(
            "1: 00000010 16 FUNC GLOBAL DEFAULT 12 rtDeviceGetBareTgid@@CANN_1\n"
            "2: 00000000 0 FUNC GLOBAL DEFAULT UND rtDeviceGetBareTgid\n"
            "3: 00000020 16 FUNC GLOBAL DEFAULT 12 unrelated\n"
        )
        entries = result["rtDeviceGetBareTgid"]
        self.assertEqual([item["defined"] for item in entries], [True, False])
        self.assertEqual(entries[0]["name"], "rtDeviceGetBareTgid@@CANN_1")
        self.assertEqual(entries[0]["visibility"], "DEFAULT")
        self.assertEqual(set(result), {"rtDeviceGetBareTgid"})

    def test_optional_git_failure_is_explicit_and_unrelated_error_propagates(self) -> None:
        """Feature: packaged-source identity.
        Description: Query revision without Git metadata and with an unexpected error.
        Expectation: Mark unavailable Git metadata while preserving unexpected errors.
        """
        error = subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])
        with patch.object(evidence.subprocess, "check_output", side_effect=error):
            result = evidence._source_identity()
        self.assertIsNone(result["source_sha"])
        self.assertIn("128", result["source_sha_error"])
        self.assertIsNone(result["worker_sha256"])
        self.assertIn("not configured", result["worker_sha256_error"])
        with patch.object(evidence.subprocess, "check_output", side_effect=ValueError("unexpected")):
            with self.assertRaisesRegex(ValueError, "unexpected"):
                evidence._source_identity()

    def test_source_identity_hashes_explicit_worker_and_launcher_paths(self) -> None:
        """Feature: relocated diagnostic source identity.
        Description: Supply real source files outside the collector directory.
        Expectation: Record their hashes and propagate an invalid configured path.
        """
        worker, launcher = self.directory / "worker.py", self.directory / "launcher.py"
        worker.write_bytes(b"worker source")
        launcher.write_bytes(b"launcher source")
        with patch.object(Path, "cwd", return_value=self.directory), pytest.MonkeyPatch.context() as monkeypatch:
            with evidence.loader_evidence_session(monkeypatch, worker_source=worker, launcher_source=launcher):
                result = evidence._source_identity()
                self.assertEqual(result["worker_sha256"], hashlib.sha256(worker.read_bytes()).hexdigest())
                self.assertEqual(result["launcher_sha256"], hashlib.sha256(launcher.read_bytes()).hexdigest())
                self.assertEqual(result["worker_source"], str(worker))
                worker.unlink()
                with self.assertRaises(FileNotFoundError):
                    evidence._source_identity()

    def test_success_replays_rank_evidence_and_omits_unrelated_environment(self) -> None:
        """Feature: successful CI evidence.
        Description: Capture a successful worker and return through the launcher.
        Expectation: Replay its evidence without dumping unrelated environment values.
        """
        output = io.StringIO()
        with patch.object(Path, "cwd", return_value=self.directory), patch.dict(
            os.environ, {"RANK": "0", "PRIVATE_TOKEN": "do-not-print"}, clear=True
        ), patch.object(evidence, "_snapshot", return_value={"stage": "before", "libraries": []}), patch.object(
            evidence, "_source_identity", return_value={"source_sha": None}
        ), patch("sys.stdout", output), pytest.MonkeyPatch.context() as monkeypatch:
            with evidence.loader_evidence_session(monkeypatch) as directory:
                evidence.capture_loader_snapshot("before")
                evidence.capture_loader_snapshot("before")
                files = list(directory.glob("rank-*.json"))
                self.assertEqual(len(files), 1)
                self.assertEqual(len(json.loads(files[0].read_text())["snapshots"]), 1)
        self.assertIn(evidence._MARKER, output.getvalue())
        self.assertNotIn("do-not-print", output.getvalue())

    def test_diagnostic_failure_does_not_replace_worker_exception(self) -> None:
        """Feature: original failure preservation.
        Description: Fail collection and the worker in the same diagnostic session.
        Expectation: Report collection failure while retaining the worker's exception.
        """
        error = RuntimeError("original SHMEM failure")
        with patch.object(Path, "cwd", return_value=self.directory), patch.object(
            evidence, "_snapshot", side_effect=OSError("maps unavailable")
        ), patch.object(evidence, "_source_identity", return_value={}), patch(
            "sys.stderr", io.StringIO()
        ) as stderr, patch("sys.stdout", io.StringIO()), pytest.MonkeyPatch.context() as monkeypatch:
            with self.assertRaises(RuntimeError) as raised:
                with evidence.loader_evidence_session(monkeypatch):
                    evidence.capture_loader_snapshot("failure")
                    raise error
        self.assertIs(raised.exception, error)
        self.assertIn("diagnostic_error", stderr.getvalue())
        self.assertIn("maps unavailable", stderr.getvalue())

    def test_missing_readelf_is_recorded_as_incomplete_evidence(self) -> None:
        """Feature: diagnostic tool availability.
        Description: Inspect a mapped file without readelf installed.
        Expectation: Retain the hash and report the missing ELF inspection tool.
        """
        library = self.directory / "libruntime.so"
        library.write_bytes(b"not-an-ELF")
        with patch.object(evidence.shutil, "which", return_value=None):
            identity = evidence._library_identity(library)
        self.assertTrue(identity["sha256"])
        self.assertEqual(identity["needed"], [])
        self.assertTrue(all("readelf is not installed" in error for error in identity["errors"]))
        self.assertFalse(identity["file_changed_during_collection"])

    def test_explicit_export_failure_keeps_success_distinct_from_worker_failure(self) -> None:
        """Feature: diagnostic artifact export.
        Description: Request failure-log export after a successful diagnostic body.
        Expectation: Record worker success separately from the deliberate export error.
        """
        output = io.StringIO()
        with patch.object(Path, "cwd", return_value=self.directory), patch.object(
            evidence, "_report_evidence"
        ), patch("sys.stdout", output), pytest.MonkeyPatch.context() as monkeypatch:
            with self.assertRaisesRegex(RuntimeError, "SHMEM_DIAGNOSTIC_EXPORT"):
                with evidence.loader_evidence_session(monkeypatch, export_on_success=True):
                    pass
        record = json.loads(output.getvalue().removeprefix(evidence._MARKER))
        self.assertTrue(record["precision_workers_succeeded"])
        self.assertTrue(record["export_on_success"])

    def test_export_option_never_replaces_an_actual_worker_failure(self) -> None:
        """Feature: native failure preservation during export.
        Description: Fail the worker while deliberate export is requested.
        Expectation: Preserve the native failure instead of reporting export success.
        """
        error = RuntimeError("original SHMEM initialization failure")
        with patch.object(Path, "cwd", return_value=self.directory), patch.object(
            evidence, "_report_evidence"
        ), patch("sys.stdout", io.StringIO()) as output, pytest.MonkeyPatch.context() as monkeypatch:
            with self.assertRaises(RuntimeError) as raised:
                with evidence.loader_evidence_session(monkeypatch, export_on_success=True):
                    raise error
        self.assertIs(raised.exception, error)
        record = json.loads(output.getvalue().removeprefix(evidence._MARKER))
        self.assertFalse(record["precision_workers_succeeded"])

    def test_loader_trace_retains_bounded_symbol_and_library_evidence(self) -> None:
        """Feature: bounded console evidence.
        Description: Extract a long loader trace containing the failing interface.
        Expectation: Report full match counts and preserve the final lookup result.
        """
        path = self.directory / "loader.123"
        path.write_text("libascendcl.so search\n" * 200 + "undefined symbol: rtDeviceGetBareTgid\n")
        result = evidence._trace_excerpt(path)
        self.assertEqual(result["library_lines_total"], 200)
        self.assertEqual(len(result["library_lines"]), 128)
        self.assertEqual(result["symbol_lines"], ["undefined symbol: rtDeviceGetBareTgid"])

    def test_importing_evidence_keeps_frameworks_unloaded(self) -> None:
        """Feature: lightweight shared diagnostics.
        Description: Import the collector in a fresh interpreter.
        Expectation: Leave Torch, Torch-NPU and HyperParallel unloaded.
        """
        script = (
            "import sys; import tests.common.shmem_loader_evidence; "
            "assert not {'torch', 'torch_npu', 'hyper_parallel'} & sys.modules.keys()"
        )
        subprocess.run(
            [sys.executable, "-c", script], cwd=Path(evidence.__file__).resolve().parents[2],
            check=True, capture_output=True, text=True,
        )

    def test_collector_imports_from_a_common_only_test_package(self) -> None:
        """Feature: packaged UT import boundary.
        Description: Import the collector from a package without the ST tree.
        Expectation: Resolve the isolated common module without loading frameworks.
        """
        package = self.directory / "tests"
        common = package / "common"
        common.mkdir(parents=True)
        (package / "__init__.py").touch()
        (common / "__init__.py").touch()
        shutil.copyfile(evidence.__file__, common / "shmem_loader_evidence.py")
        script = (
            "import sys; from tests.common import shmem_loader_evidence as evidence; "
            f"assert evidence.__file__.startswith({str(common)!r}); "
            "assert not {'torch', 'torch_npu', 'hyper_parallel'} & sys.modules.keys()"
        )
        subprocess.run(
            [sys.executable, "-c", script], cwd=self.directory, check=True, capture_output=True, text=True,
        )
