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
"""Representative failure propagation, container ownership and dependency-patch regressions."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import rl
from rl.trainer import SyncTrainer
from rl.utils.failure_reporting import report_failure

RL_ROOT = Path(rl.__file__).resolve().parent.parent
SCRIPTS = RL_ROOT / "examples/gsm8k/scripts"
PATCHES = RL_ROOT / "docker/patches"


def load_tool(name: str, directory: Path = SCRIPTS) -> ModuleType:
    """Load host utilities without starting devices or containers.

    Args:
        name: Tool filename without its extension.
        directory: Directory containing the tool.

    Returns:
        The imported tool module.
    """
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFailureReporting(unittest.TestCase):
    """Keep primary errors visible and limit host cleanup to the owned container."""

    def test_original_failure_is_published_before_cleanup(self) -> None:
        """Cleanup errors must neither overwrite rank evidence nor replace the original exception."""
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"HYPER_RL_FAILURE_DIR": directory, "RANK": "3"},
        ):
            target = Path(directory) / "failure-rank-3.json"
            trainer = object.__new__(SyncTrainer)
            trainer.state = SimpleNamespace(global_step=0)
            original = RuntimeError("actor OOM")
            trainer._train_and_checkpoint = Mock(side_effect=original)

            def cleanup() -> None:
                """Check publication ordering and inject a secondary failure."""
                self.assertTrue(target.exists())
                report_failure(ValueError("cleanup failed"), stage="cleanup")
                raise ValueError("cleanup failed")

            trainer._cleanup = cleanup
            with patch("rl.trainer.dist.barrier") as barrier, self.assertRaises(RuntimeError) as caught:
                trainer.train()
            self.assertIs(caught.exception, original)
            barrier.assert_not_called()
            record = json.loads(target.read_text())
            self.assertEqual((record["stage"], record["step"]), ("training", 0))
            self.assertIn("actor OOM", record["traceback"])
            self.assertNotIn("cleanup failed", record["traceback"])
            self.assertEqual(load_tool("monitor_8npu_run").read_failure(Path(directory))["record"], record)

    def test_stop_and_kill_are_bounded_and_container_specific(self) -> None:
        """Only the owned ID may be stopped; timeout escalates and a live survivor is an error."""
        monitor = load_tool("monitor_8npu_run")
        identity = "a" * 64
        for live, survives in ((False, False), (True, False), (True, True)):
            responses = [subprocess.TimeoutExpired("docker stop", 20) if live else SimpleNamespace(returncode=0),
                         SimpleNamespace(stdout=str(live).lower())]
            if live:
                responses.append(SimpleNamespace(returncode=0))
            responses.append(SimpleNamespace(stdout=str(survives).lower()))
            with self.subTest(live=live, survives=survives), patch.object(
                monitor.subprocess, "run", side_effect=responses,
            ) as run:
                if survives:
                    with self.assertRaisesRegex(RuntimeError, "did not stop"):
                        monitor.stop_container(identity)
                else:
                    monitor.stop_container(identity)
                self.assertEqual(any(call.args[0][1] == "kill" for call in run.call_args_list), live)
                self.assertTrue(all(call.args[0][-1] == identity and call.kwargs["timeout"] <= 20
                                    for call in run.call_args_list))
        with patch.object(monitor.subprocess, "run") as run, self.assertRaises(ValueError):
            monitor.stop_container("all-containers")
        run.assert_not_called()

    def test_sampler_patch_is_exact_idempotent_and_rejects_unknown_sources(self) -> None:
        """A patch never silently applies to an unverified dependency version."""
        sampler = load_tool("patch_sampler", PATCHES)
        original = b"def sample():\n    torch.npu.current_stream().wait_stream(global_stream())\n    return q\n"
        fixed = original.replace(b"    return q", b"    q.record_stream(torch.npu.current_stream())\n    return q")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sampler.py"
            for prefix in ("", "VERL_"):
                target.write_bytes(original)
                with patch.multiple(sampler, **{prefix + "ORIGINAL_SHA256": hashlib.sha256(original).hexdigest(),
                                               prefix + "PATCHED_SHA256": hashlib.sha256(fixed).hexdigest()}):
                    self.assertEqual(sampler.apply_patch(target)["status"], "applied")
                    self.assertEqual(sampler.apply_patch(target)["status"], "already_applied")
                self.assertEqual(target.read_bytes(), fixed)
            target.write_bytes(b"unknown implementation")
            with self.assertRaisesRegex(ValueError, "Unknown sampler"):
                sampler.apply_patch(target)
            self.assertEqual(target.read_bytes(), b"unknown implementation")

    @staticmethod
    def process_stat(state: str, started: str = "100") -> str:
        """Build a proc stat fixture whose command contains spaces and nested parentheses."""
        return "123 (worker (retired)) " + " ".join([state] + ["0"] * 18 + [started])

    def test_process_identity_handles_exit_reuse_and_unreadable_state(self) -> None:
        """Only live processes with stable ownership can count as experiment workers."""
        self.monitor = load_tool("monitor_8npu_run")
        stat = self.process_stat
        cases = [([stat("S"), "0::/docker/owned-id", stat("R")], "owned"),
                 ([stat("S"), "0::/docker/other-id", stat("R")], "external"),
                 ([stat("Z")], "exited"), ([stat("S"), "0::/", stat("Z")], "exited"),
                 ([stat("S"), "0::/docker/owned-id", stat("S", "200")], "unknown"),
                 (PermissionError(), "unknown"), (ValueError(), "unknown"), (FileNotFoundError(), "gone")]
        for reads, expected in cases:
            with self.subTest(expected=expected, reads=reads), patch.object(
                self.monitor.Path, "read_text", side_effect=reads,
            ):
                self.assertEqual(self.monitor.process_identity("123", "owned-id")["owner"], expected)
