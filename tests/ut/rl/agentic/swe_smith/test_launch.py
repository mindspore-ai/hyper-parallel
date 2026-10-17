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
"""Controller lifecycle and mount boundaries without Docker or accelerators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from examples.swe_smith import launch


class TestLaunch(unittest.TestCase):
    """Launch orchestration preserves user files and owns only its controller."""

    def setUp(self) -> None:
        """Keep every test artifact in the task's temporary root."""
        self.root = Path(tempfile.mkdtemp(prefix="launch-ut-"))
        self.addCleanup(shutil.rmtree, self.root)
        environment = patch.dict(launch.os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        launch.os.environ.pop("HCCL_IF_BASE_PORT", None)
        launch.os.environ.pop("HCCL_NPU_SOCKET_PORT_RANGE", None)
        self.args = argparse.Namespace(data_dir=self.root / "data", images_path=self.root / "images.json",
                                       model_dir=self.root / "model", output_dir=self.root / "output",
                                       checkpoint_dir=self.root / "checkpoint", devices=(6, 7), docker_host=None,
                                       resume=None, image="controller:fixed", mode="train", instance_id=[],
                                       privileged_controller=False)
        self.args.output_dir.mkdir()

    def _script(self, repository: Path | None = None, *, ssh_directory: Path | None = None) -> str:
        return launch._script(self.args, repository or self.root / "repo", self.root / "tmp",
                              trainer_hccl=launch._trainer_hccl_environment(), ssh_directory=ssh_directory)

    def _controller_command(self, *, ssh_directory: Path | None = None) -> list[str]:
        return launch._controller_command(self.args, self.root / "repo", self.root / "tmp", "swe-smith-owned",
                                          ssh_directory=ssh_directory)

    def _arguments(self, *, output: Path | None = None, privileged: bool = False) -> list[str]:
        arguments = ["--data-dir", str(self.args.data_dir), "--images-path", str(self.args.images_path),
                     "--model-dir", str(self.args.model_dir), "--output-dir", str(output or self.args.output_dir),
                     "--checkpoint-dir", str(self.args.checkpoint_dir), "--devices", "6,7"]
        if privileged:
            arguments.append("--privileged-controller")
        return arguments

    def test_trainer_hccl_defaults_and_paired_host_override_are_saved_in_script(self) -> None:
        """Trainer ports remain independent of the rollout server and survive container creation."""
        self.assertEqual(launch._trainer_hccl_environment(),
                         {"HCCL_IF_BASE_PORT": "63200", "HCCL_NPU_SOCKET_PORT_RANGE": "63200-63299"})
        override = {"HCCL_IF_BASE_PORT": "63300", "HCCL_NPU_SOCKET_PORT_RANGE": "63300-63399"}
        with patch.dict(launch.os.environ, override):
            script = self._script()
        self.assertIn("export HCCL_IF_BASE_PORT=63300", script)
        self.assertIn("export HCCL_NPU_SOCKET_PORT_RANGE=63300-63399", script)

    def test_trainer_hccl_rejects_unpaired_malformed_and_out_of_contract_ports(self) -> None:
        """Host overrides obey the same port bounds and containment contract as RL configuration."""
        invalid = [
            {"HCCL_IF_BASE_PORT": "63200"},
            {"HCCL_NPU_SOCKET_PORT_RANGE": "63200-63299"},
            {"HCCL_IF_BASE_PORT": "63200;false", "HCCL_NPU_SOCKET_PORT_RANGE": "63200-63299"},
            {"HCCL_IF_BASE_PORT": "1024", "HCCL_NPU_SOCKET_PORT_RANGE": "1023-1100"},
            {"HCCL_IF_BASE_PORT": "65521", "HCCL_NPU_SOCKET_PORT_RANGE": "65520-65521"},
            {"HCCL_IF_BASE_PORT": "63200", "HCCL_NPU_SOCKET_PORT_RANGE": "63201-63299"},
            {"HCCL_IF_BASE_PORT": "63200", "HCCL_NPU_SOCKET_PORT_RANGE": "63299-63200"},
            {"HCCL_IF_BASE_PORT": "63200", "HCCL_NPU_SOCKET_PORT_RANGE": "63200 - 63299"},
        ]
        for override in invalid:
            with self.subTest(override=override), patch.dict(launch.os.environ, override):
                with self.assertRaises(ValueError):
                    launch._trainer_hccl_environment()
        with patch.dict(launch.os.environ,
                        {"HCCL_IF_BASE_PORT": "1024", "HCCL_NPU_SOCKET_PORT_RANGE": "1024-65520"}):
            self.assertEqual(launch._trainer_hccl_environment()["HCCL_IF_BASE_PORT"], "1024")

    def test_devices_reject_duplicates_and_negative_or_missing_ids(self) -> None:
        """Physical device lists must describe a distinct nonempty selection."""
        self.assertEqual(launch._devices("6, 7"), (6, 7))
        for value in ("", "1,1", "-1", "1,", "cpu"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                launch._devices(value)

    def test_local_controller_is_nonroot_and_mounts_only_selected_devices(self) -> None:
        """Rootless execution retains physical device boundaries and local socket access."""
        fake_stat = SimpleNamespace(st_gid=985, st_mode=stat.S_IFSOCK)
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat", return_value=fake_stat):
            command = self._controller_command()
        self.assertNotIn("--privileged", command)
        self.assertIn("/dev/davinci6:/dev/davinci6", command)
        self.assertIn("/dev/davinci7:/dev/davinci7", command)
        self.assertNotIn("/dev/davinci4:/dev/davinci4", command)
        self.assertIn("type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock", command)
        self.assertIn(f"type=bind,src={self.args.model_dir},dst={self.args.model_dir},readonly", command)
        self.assertEqual(command[-3:], ["controller:fixed", "bash", str(self.args.output_dir / "launch.sh")])

    def test_remote_mounts_agent_and_known_hosts_without_private_keys_or_docker_socket(self) -> None:
        """The controller authenticates SSH through the authorized agent only."""
        self.args.docker_host = "ssh://user@127.0.0.1:2233"
        agent = self.root / "agent.sock"
        fake_stat = SimpleNamespace(st_gid=985, st_mode=stat.S_IFSOCK)
        with patch.dict(launch.os.environ, {"SSH_AUTH_SOCK": str(agent)}), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "stat", return_value=fake_stat):
            command = self._controller_command(ssh_directory=self.root / ".ssh")
        mounts = [item for item in command if item.startswith("type=bind")]
        self.assertTrue(any(str(agent) in item and "readonly" in item for item in mounts))
        self.assertTrue(any(".ssh/known_hosts" in item for item in mounts))
        self.assertFalse(any("id_ed25519" in item or "id_rsa" in item or "docker.sock" in item for item in mounts))

    def test_script_preserves_cann_pythonpath_and_quotes_host_paths(self) -> None:
        """An external executable script carries startup environments and safe shell quoting."""
        repository = self.root / "repo with spaces $(unsafe)"
        script = self._script(repository)
        checked = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn('"${PYTHONPATH:-}"', script)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=0,1", script)
        self.assertIn("--nproc_per_node=2", script)
        self.assertIn("--no-deps --no-build-isolation", script)
        self.assertTrue(any(line.startswith("torchrun ") for line in script.splitlines()))
        self.assertEqual(script.splitlines()[-1], 'wait "$training_pid"')

    def test_privileged_controller_requires_opt_in_and_uses_physical_visibility(self) -> None:
        """Explicit privilege changes the trusted controller only and retains the selected physical IDs."""
        self.args.privileged_controller = True
        fake_stat = SimpleNamespace(st_gid=985, st_mode=stat.S_IFSOCK)
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "stat", return_value=fake_stat):
            command = self._controller_command()
        self.assertIn("--privileged", command)
        self.assertEqual(command[command.index("--user") + 1], f"0:{launch.os.getgid()}")
        script = self._script()
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=6,7", script)
        self.assertIn("umask 0002", script)
        self.assertIn("git config --global --add safe.directory", script)
        self.assertIn("PIP_DISABLE_PIP_VERSION_CHECK=1", script)
        self.assertIn("trap restore_output_ownership EXIT", script)
        ownership = next(line for line in script.splitlines() if "chown --recursive" in line)
        self.assertIn("--no-dereference", ownership)
        self.assertIn(str(self.args.output_dir), ownership)
        self.assertIn(str(self.args.checkpoint_dir), ownership)
        self.assertIn(str(self.root / "tmp"), ownership)
        self.assertNotIn(str(self.root / "repo"), ownership)
        self.assertNotIn(str(self.args.model_dir), ownership)
        self.assertNotIn("exec torchrun", script)

    def test_remote_root_script_uses_explicit_host_keys_and_agent_ssh_wrapper(self) -> None:
        """Root SSH must not search an unrelated root home or a broken system SSH configuration."""
        self.args.docker_host = "ssh://user@127.0.0.1:2233"
        self.args.privileged_controller = True
        script = self._script(ssh_directory=self.root / ".ssh")
        checked = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=False)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("exec /usr/bin/ssh -F /dev/null", script)
        self.assertIn(f"UserKnownHostsFile={self.root}/.ssh/known_hosts", script)
        self.assertIn("StrictHostKeyChecking=yes", script)
        self.assertIn("BatchMode=yes", script)
        self.assertNotIn("id_ed25519", script)

    def test_temporary_cleanup_failure_preserves_recorded_training_failure(self) -> None:
        """Forced termination can bypass the root ownership trap without masking the training failure."""
        manifest = self.args.output_dir / "run-manifest.json"
        manifest.write_text(json.dumps({"status": "failed", "exit_code": 42}))
        with patch.object(launch.shutil, "rmtree", side_effect=PermissionError("root-owned cache")):
            launch._cleanup_temporary(self.root / "tmp", self.args.output_dir)
        result = json.loads(manifest.read_text())
        self.assertEqual(result["exit_code"], 42)
        self.assertEqual(result["status"], "failed")
        self.assertIn("root-owned cache", result["temporary_cleanup_error"])

    def test_root_controller_rejects_existing_checkpoint_contents(self) -> None:
        """The ownership trap must never operate on an existing shared checkpoint tree."""
        self.args.checkpoint_dir.mkdir()
        checkpoint = self.args.checkpoint_dir / "existing.pt"
        checkpoint.write_text("keep")
        with patch.object(launch, "build_config") as builder:
            with self.assertRaisesRegex(ValueError, "dedicated checkpoint"):
                launch.main(self._arguments(privileged=True))
        builder.assert_not_called()
        self.assertEqual(checkpoint.read_text(), "keep")

    def test_inference_script_uses_both_splits_and_explicit_instance_filters(self) -> None:
        """Inference uses the shared prepared datasets and preserves requested identities."""
        self.args.mode = "infer"
        self.args.instance_id = ["task-one", "task-two"]
        script = self._script()
        self.assertIn("-m examples.swe_smith.infer", script)
        self.assertIn(str(self.args.data_dir / "train.parquet"), script)
        self.assertIn(str(self.args.data_dir / "validation.parquet"), script)
        self.assertIn("--instance-id task-one --instance-id task-two", script)

    def _run_fake(self, wait: subprocess.CompletedProcess | BaseException) -> dict:
        """Drive an owned fake controller through completion or interruption."""
        logger = MagicMock()
        logger.__enter__.return_value = logger
        started = subprocess.CompletedProcess(["run"], 0, stdout="owned-id\n", stderr="")
        removed = subprocess.CompletedProcess(["rm"], 0, stdout="swe-smith-owned\n", stderr="")
        manifest = {"status": "prepared"}
        results = [started, wait]
        if isinstance(wait, (subprocess.TimeoutExpired, KeyboardInterrupt)):
            results.append(subprocess.CompletedProcess(["stop"], 0, stdout="swe-smith-owned\n", stderr=""))
        results.append(removed)
        with patch.object(launch.subprocess, "run", side_effect=results) as run, \
                patch.object(launch.subprocess, "Popen", return_value=logger):
            try:
                launch._run_controller(["docker", "run"], "swe-smith-owned", self.args.output_dir, 10, manifest)
            finally:
                self.assertEqual(run.call_args.args[0][-3:], ["rm", "--force", "swe-smith-owned"])
                logger.wait.assert_called_once_with(timeout=15)
        return json.loads((self.args.output_dir / "run-manifest.json").read_text())

    def test_success_records_exit_and_removes_only_owned_controller(self) -> None:
        """A successful wait retains durable output before removing its own controller."""
        result = self._run_fake(subprocess.CompletedProcess(["wait"], 0, stdout="0\n", stderr=""))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["controller_id"], "owned-id")

    def test_controller_failure_preserves_exit_code_and_cleans_up(self) -> None:
        """A failed model process cannot be reported as a successful launcher run."""
        with self.assertRaisesRegex(RuntimeError, "status 42"):
            self._run_fake(subprocess.CompletedProcess(["wait"], 0, stdout="42\n", stderr=""))
        result = json.loads((self.args.output_dir / "run-manifest.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 42)

    def test_timeout_is_explicit_and_still_cleans_up(self) -> None:
        """A deadline violation records its cause and removes only the owned controller."""
        with self.assertRaisesRegex(TimeoutError, "exceeded 10 seconds"):
            self._run_fake(subprocess.TimeoutExpired(["wait"], 10))
        result = json.loads((self.args.output_dir / "run-manifest.json").read_text())
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["controller_stop_returncode"], 0)
        self.assertIn("exact hyper-rl.run-id", result["sandbox_cleanup_warning"])

    def test_sigterm_becomes_interrupt_and_restores_handler_after_cleanup(self) -> None:
        """Termination reaches bounded cleanup and does not retain a process-global handler."""
        with patch.object(launch.signal, "signal"):
            with self.assertRaisesRegex(KeyboardInterrupt, "Termination signal"):
                launch._interrupt_on_signal(signal.SIGTERM, None)
        before = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            self._run_fake(KeyboardInterrupt("SIGTERM"))
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)
        result = json.loads((self.args.output_dir / "run-manifest.json").read_text())
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["controller_stop_returncode"], 0)

    def test_failed_container_start_cleans_up_and_preserves_daemon_error(self) -> None:
        """Docker may create a stopped container before reporting a startup failure."""
        failure = subprocess.CalledProcessError(1, ["docker", "run"], stderr="device setup failed")
        removed = subprocess.CompletedProcess(["rm"], 0, stdout="swe-smith-owned\n", stderr="")
        manifest = {"status": "prepared"}
        with patch.object(launch.subprocess, "run", side_effect=[failure, removed]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                launch._run_controller(["docker", "run"], "swe-smith-owned", self.args.output_dir, 10, manifest)
        self.assertEqual(run.call_args.args[0][-3:], ["rm", "--force", "swe-smith-owned"])
        result = json.loads((self.args.output_dir / "run-manifest.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIn("device setup failed", result["error"])

    def test_existing_output_is_never_overwritten(self) -> None:
        """Reject populated destinations before invoking configuration or Docker."""
        existing = self.args.output_dir / "keep.txt"
        existing.write_text("keep")
        with patch.object(launch, "build_config") as builder, patch.object(launch.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "new or empty"):
                launch.main(self._arguments())
        builder.assert_not_called()
        run.assert_not_called()
        self.assertEqual(existing.read_text(), "keep")

    def test_invalid_configuration_creates_no_output_or_checkpoint_directories(self) -> None:
        """Input overlap and identity validation must precede any host directory writes."""
        output = self.root / "new-output"
        with patch.object(launch, "build_config", side_effect=ValueError("input overlap")), \
                patch.object(launch.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "input overlap"):
                launch.main(self._arguments(output=output))
        self.assertFalse(output.exists())
        self.assertFalse(self.args.checkpoint_dir.exists())
        run.assert_not_called()

    def test_main_shares_resolved_endpoint_and_validated_environment(self) -> None:
        """Mounts, the launch script and manifest consume the same resolved host inputs."""
        config = {"train": {"accelerator": {"dp_shard": 2}},
                  "agentic": {"codex": {"workspace": {"docker_host": launch.LOCAL_DOCKER_HOST}}}}
        environment = {"HCCL_IF_BASE_PORT": "63200", "HCCL_NPU_SOCKET_PORT_RANGE": "63200-63299"}
        image = subprocess.CompletedProcess(["inspect"], 0, stdout="sha256:controller\n", stderr="")
        with patch.object(launch, "build_config", return_value=config), \
                patch.object(launch, "_trainer_hccl_environment", return_value=environment) as hccl, \
                patch.object(launch, "_controller_command", return_value=["docker", "run"]) as command, \
                patch.object(launch, "_script", return_value="#!/bin/bash\nexit 0\n") as script, \
                patch.object(launch.subprocess, "run", return_value=image), \
                patch.object(launch, "_run_controller"):
            launch.main(self._arguments())
        hccl.assert_called_once_with()
        self.assertIsNone(command.call_args.kwargs["ssh_directory"])
        self.assertIsNone(script.call_args.kwargs["ssh_directory"])
        self.assertIs(script.call_args.kwargs["trainer_hccl"], environment)
        self.assertEqual(command.call_args.args[0].docker_host, launch.LOCAL_DOCKER_HOST)
        temporary = command.call_args.args[2]
        self.assertEqual(script.call_args.args[2], temporary)
        self.assertFalse(temporary.exists())
        manifest = json.loads((self.args.output_dir / "run-manifest.json").read_text())
        self.assertEqual(manifest["sandbox_docker_host"], launch.LOCAL_DOCKER_HOST)
        self.assertEqual(manifest["temporary_dir"], str(temporary))


if __name__ == "__main__":
    unittest.main()
