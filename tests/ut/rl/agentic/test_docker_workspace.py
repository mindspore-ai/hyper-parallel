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
"""Docker lifecycle, bounded command and non-extracting snapshot contracts."""

import asyncio
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from rl.agentic.envs.docker_workspace import (
    CommandResult, DockerWorkspace, InvalidSubmissionError, WorkspaceOutputLimitError,
    WorkspaceTimeoutError, _command, _fixture_archive, _snapshot_files, read_snapshot,
)


def _archive(entries: list[tuple[str, bytes, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(content) if kind == tarfile.REGTYPE else 0
            archive.addfile(member, io.BytesIO(content) if member.size else None)
    return buffer.getvalue()


class TestSnapshot(unittest.TestCase):
    """Never extract candidate-controlled paths onto the host."""

    def test_plain_snapshot_and_budgets(self) -> None:
        """Preserve exact bytes while enforcing both semantic budgets."""
        raw = _archive([("src/file.py", b"content", tarfile.REGTYPE)])
        self.assertEqual(_snapshot_files(raw, 7, 1), {"src/file.py": b"content"})
        with self.assertRaises(InvalidSubmissionError):
            _snapshot_files(raw, 6, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.tar"
            path.write_bytes(raw)
            self.assertEqual(read_snapshot(path), {"src/file.py": b"content"})
            path.write_bytes(raw + b"x" * 30000)
            with self.assertRaisesRegex(InvalidSubmissionError, "transport"):
                read_snapshot(path, max_bytes=7, max_files=1)

    def test_reject_paths_types_duplicates_and_conflicts(self) -> None:
        """Reject links, alternate spellings, file parents and member overflow."""
        invalid = [
            [(name, b"x", tarfile.REGTYPE)]
            for name in ("../escape", "/absolute", "src/../escape", "src\\escape")
        ]
        invalid.extend([[("link", b"", kind)] for kind in
                        (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE)])
        invalid.extend([
            [("src/x", b"1", tarfile.REGTYPE), ("./src/x", b"2", tarfile.REGTYPE)],
            [("src", b"1", tarfile.REGTYPE), ("src/x", b"2", tarfile.REGTYPE)],
        ])
        for entries in invalid:
            with self.subTest(entries=entries), self.assertRaises(InvalidSubmissionError):
                _snapshot_files(_archive(entries), 100, 100)
        with self.assertRaises(InvalidSubmissionError):
            _snapshot_files(_archive([("a", b"", tarfile.DIRTYPE), ("b", b"", tarfile.DIRTYPE)]), 100, 1)

    def test_reject_truncated_and_ambiguous_archives(self) -> None:
        """Tarfile's permissive EOF handling must not accept incomplete transport."""
        raw = _archive([("a", b"x", tarfile.REGTYPE)])
        for invalid in (raw[:1024], raw[:513], raw + b"unparsed", b"not tar"):
            with self.subTest(size=len(invalid)), self.assertRaises(InvalidSubmissionError):
                _snapshot_files(invalid, 100, 100)

    def test_copy_owns_files_and_rejects_links(self) -> None:
        """Copied files use container root, not the controller's numeric UID."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("pass")
            with tarfile.open(fileobj=io.BytesIO(_fixture_archive(root))) as archive:
                self.assertTrue(all(member.uid == member.gid == 0 for member in archive))
            (root / "link").symlink_to("source.py")
            with self.assertRaisesRegex(ValueError, "link"):
                _fixture_archive(root)


class TestBoundedCommand(unittest.IsolatedAsyncioTestCase):
    """Use actual short CPU subprocesses to exercise pipe cleanup."""

    async def test_runtime_selection_preserves_default_and_isolation(self) -> None:
        """CPU tasks can select runc without changing callers that inherit the daemon default."""
        for runtime in (None, "runc"):
            with self.subTest(runtime=runtime):
                workspace = DockerWorkspace("sha256:" + "a" * 64, read_only=True, runtime=runtime)
                workspace._docker = AsyncMock(return_value=(CommandResult(0, "", ""), b""))
                await workspace.start()
                create = workspace._docker.await_args_list[0].args
                if runtime is None:
                    self.assertNotIn("--runtime", create)
                else:
                    self.assertEqual(create[create.index("--runtime") + 1], runtime)
                self.assertIn("--read-only", create)
                self.assertEqual(create[create.index("--network") + 1], "none")
                self.assertEqual(create[create.index("--cap-drop") + 1], "ALL")
        for runtime in ("", "runc --privileged", "/usr/bin/runc", 1):
            with self.subTest(invalid_runtime=runtime), self.assertRaises(ValueError):
                DockerWorkspace("task-image", runtime=runtime)

    async def test_readonly_candidate_profile_keeps_mounts_and_numeric_identity(self) -> None:
        """A host broker can use a pinned image without changing repository workspace defaults."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = DockerWorkspace("sha256:" + "a" * 64, user="1005:1001", read_only=True,
                                        mounts=((root, "/work", False), (root, "/work/articles", True)))
            workspace._docker = AsyncMock(return_value=(CommandResult(0, "", ""), b""))
            await workspace.start()
        create = workspace._docker.await_args_list[0].args
        self.assertEqual(create[0], "create")
        self.assertIn("--read-only", create)
        self.assertIn("1005:1001", create)
        self.assertIn(f"{root}:/work/articles:ro", create)
        self.assertNotIn("--privileged", create)
        self.assertEqual(len(workspace._docker.await_args_list), 2)
        for user in (":", "user:group", "0", "-1:0"):
            with self.subTest(user=user), self.assertRaises(ValueError):
                DockerWorkspace("task-image", user=user)

    async def test_remote_endpoint_is_used_without_local_fallback(self) -> None:
        """Remote management errors propagate without consulting the local daemon."""
        workspace = DockerWorkspace("task-image", docker_host="ssh://user@127.0.0.1:2233")
        with patch("rl.agentic.envs.docker_workspace._command", new_callable=AsyncMock) as command:
            command.return_value = (CommandResult(1, "", "connection lost"), b"", False, False)
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                await workspace.start()
        command.assert_awaited_once()
        self.assertEqual(command.call_args.args[0][:3],
                         ["docker", "--host", "ssh://user@127.0.0.1:2233"])
        self.assertEqual(DockerWorkspace("task-image").docker_command("exec", "name"),
                         ["docker", "exec", "name"])

    async def test_output_timeout_and_stdin_are_bounded(self) -> None:
        """Output overflow and blocked stdin both terminate promptly."""
        result, raw, timed_out, overflow = await _command(
            [sys.executable, "-c", "import sys;sys.stdout.write('x'*10000)"], limit=32,
        )
        self.assertEqual(len(raw), 32)
        self.assertTrue(overflow)
        self.assertFalse(timed_out)
        self.assertEqual(result.stderr, "")
        _, _, timed_out, _ = await _command(
            [sys.executable, "-c", "import time;time.sleep(10)"], stdin=b"x" * 1000000, timeout=0.05,
        )
        self.assertTrue(timed_out)

    async def test_cancellation_propagates(self) -> None:
        """Cancellation is not converted into an apparently successful result."""
        task = asyncio.create_task(_command([sys.executable, "-c", "import time;time.sleep(10)"]))
        await asyncio.sleep(0.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)


class TestDockerWorkspace(unittest.IsolatedAsyncioTestCase):
    """Check ownership and error classification without a Docker daemon."""

    async def test_export_requires_confirmed_stop_and_keeps_transport_failure(self) -> None:
        """Only a stopped container may export; transport overflow is infrastructure."""
        workspace = DockerWorkspace("fixture")
        workspace._created = True
        workspace._docker = AsyncMock(return_value=(CommandResult(0, "true", ""), b""))
        with self.assertRaisesRegex(RuntimeError, "running"):
            await workspace.export(Path("unused"))
        error = RuntimeError("Docker cp management output exceeded its transport budget")
        workspace._docker.side_effect = [(CommandResult(0, "false", ""), b""), error]
        with self.assertRaisesRegex(RuntimeError, "transport"):
            await workspace.export(Path("unused"))

    async def test_management_budgets_are_not_candidate_failures(self) -> None:
        """Docker inspect/copy budget failures cannot be scored as bad candidate code."""
        for timed_out, overflow in ((True, False), (False, True)):
            with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                    return_value=(CommandResult(-1, "", ""), b"", timed_out, overflow))):
                with self.assertRaisesRegex(RuntimeError, "management") as caught:
                    await DockerWorkspace("task-image")._docker("inspect", "owned")
                self.assertNotIsInstance(caught.exception, WorkspaceOutputLimitError)
                self.assertNotIsInstance(caught.exception, WorkspaceTimeoutError)

    async def test_exec_budget_and_daemon_failures_stop_workspace(self) -> None:
        """Command budgets stop descendants; Docker failures never become candidate results."""
        workspace = DockerWorkspace("fixture")
        workspace._running = True
        workspace.stop = AsyncMock()
        for timed_out, overflow, expected in [(True, False, WorkspaceTimeoutError),
                                               (False, True, WorkspaceOutputLimitError)]:
            with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                    return_value=(CommandResult(-1, "partial", ""), b"", timed_out, overflow))):
                with self.assertRaises(expected):
                    await workspace.exec(["python", "solution.py"])
        self.assertEqual(workspace.stop.await_count, 2)
        workspace._docker = AsyncMock(return_value=(
            CommandResult(0, '{"Running":true,"OOMKilled":false}', ""), b""))
        for status in (125, 126, 127, 137):
            with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                    return_value=(CommandResult(status, "", "daemon"), b"", False, False))):
                with self.assertRaisesRegex(RuntimeError, "Docker exec"):
                    await workspace.exec(["python", "solution.py"])

    async def test_close_owned_container_is_idempotent(self) -> None:
        """Cleanup filters both identity labels and never prunes foreign containers."""
        workspace = DockerWorkspace("fixture", run_id="test-run")
        workspace._created = True
        workspace._docker = AsyncMock(return_value=(CommandResult(0, "owned-id", ""), b""))
        await workspace.close()
        await workspace.close()
        self.assertEqual(workspace._docker.await_count, 2)
        args = workspace._docker.await_args_list[0].args
        self.assertIn("label=hyper-rl.owner=code-agent", args)
        self.assertIn("label=hyper-rl.run-id=test-run", args)
        self.assertEqual(workspace._docker.await_args_list[1].args, ("rm", "--force", workspace.name))

    async def test_controller_stop_requires_matching_exit_and_container_state(self) -> None:
        """Preserve stop output without masking OOM, daemon errors or active containers."""
        for signaled, running, oom, code, accepted in (
                (True, False, False, 137, True), (True, False, False, 143, True),
                (False, False, False, 137, False), (True, False, True, 137, False),
                (True, True, False, 137, False), (True, False, False, 1, False)):
            with self.subTest(signaled=signaled, running=running, oom=oom, code=code):
                workspace = DockerWorkspace("fixture")
                workspace._running = True
                workspace.stop = AsyncMock()
                state = '{"Running":%s,"OOMKilled":%s}' % (str(running).lower(), str(oom).lower())
                workspace._docker = AsyncMock(return_value=(CommandResult(0, state, ""), b""))
                event = asyncio.Event()
                if signaled:
                    event.set()
                    workspace._stop_task = asyncio.create_task(asyncio.sleep(0))
                    await workspace._stop_task
                result = CommandResult(code, "real stdout", "real stderr")
                with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                        return_value=(result, b"", False, False))):
                    if accepted:
                        self.assertEqual(await workspace.exec(["codex"], controller_stop=event), result)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Docker exec"):
                            await workspace.exec(["codex"], controller_stop=event)

    async def test_controller_stop_waits_for_real_stop_before_inspection(self) -> None:
        """A signal exit arriving before Docker stop completes cannot observe a transient Running state."""
        workspace = DockerWorkspace("fixture")
        workspace._created = workspace._running = True
        stopping, release = asyncio.Event(), asyncio.Event()
        event = asyncio.Event()
        event.set()

        async def docker(*args: str, **_kwargs: object) -> tuple[CommandResult, bytes]:
            """Delay Docker stop and expose final state only once it finishes."""
            if args[0] == "stop":
                stopping.set()
                await release.wait()
                return CommandResult(0, "", ""), b""
            if args[2] == "{{.State.Running}}":
                return CommandResult(0, "false", ""), b""
            self.assertTrue(release.is_set())
            return CommandResult(0, '{"Running":false,"OOMKilled":false}', ""), b""

        workspace._docker = AsyncMock(side_effect=docker)
        result = CommandResult(137, "actual output", "")
        with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                return_value=(result, b"", False, False))):
            stop = asyncio.create_task(workspace.stop())
            await stopping.wait()
            execution = asyncio.create_task(workspace.exec(["codex"], controller_stop=event))
            await asyncio.sleep(0)
            self.assertFalse(execution.done())
            release.set()
            await stop
            self.assertEqual(await execution, result)

    async def test_controller_signal_without_actual_stop_is_rejected(self) -> None:
        """The controller flag alone cannot authorize accepting an externally killed command."""
        workspace = DockerWorkspace("fixture")
        workspace._running = True
        workspace.stop = AsyncMock()
        event = asyncio.Event()
        event.set()
        with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                return_value=(CommandResult(137, "", ""), b"", False, False))):
            with self.assertRaisesRegex(RuntimeError, "without a workspace stop"):
                await workspace.exec(["codex"], controller_stop=event)
    async def test_failed_stop_overrides_candidate_budget_classification(self) -> None:
        """A failed freeze is infrastructure failure even after candidate timeout."""
        workspace = DockerWorkspace("fixture")
        workspace._running = True
        workspace.stop = AsyncMock(side_effect=RuntimeError("daemon unavailable"))
        with patch("rl.agentic.envs.docker_workspace._command", new=AsyncMock(
                return_value=(CommandResult(-1, "", ""), b"", True, False))):
            with self.assertLogs("rl.agentic.envs.docker_workspace", level="ERROR"):
                with self.assertRaisesRegex(RuntimeError, "could not stop"):
                    await workspace.exec(["python", "solution.py"])

    async def test_start_cleanup_preserves_original_error(self) -> None:
        """A failed cleanup is logged without replacing the creation failure."""
        workspace = DockerWorkspace("fixture")
        workspace._docker = AsyncMock(side_effect=[
            (CommandResult(0, "sha256:" + "a" * 64, ""), b""), RuntimeError("create failed"),
        ])
        workspace.close = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        with self.assertLogs("rl.agentic.envs.docker_workspace", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "create failed"):
                await workspace.start()
        workspace.close.assert_awaited_once()

    def test_reject_shared_host_network(self) -> None:
        """A configurable network must not grant host or peer namespace access."""
        for network in ("host", "container:foreign"):
            with self.assertRaises(ValueError):
                DockerWorkspace("fixture", network=network)
