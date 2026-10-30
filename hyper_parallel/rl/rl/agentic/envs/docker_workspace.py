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
"""Bounded Docker workspaces and stopped-container artifact collection."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import fcntl
import io
from itertools import chain
import json
import logging
import math
from pathlib import Path, PurePosixPath
import stat
import tarfile
from typing import AsyncIterator, Mapping, Optional, Sequence
from uuid import uuid4


_LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def acquire_workspace_slot(directory: Path, capacity: int = 1, timeout: float = 600) -> AsyncIterator[None]:
    """Hold one process-shared node lease through candidate execution and grading."""
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("Workspace capacity must be a positive integer")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Workspace slot timeout must be positive and finite")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    deadline = asyncio.get_running_loop().time() + timeout
    held = None
    try:
        while held is None:
            for index in range(capacity):
                stream = (directory / f"workspace-{index}.lock").open("a+b")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                except BaseException:
                    stream.close()
                    raise
                else:
                    held = stream
                    break
            if held is None:
                if asyncio.get_running_loop().time() >= deadline:
                    raise RuntimeError("Timed out waiting for a repository workspace slot")
                await asyncio.sleep(0.05)
        yield
    finally:
        if held is not None:
            fcntl.flock(held.fileno(), fcntl.LOCK_UN)
            held.close()


@dataclass(frozen=True)
class CommandResult:
    """Decoded command output; nonzero exits are candidate results."""

    returncode: int
    stdout: str
    stderr: str


class InvalidSubmissionError(ValueError):
    """Candidate archive violates the submission path or resource contract."""


class WorkspaceTimeoutError(TimeoutError):
    """A candidate command exceeded its deadline; workspace stop was requested."""

    def __init__(self, result: CommandResult) -> None:
        """Retain partial command output for budget diagnostics."""
        super().__init__("Docker workspace command timed out")
        self.result = result


class WorkspaceOutputLimitError(RuntimeError):
    """A command exceeded its combined raw output budget."""

    def __init__(self, result: CommandResult) -> None:
        """Retain the bounded output prefix for budget diagnostics."""
        super().__init__("Docker workspace command exceeded its output budget")
        self.result = result


def _snapshot_files(data: bytes, max_bytes: int, max_files: int) -> dict[str, bytes]:
    """Validate every tar member without extracting candidate paths."""
    try:
        return _read_tar(data, max_bytes, max_files)
    except (tarfile.TarError, ValueError, EOFError) as error:
        raise InvalidSubmissionError(f"Invalid workspace tar archive: {error}") from error


def _archive_member_name(member: tarfile.TarInfo, count: int, max_files: int, seen: set[str]) -> str:
    """Reject unsafe member names, duplicate paths, and nonregular archive types."""
    path = PurePosixPath(member.name)
    if (count > max_files or not member.name or path.is_absolute() or ".." in path.parts
            or "\x00" in member.name or "\\" in member.name):
        raise ValueError("Unsafe or excessive workspace archive members")
    name = str(path)
    if name in seen or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
        raise ValueError("Duplicate, linked, or special workspace archive member")
    seen.add(name)
    if name == "." and not member.isdir():
        raise ValueError("Invalid workspace root member")
    return name


def _read_tar(data: bytes, max_bytes: int, max_files: int) -> dict[str, bytes]:
    """Collect only plain regular files after checking their member paths."""
    files = {}
    seen = set()
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for count, member in enumerate(archive, 1):
            name = _archive_member_name(member, count, max_files, seen)
            if member.isdir():
                continue
            total += member.size
            if member.size < 0 or total > max_bytes:
                raise ValueError("Workspace artifact exceeds its byte budget")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Workspace archive member has no contents")
            with stream:
                content = stream.read(member.size + 1)
            if len(content) != member.size:
                raise ValueError("Incomplete workspace archive member")
            files[name] = content
        tail = data[archive.offset:]
        if len(tail) < 1024 or any(tail):
            raise ValueError("Workspace archive has missing end markers or trailing contents")
    for name in seen:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise ValueError("Workspace file is also used as a directory")
    return files


def read_snapshot(archive: Path, *, max_bytes: int = 16 * 1024 * 1024,
                  max_files: int = 4096) -> dict[str, bytes]:
    """Read validated regular files from an uncompressed stopped-workspace tar."""
    if max_bytes <= 0 or max_files <= 0:
        raise ValueError("Artifact budgets must be positive")
    with Path(archive).open("rb") as stream:
        data = stream.read(max_bytes + max_files * 2048 + 10241)
    if len(data) > max_bytes + max_files * 2048 + 10240:
        raise InvalidSubmissionError("Workspace tar exceeds its transport budget")
    return _snapshot_files(data, max_bytes, max_files)


def _fixture_archive(source: Path) -> bytes:
    """Copy trusted fixture bytes with container-local ownership, never host ownership."""
    buffer = io.BytesIO()
    total = 0
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for count, path in enumerate(chain([source], source.rglob("*")), 1):
            if count > 4096:
                raise ValueError("Workspace fixture exceeds its file budget")
            metadata = path.lstat()
            if (not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
                    or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)):
                raise ValueError("Workspace fixture contains a link or special file")
            total += metadata.st_size if path.is_file() else 0
            if total > 16 * 1024 * 1024:
                raise ValueError("Workspace fixture exceeds its byte budget")
            member = archive.gettarinfo(str(path), arcname=str(path.relative_to(source)))
            # Input belongs to the controller. Its numeric UID must not leak into
            # the candidate, whose root intentionally has no DAC_OVERRIDE/CHOWN.
            member.uid = member.gid = 0
            member.uname = member.gname = "root"
            member.mode &= 0o777
            if member.isfile():
                with path.open("rb") as stream:
                    archive.addfile(member, stream)
            else:
                archive.addfile(member)
    return buffer.getvalue()


async def _command(argv: Sequence[str], *, stdin: Optional[bytes] = None,
                   timeout: float = 30, limit: int = 1024 * 1024) -> tuple[CommandResult, bytes, bool, bool]:
    """Run a bounded CLI, concurrently draining both pipes and feeding input."""
    creation = asyncio.create_task(asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    ))
    try:
        process = await asyncio.shield(creation)
    except asyncio.CancelledError:
        process = await creation
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.communicate(), timeout=2)
        except asyncio.TimeoutError:
            _LOGGER.warning("Command pipes remained open after cancellation")
        raise
    buffers = [bytearray(), bytearray()]
    retained = 0
    overflow = asyncio.Event()

    async def drain(pipe: asyncio.StreamReader, buffer: bytearray) -> None:
        """Drain one stream while sharing the retained-byte budget."""
        nonlocal retained
        while True:
            chunk = await pipe.read(65536)
            if not chunk:
                return
            keep = min(len(chunk), limit - retained)
            buffer.extend(chunk[:keep])
            retained += keep
            if keep < len(chunk):
                overflow.set()

    async def feed() -> None:
        """Feed bounded chunks so unread stdin cannot bypass the timeout."""
        try:
            if stdin is not None:
                for offset in range(0, len(stdin), 65536):
                    process.stdin.write(stdin[offset:offset + 65536])
                    await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    tasks = [asyncio.create_task(drain(process.stdout, buffers[0])),
             asyncio.create_task(drain(process.stderr, buffers[1])),
             asyncio.create_task(feed()), asyncio.create_task(process.wait())]
    complete = asyncio.gather(*tasks)
    exceeded = asyncio.create_task(overflow.wait())
    timed_out = False
    try:
        done, _ = await asyncio.wait([complete, exceeded], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        timed_out = not done
        if complete in done:
            complete.result()
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        exceeded.cancel()
        _, pending = await asyncio.wait(tasks, timeout=2)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, complete, exceeded, return_exceptions=True)
    result = CommandResult(process.returncode if process.returncode is not None else -1,
                           buffers[0].decode("utf-8", errors="replace"),
                           buffers[1].decode("utf-8", errors="replace"))
    return result, bytes(buffers[0]), timed_out, overflow.is_set()


class DockerWorkspace:
    """Own one container with bounded input mounts, no devices and no privileged execution."""

    def __init__(self, image: str, *, network: str = "none", memory: str = "1g", cpus: float = 1.0,
                 pids_limit: int = 64, name_prefix: str = "hyper-workspace", run_id: Optional[str] = None,
                 docker_host: Optional[str] = None, user: str = "0:0", read_only: bool = False,
                 mounts: Sequence[tuple[Path, str, bool]] = (), runtime: Optional[str] = None) -> None:
        """Record trusted settings; resolve the image on the selected daemon at start."""
        if not image or not network or not memory or not math.isfinite(cpus) or cpus <= 0 or pids_limit <= 0:
            raise ValueError("Workspace image, network and positive resource limits are required")
        if not name_prefix or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in name_prefix):
            raise ValueError("Invalid workspace name prefix")
        if network == "host" or network.startswith("container:"):
            raise ValueError("Workspace requires an isolated Docker network")
        self._validate_execution_options(docker_host, runtime, user, read_only)
        self.docker_host = docker_host
        self.runtime = runtime
        self.run_id = run_id or uuid4().hex
        self.image = image
        self.name = f"{name_prefix}-{uuid4().hex}"
        self.network, self.memory, self.cpus, self.pids_limit = network, memory, cpus, pids_limit
        self.user, self.read_only = user, read_only
        self.mounts = self._validate_mounts(mounts)
        self._created = False
        self._running = False
        self._lock = asyncio.Lock()
        self._stop_task: asyncio.Task | None = None

    @staticmethod
    def _validate_execution_options(docker_host: Optional[str], runtime: Optional[str],
                                    user: str, read_only: bool) -> None:
        """Validate trusted daemon/runtime options and the container user boundary."""
        if docker_host is not None and (not isinstance(docker_host, str) or not docker_host
                                        or any(char.isspace() for char in docker_host)):
            raise ValueError("docker_host must be a nonempty Docker endpoint without whitespace")
        if runtime is not None and (not isinstance(runtime, str) or not runtime
                                    or any(not (char.isascii() and (char.isalnum() or char in "_.-"))
                                           for char in runtime)):
            raise ValueError("Workspace runtime must be a registered Docker runtime name")
        if not isinstance(read_only, bool):
            raise ValueError("Workspace read_only must be a boolean")
        if not isinstance(user, str) or len(user.split(":")) != 2 or not all(
                part.isdigit() for part in user.split(":")):
            raise ValueError("Workspace user must be a numeric uid:gid")

    @staticmethod
    def _validate_mounts(mounts: Sequence[tuple[Path, str, bool]]) -> list[tuple[Path, str, bool]]:
        """Resolve mount sources and restrict destinations to candidate directories."""
        validated = []
        for source, target, readonly in mounts:
            source = Path(source).resolve(strict=True)
            destination = PurePosixPath(target)
            in_workspace = any(target == root or target.startswith(root + "/")
                               for root in ("/work", "/workspace", "/codex-home"))
            if (":" in str(source) or not isinstance(readonly, bool) or not in_workspace
                    or ".." in destination.parts or str(destination) != target):
                raise ValueError("Input mounts must stay within workspace or candidate-home directories")
            validated.append((source, target, readonly))
        return validated

    async def _cleanup_after_error(self, *, remove: bool = False) -> bool:
        """Complete bounded cleanup without replacing the original failure."""
        cleanup = asyncio.create_task(self.close() if remove else self.stop())
        try:
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
        except Exception:  # pylint: disable=broad-exception-caught
            _LOGGER.exception("Workspace cleanup failed for owned container %s", self.name)
            return False
        return True

    def docker_command(self, *args: str) -> list[str]:
        """Build argv for the same daemon used by workspace management and relay."""
        return ["docker", *(["--host", self.docker_host] if self.docker_host is not None else []), *args]

    async def _docker(self, *args: str, stdin: Optional[bytes] = None,
                      timeout: float = 30, limit: int = 1024 * 1024) -> tuple[CommandResult, bytes]:
        """Run a bounded management command and reject transport or Docker failures."""
        result, raw, timed_out, overflow = await _command(
            self.docker_command(*args), stdin=stdin, timeout=timeout, limit=limit,
        )
        if timed_out:
            raise RuntimeError(f"Docker {args[0]} management command timed out")
        if overflow:
            raise RuntimeError(f"Docker {args[0]} management output exceeded its transport budget")
        if result.returncode:
            raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[:1000]}")
        return result, raw

    async def start(self) -> None:
        """Create the fixed-image workspace; never implicitly pull an image."""
        if self._created:
            raise RuntimeError("Workspace has already been started")
        if not (self.image.startswith("sha256:") and len(self.image) == 71
                and all(value in "0123456789abcdef" for value in self.image[7:])):
            result, _ = await self._docker("image", "inspect", "--format", "{{.Id}}", self.image)
            self.image = result.stdout.strip()
            if not self.image.startswith("sha256:"):
                raise RuntimeError("Docker did not resolve a local image ID")
        self._created = True
        try:
            labels = ["--label", "hyper-rl.owner=code-agent", "--label", f"hyper-rl.run-id={self.run_id}"]
            options = []
            if self.runtime is not None:
                options.extend(("--runtime", self.runtime))
            if self.read_only:
                options.extend(("--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777"))
            for source, target, readonly in self.mounts:
                options.extend(("-v", f"{source}:{target}:{'ro' if readonly else 'rw'}"))
            await self._docker("create", "--name", self.name, *labels, *options, "--network", self.network,
                               "--memory", self.memory, "--memory-swap", self.memory,
                               "--cpus", str(self.cpus), "--pids-limit", str(self.pids_limit),
                               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                               "--user", self.user, "--entrypoint", "/bin/sleep", self.image, "infinity")
            await self._docker("start", self.name)
            self._running = True
            if self.read_only:
                return
            result = await self.exec(["rm", "-rf", "--", "/workspace"], cwd="/")
            if result.returncode:
                raise RuntimeError("Cannot clear image workspace contents")
            result = await self.exec(["mkdir", "-p", "/workspace"], cwd="/")
            if result.returncode:
                raise RuntimeError("Cannot create workspace directory")
        except BaseException:
            await self._cleanup_after_error(remove=True)
            raise

    async def inspect(self) -> dict:
        """Read the owned container's state and boundary before removal."""
        result, _ = await self._docker("inspect", self.name)
        return json.loads(result.stdout)[0]

    async def copy_in(self, source: Path, destination: str = "/workspace") -> None:
        """Copy trusted repository or session configuration into fixed directories."""
        if (not self._running or destination not in ("/workspace", "/opt/hyper-codex-home")
                or not Path(source).is_dir()):
            raise ValueError("copy_in requires a running workspace and a trusted source directory")
        try:
            result = await self.exec(["mkdir", "-p", destination], cwd="/")
            if result.returncode:
                raise RuntimeError("Cannot create workspace copy destination")
            await self._docker("cp", "-", f"{self.name}:{destination}", stdin=_fixture_archive(Path(source)))
        except BaseException:
            await self._cleanup_after_error()
            raise

    def _exec_command(self, argv: Sequence[str], cwd: str, env: Optional[Mapping[str, str]]) -> list[str]:
        """Build the Docker command while checking each explicit environment value."""
        command = self.docker_command("exec", "-i", "-w", cwd)
        for key, value in (env or {}).items():
            if not key or "=" in key or not isinstance(value, str):
                raise ValueError("Invalid command environment")
            command.extend(["-e", f"{key}={value}"])
        command.extend([self.name, *argv])
        return command

    async def exec(self, argv: Sequence[str], *, cwd: str = "/workspace",
                   env: Optional[Mapping[str, str]] = None, stdin: Optional[str] = None,
                   timeout: float = 30, output_limit_bytes: int = 1024 * 1024,
                   controller_stop: asyncio.Event | None = None) -> CommandResult:
        """Execute argv; budget failure or cancellation stops the entire container."""
        if (isinstance(argv, str) or not argv or any(not isinstance(arg, str) for arg in argv)
                or not math.isfinite(timeout) or timeout <= 0 or output_limit_bytes <= 0):
            raise ValueError("Command argv and positive finite budgets are required")
        async with self._lock:
            if not self._running:
                raise RuntimeError("Workspace is not running")
            command = self._exec_command(argv, cwd, env)
            try:
                result, _, timed_out, overflow = await _command(
                    command, stdin=None if stdin is None else stdin.encode(), timeout=timeout,
                    limit=output_limit_bytes,
                )
                if overflow:
                    raise WorkspaceOutputLimitError(result)
                if timed_out:
                    raise WorkspaceTimeoutError(result)
                return await self._check_exec_state(result, controller_stop)
            except BaseException as error:
                cleaned = await self._cleanup_after_error()
                if not cleaned and isinstance(error, (WorkspaceTimeoutError, WorkspaceOutputLimitError)):
                    raise RuntimeError("Workspace could not stop after candidate budget failure") from error
                raise

    async def _check_exec_state(self, result: CommandResult,
                                controller_stop: asyncio.Event | None) -> CommandResult:
        """Distinguish controller cancellation from process or container failure."""
        if controller_stop is not None and controller_stop.is_set() and result.returncode in (137, 143):
            if self._stop_task is None:
                raise RuntimeError("Controller stop was signaled without a workspace stop operation")
            await asyncio.wait_for(asyncio.shield(self._stop_task), timeout=45)
        state, _ = await self._docker("inspect", "--format", "{{json .State}}", self.name)
        state = json.loads(state.stdout)
        if (controller_stop is not None and controller_stop.is_set() and not state["Running"]
                and not state["OOMKilled"] and result.returncode in (137, 143)):
            return result
        if not state["Running"] or state["OOMKilled"] or result.returncode in (125, 126, 127, 137):
            raise RuntimeError(f"Docker exec or workspace failed: exit={result.returncode}, "
                               f"running={state['Running']}, oom_killed={state['OOMKilled']}; "
                               f"{result.stderr[:1000]}")
        return result

    async def stop(self) -> None:
        """Stop and confirm the container is no longer running, preserving files."""
        if not self._created:
            return
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(self._stop_container())
        await asyncio.shield(self._stop_task)

    async def _stop_container(self) -> None:
        """Let a concurrent exec await the same completed Docker stop and state check."""
        await self._docker("stop", "--time", "1", self.name, timeout=15)
        result, _ = await self._docker("inspect", "--format", "{{.State.Running}}", self.name)
        if json.loads(result.stdout) is not False:
            raise RuntimeError("Workspace did not stop")
        self._running = False

    async def export(self, destination: Path, *, max_bytes: int = 16 * 1024 * 1024,
                     max_files: int = 4096) -> Path:
        """Export the stopped /workspace as a validated tar, never extracting paths."""
        if not self._created or self._running or max_bytes <= 0 or max_files <= 0:
            raise RuntimeError("Export requires a stopped workspace and positive budgets")
        result, _ = await self._docker("inspect", "--format", "{{.State.Running}}", self.name)
        if json.loads(result.stdout) is not False:
            raise RuntimeError("Refusing to export a running workspace")
        return await self.snapshot(destination, max_bytes=max_bytes, max_files=max_files)

    async def snapshot(self, destination: Path, *, max_bytes: int = 16 * 1024 * 1024,
                       max_files: int = 4096) -> Path:
        """Copy trusted setup state before tools start; use export for candidate submissions."""
        if not self._created or max_bytes <= 0 or max_files <= 0:
            raise RuntimeError("Snapshot requires a created workspace and positive budgets")
        _, raw = await self._docker("cp", f"{self.name}:/workspace/.", "-",
                                    limit=max_bytes + max_files * 2048 + 10240)
        _snapshot_files(raw, max_bytes, max_files)
        destination = Path(destination)
        with destination.open("xb") as stream:
            stream.write(raw)
        return destination

    async def close(self) -> None:
        """Remove the owned container and every remaining process."""
        if self._created:
            result, _ = await self._docker(
                "ps", "--all", "--quiet", "--filter", f"name=^/{self.name}$",
                "--filter", "label=hyper-rl.owner=code-agent", "--filter", f"label=hyper-rl.run-id={self.run_id}",
            )
            if result.stdout.strip():
                await self._docker("rm", "--force", self.name)
            self._created = False
            self._running = False
