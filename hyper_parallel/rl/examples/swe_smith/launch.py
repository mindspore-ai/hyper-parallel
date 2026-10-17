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
"""Launch SWE-Smith training or inference with local or SSH Docker sandboxes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import grp
import json
import logging
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
from typing import Any, Sequence
import uuid

import yaml

from examples.swe_smith.run_config import LOCAL_DOCKER_HOST, build_config


_LOG = logging.getLogger(__name__)
_LOCAL_DOCKER_SOCKET = Path(LOCAL_DOCKER_HOST.removeprefix("unix://"))
_CONTROLLER_DOCKER = ["/usr/bin/docker", "--host", LOCAL_DOCKER_HOST]
_DEFAULT_IMAGE = "swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/hyper-rl:v0.23.0.post1-code-agent-arm64"
# Importing rl.config would load Torch in the host launcher.
_HCCL_MIN_PORT = 1024
_HCCL_MAX_PORT = 65520


def _trainer_hccl_environment() -> dict[str, str]:
    """Resolve a complete, valid HCCL port override for the training processes."""
    base = os.environ.get("HCCL_IF_BASE_PORT")
    interval = os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE")
    if base is None and interval is None:
        base, interval = "63200", "63200-63299"
    if base is None or interval is None:
        raise ValueError("Override HCCL_IF_BASE_PORT and HCCL_NPU_SOCKET_PORT_RANGE together")
    bounds = re.fullmatch(r"([0-9]{1,5})-([0-9]{1,5})", interval)
    if re.fullmatch(r"[0-9]{1,5}", base) is None or bounds is None:
        raise ValueError("Trainer HCCL ports require a numeric base port and a low-high port range")
    port = int(base)
    lower, upper = map(int, bounds.groups())
    if not _HCCL_MIN_PORT <= lower <= port <= upper <= _HCCL_MAX_PORT:
        raise ValueError(
            "Trainer HCCL ports must be within 1024..65520 and the socket range must contain the base port")
    return {"HCCL_IF_BASE_PORT": str(port), "HCCL_NPU_SOCKET_PORT_RANGE": f"{lower}-{upper}"}


def _devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--devices requires comma-separated nonnegative device IDs") from exc
    if not devices or min(devices) < 0 or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("--devices must contain distinct nonnegative device IDs")
    return devices


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse host launch options and reject incompatible execution modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qwen3_5_9b.yaml")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--images-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--devices", type=_devices, required=True)
    parser.add_argument("--docker-host", default=None)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mode", choices=("train", "infer"), default="train")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    parser.add_argument("--privileged-controller", action="store_true",
                        help="Explicitly run the trusted controller as root with privileged device access")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0
            or (args.steps is not None and args.steps <= 0)):
        parser.error("--timeout-seconds and --steps must be positive")
    if args.mode == "train" and args.instance_id:
        parser.error("--instance-id is only supported in inference mode")
    if args.mode == "infer" and args.resume is not None:
        parser.error("--resume is only supported in training mode")
    return args


def _bind(command: list[str], source: Path, *, target: Path | None = None, readonly: bool = False) -> None:
    destination = target or source
    for path in (source, destination):
        if "," in str(path) or "\n" in str(path):
            raise ValueError(f"Docker bind paths cannot contain commas or newlines: {path}")
    command.extend(["--mount", f"type=bind,src={source},dst={destination}" + (",readonly" if readonly else "")])


def _ssh_mounts(command: list[str], ssh_directory: Path) -> set[int]:
    """Mount the verified SSH identity files and agent socket for remote sandboxes."""
    agent = os.environ.get("SSH_AUTH_SOCK")
    socket = Path(agent).resolve() if agent else None
    if socket is None or not socket.exists() or not stat.S_ISSOCK(socket.stat().st_mode):
        raise ValueError("SSH sandboxes require an unlocked SSH agent exported through SSH_AUTH_SOCK")
    _bind(command, socket, readonly=True)
    command.extend(["--env", f"SSH_AUTH_SOCK={socket}"])
    known_hosts = ssh_directory / "known_hosts"
    if not known_hosts.is_file():
        raise ValueError("SSH sandboxes require ~/.ssh/known_hosts; connect to and verify the sandbox host first")
    for filename in ("config", "known_hosts", "known_hosts2"):
        path = ssh_directory / filename
        if path.is_file():
            _bind(command, path, readonly=True)
    return {socket.stat().st_gid}


def _controller_command(args: argparse.Namespace, repository: Path, temporary: Path, name: str,
                        *, ssh_directory: Path | None) -> list[str]:
    """Build the trusted controller command with explicit device and bind ownership."""
    user_id = 0 if args.privileged_controller else os.getuid()
    command = [*_CONTROLLER_DOCKER, "run", "--detach", "--name", name, "--network", "host", "--shm-size", "32g",
               "--user", f"{user_id}:{os.getgid()}", "--workdir", str(repository / "hyper_parallel/rl")]
    if args.privileged_controller:
        command.append("--privileged")
    groups = set(os.getgroups())
    for physical in args.devices:
        source = Path(f"/dev/davinci{physical}")
        if not source.exists():
            raise ValueError(f"NPU device does not exist: {source}")
        command.extend(["--device", f"{source}:{source}"])
        groups.add(source.stat().st_gid)
    for source in (Path("/dev/davinci_manager"), Path("/dev/devmm_svm"), Path("/dev/hisi_hdc")):
        if not source.exists():
            raise ValueError(f"Required Ascend device does not exist: {source}")
        command.extend(["--device", str(source)])
        groups.add(source.stat().st_gid)
    for source in (Path("/usr/local/Ascend/driver/lib64"), Path("/usr/local/Ascend/driver/version.info"),
                   Path("/usr/local/dcmi"), Path("/etc/ascend_install.info"), Path("/etc/passwd"), Path("/etc/group")):
        if not source.exists():
            raise ValueError(f"Required Ascend/controller host path does not exist: {source}")
        _bind(command, source, readonly=True)
    for source, readonly in ((repository, False), (args.data_dir, True), (args.images_path, True),
                             (args.model_dir, True), (args.output_dir, False), (args.checkpoint_dir, False),
                             (temporary, False)):
        _bind(command, source, readonly=readonly)
    if args.resume is not None and not args.resume.is_relative_to(args.checkpoint_dir):
        _bind(command, args.resume, readonly=True)
    _bind(command, Path(_CONTROLLER_DOCKER[0]), target=Path("/opt/hyper-tools/docker"), readonly=True)
    for source in (Path("/usr/lib64/libltdl.so.7"), Path("/usr/lib/aarch64-linux-gnu/libltdl.so.7")):
        if source.exists():
            _bind(command, source.resolve(), target=Path("/opt/hyper-tools/lib/libltdl.so.7"), readonly=True)
            break
    if ssh_directory is not None:
        groups.update(_ssh_mounts(command, ssh_directory))
    else:
        socket = _LOCAL_DOCKER_SOCKET
        if not socket.exists() or not stat.S_ISSOCK(socket.stat().st_mode):
            raise ValueError(f"Local Docker socket does not exist: {socket}")
        _bind(command, socket)
        groups.add(socket.stat().st_gid)
    try:
        groups.add(grp.getgrnam("nobody").gr_gid)
    except KeyError:
        groups.add(pwd.getpwnam("nobody").pw_gid)
    for group in sorted(groups - {os.getgid()}):
        command.extend(["--group-add", str(group)])
    command.extend([args.image, "bash", str(args.output_dir / "launch.sh")])
    return command


def _script(args: argparse.Namespace, repository: Path, temporary: Path, *,
            trainer_hccl: dict[str, str], ssh_directory: Path | None) -> str:
    """Render container setup and the selected training or inference command."""
    source_root = repository / "hyper_parallel/rl"
    resolved_config = args.output_dir / "resolved_config.yaml"
    command = ["torchrun", "--standalone", f"--nproc_per_node={len(args.devices)}"]
    if args.mode == "train":
        command.extend([str(source_root / "train_rl.py"), str(resolved_config)])
    else:
        command.extend(["-m", "examples.swe_smith.infer", str(resolved_config),
                        "--dataset", str(args.data_dir / "train.parquet"),
                        "--dataset", str(args.data_dir / "validation.parquet"),
                        "--output", str(args.output_dir / "predictions.jsonl")])
        for identity in args.instance_id:
            command.extend(["--instance-id", identity])
    indices = args.devices if args.privileged_controller else range(len(args.devices))
    visible = ",".join(str(index) for index in indices)
    verification = ("import pathlib,hyper_parallel,rl; "
                    f"root=pathlib.Path({str(repository)!r}).resolve(); "
                    "modules=(hyper_parallel,rl); "
                    "paths=[pathlib.Path(module.__file__).resolve() for module in modules]; "
                    "print('Editable import paths:',paths); "
                    "assert all(path.is_relative_to(root) for path in paths), paths")
    ownership_cleanup = []
    if args.privileged_controller:
        ownership_cleanup = [
            "restore_output_ownership() {", "    local result=$?", "    trap - EXIT",
            "    if ! " + shlex.join(["chown", "--recursive", "--no-dereference", "--",
                                      f"{os.getuid()}:{os.getgid()}", str(args.output_dir),
                                      str(args.checkpoint_dir), str(temporary)]) + "; then",
            "        echo 'Could not restore ownership of this run output/checkpoint/temp' >&2",
            "        if [[ $result -eq 0 ]]; then result=1; fi", "    fi", '    exit "$result"', "}",
            "trap restore_output_ownership EXIT",
        ]
    ssh_wrapper = []
    if ssh_directory is not None:
        ssh_config = ssh_directory / "config"
        ssh_command = ["/usr/bin/ssh", "-F", str(ssh_config) if ssh_config.is_file() else "/dev/null",
                       "-o", f"UserKnownHostsFile={ssh_directory / 'known_hosts'}", "-o", "StrictHostKeyChecking=yes",
                       "-o", "BatchMode=yes"]
        wrapper = temporary / "bin/ssh"
        ssh_wrapper = [shlex.join(["mkdir", "--parents", str(wrapper.parent)]),
                       "cat > " + shlex.quote(str(wrapper)) + " <<'HYPER_SWE_SSH'", "#!/usr/bin/env bash",
                       "exec " + shlex.join(ssh_command) + ' "$@"', "HYPER_SWE_SSH",
                       shlex.join(["chmod", "700", str(wrapper)])]
    return "\n".join([
        "#!/usr/bin/env bash", "set -eo pipefail", "umask 0002", *ownership_cleanup,
        f"export HOME={shlex.quote(str(temporary / 'home'))}",
        f"export TMPDIR={shlex.quote(str(temporary / 'tmp'))}",
        f"export PIP_CACHE_DIR={shlex.quote(str(temporary / 'pip-cache'))}",
        f"export HF_HOME={shlex.quote(str(temporary / 'huggingface'))}",
        "export PIP_DISABLE_PIP_VERSION_CHECK=1",
        'mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"',
        *ssh_wrapper,
        shlex.join(["git", "config", "--global", "--add", "safe.directory", str(repository)]),
        "if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then",
        "    source /usr/local/Ascend/ascend-toolkit/set_env.sh",
        "elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then",
        "    source /usr/local/Ascend/cann/set_env.sh", "else",
        "    echo 'CANN set_env.sh is missing from the controller image' >&2", "    exit 1", "fi",
        "set -u", f'export PATH={shlex.quote(str(temporary / "bin"))}:"/opt/hyper-tools:$PATH"',
        'export LD_LIBRARY_PATH="/opt/hyper-tools/lib:${LD_LIBRARY_PATH:-}"',
        f'export PYTHONPATH={shlex.quote(str(repository) + ":" + str(source_root))}:"${{PYTHONPATH:-}}"',
        f"export ASCEND_RT_VISIBLE_DEVICES={visible}", 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"',
        *(f"export {name}={shlex.quote(value)}" for name, value in trainer_hccl.items()),
        shlex.join(["python", "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--editable",
                    str(repository), "--editable", str(source_root)]),
        shlex.join(["python", "-c", verification]),
        "stop_training() {", "    trap - TERM INT",
        '    if [[ -n ${training_pid:-} ]]; then',
        '        kill -INT "$training_pid" 2>/dev/null || true',
        '        wait "$training_pid" || true', "    fi", "    exit 143", "}",
        "trap stop_training TERM INT", shlex.join(command) + " &", "training_pid=$!",
        'wait "$training_pid"', "",
    ])


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cleanup_temporary(temporary: Path, output_dir: Path) -> None:
    """Remove this run's temporary directory and record any cleanup failure."""
    try:
        shutil.rmtree(temporary)
    except OSError as exc:
        _LOG.error("Could not remove this run's temporary directory %s: %s", temporary, exc)
        manifest_path = output_dir / "run-manifest.json"
        try:
            if manifest_path.is_file():
                recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
                recorded["temporary_cleanup_error"] = str(exc)
                recorded["temporary_dir"] = str(temporary)
                _write_manifest(manifest_path, recorded)
        except (OSError, ValueError) as record_error:
            _LOG.error("Could not record temporary cleanup failure: %s", record_error)


def _interrupt_on_signal(signum: int, frame: Any) -> None:
    """Translate termination into the controller's normal interruption cleanup."""
    del frame
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt(f"Termination signal {signum} received")


def _cleanup_controller(name: str, manifest: dict[str, Any]) -> None:
    """Stop and remove only the owned controller, preserving failure evidence."""
    if manifest.get("status") in ("timed_out", "interrupted"):
        manifest["sandbox_cleanup_warning"] = (
            "Candidate/grader removal cannot be guaranteed after interruption. Inspect this run's "
            "session.registered events and exact hyper-rl.run-id labels on its sandbox Docker endpoint.")
        try:
            stopped = subprocess.run([*_CONTROLLER_DOCKER, "stop", "--time", "30", name],
                                     capture_output=True, text=True, check=False, timeout=45)
            manifest["controller_stop_returncode"] = stopped.returncode
            if stopped.returncode and "No such container" not in stopped.stderr:
                manifest["controller_stop_error"] = stopped.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            manifest["controller_stop_error"] = str(exc)
    try:
        removed = subprocess.run([*_CONTROLLER_DOCKER, "rm", "--force", name],
                                 capture_output=True, text=True, check=False, timeout=30)
        if removed.returncode and "No such container" not in removed.stderr:
            manifest["cleanup_error"] = removed.stderr.strip()
            _LOG.error("Could not remove controller %s: %s", name, removed.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        manifest["cleanup_error"] = str(exc)
        _LOG.error("Could not remove controller %s: %s", name, exc)


def _run_controller(command: list[str], name: str, output_dir: Path, timeout: float,
                    manifest: dict[str, Any]) -> None:
    """Run the controller with bounded waiting, log capture and final cleanup."""
    with ExitStack() as resources:
        previous_handler = signal.signal(signal.SIGTERM, _interrupt_on_signal)
        resources.callback(signal.signal, signal.SIGTERM, previous_handler)
        manifest_path = output_dir / "run-manifest.json"
        logger = None
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            manifest.update(controller_id=result.stdout.strip(), status="running")
            _write_manifest(manifest_path, manifest)
            with (output_dir / "runtime.log").open("w", encoding="utf-8") as log:
                logger = resources.enter_context(subprocess.Popen(
                    [*_CONTROLLER_DOCKER, "logs", "--follow", name], stdout=log, stderr=subprocess.STDOUT, text=True))
                result = subprocess.run([*_CONTROLLER_DOCKER, "wait", name], check=True, capture_output=True,
                                        text=True, timeout=timeout)
                code = int(result.stdout.strip())
                manifest.update(exit_code=code, status="completed" if code == 0 else "failed")
                if code:
                    raise RuntimeError(
                        f"SWE-Smith controller exited with status {code}; see {output_dir / 'runtime.log'}")
        except subprocess.TimeoutExpired as exc:
            manifest.update(status="timed_out", error=f"Controller exceeded {timeout} seconds")
            raise TimeoutError(manifest["error"]) from exc
        except KeyboardInterrupt as exc:
            manifest.update(status="interrupted", error=str(exc) or "Launcher interrupted")
            raise
        except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as exc:
            if manifest.get("status") != "timed_out":
                detail = str(exc)
                if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                    detail += ": " + exc.stderr.strip()
                manifest.update(status="failed", error=detail)
            raise
        finally:
            _cleanup_controller(name, manifest)
            if logger is not None:
                try:
                    logger.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logger.terminate()
                    try:
                        logger.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        logger.kill()
                        logger.wait()
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_manifest(manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """Resolve a recipe, record its launch inputs and run one owned controller."""
    logging.basicConfig(level=logging.INFO)
    args = _arguments(argv)
    trainer_hccl = _trainer_hccl_environment()
    for name in ("config", "data_dir", "images_path", "model_dir", "output_dir", "checkpoint_dir", "resume"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.output_dir == args.checkpoint_dir:
        raise ValueError("--output-dir and --checkpoint-dir must be separate directories")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {args.output_dir}")
    if (args.privileged_controller and args.checkpoint_dir.exists()
            and any(args.checkpoint_dir.iterdir())):
        raise ValueError("Root privileged controllers require a new or empty dedicated checkpoint directory")
    template = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = build_config(template, data_dir=args.data_dir, images_path=args.images_path, model_dir=args.model_dir,
                          output_dir=args.output_dir, checkpoint_dir=args.checkpoint_dir,
                          docker_host=args.docker_host, steps=args.steps, resume=args.resume)
    args.docker_host = config["agentic"]["codex"]["workspace"]["docker_host"]
    ssh_directory = (Path(pwd.getpwuid(os.getuid()).pw_dir) / ".ssh"
                     if args.docker_host.startswith("ssh://") else None)
    if args.mode == "train":
        accelerator = config["train"]["accelerator"]
        expected = 1
        for dimension in ("dp_replicate", "dp_shard", "tp", "cp", "pp"):
            expected *= int(accelerator.get(dimension, 1))
    else:
        engine = config["rollout"]["vllm"]
        expected = int(engine["tensor_parallel_size"]) * int(engine.get("data_parallel_size", 1))
    if expected != len(args.devices):
        raise ValueError(f"Recipe requires {expected} devices; --devices selected {len(args.devices)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(args.checkpoint_dir, os.W_OK):
        raise ValueError(f"Checkpoint directory is not writable: {args.checkpoint_dir}")
    name = "swe-smith-" + uuid.uuid4().hex
    temporary = Path(tempfile.mkdtemp(prefix=name + "-"))
    repository = Path(__file__).resolve().parents[4]
    try:
        command = _controller_command(args, repository, temporary, name, ssh_directory=ssh_directory)
        args.output_dir.joinpath("resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                                                   encoding="utf-8")
        script = args.output_dir / "launch.sh"
        script.write_text(_script(args, repository, temporary, trainer_hccl=trainer_hccl,
                                  ssh_directory=ssh_directory), encoding="utf-8")
        script.chmod(0o700)
        image = subprocess.run([*_CONTROLLER_DOCKER, "image", "inspect", args.image, "--format", "{{.Id}}"],
                               check=True, capture_output=True, text=True).stdout.strip()
        manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared",
                    "controller_name": name, "controller_image": args.image, "controller_image_id": image,
                    "privileged_controller": args.privileged_controller,
                    "mode": args.mode, "physical_devices": list(args.devices), "docker_argv": command,
                    "sandbox_docker_host": args.docker_host,
                    "config": str(args.output_dir / "resolved_config.yaml"), "temporary_dir": str(temporary)}
        _write_manifest(args.output_dir / "run-manifest.json", manifest)
        _LOG.info("Launching %s; log: %s", name, args.output_dir / "runtime.log")
        _run_controller(command, name, args.output_dir, args.timeout_seconds, manifest)
    finally:
        _cleanup_temporary(temporary, args.output_dir)


if __name__ == "__main__":
    main()
