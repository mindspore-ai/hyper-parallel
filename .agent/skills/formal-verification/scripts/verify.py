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
"""Adapt FM-Agent's public CLI to configurable distributed training frameworks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_ROOT.parents[2]
LOGGER = logging.getLogger(__name__)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], check=True,
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    return result.stdout.strip()


def _markdown(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Knowledge and validator paths must be non-empty strings")
    path = (base / value).resolve()
    if not path.is_file() or path.suffix.lower() != ".md":
        raise ValueError(f"Expected an existing Markdown file: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    """Load a framework profile and resolve its Markdown inputs relative to the config.

    Args:
        path: JSON configuration file.

    Returns:
        Validated configuration with absolute knowledge and validator paths.
    """
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("framework"), str):
        raise ValueError("Config must be an object with a framework string")
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Config must contain non-empty profiles")
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Profile {name} must be an object")
        scopes = profile.get("submodules")
        if not isinstance(scopes, list) or not scopes or any(not isinstance(item, str) for item in scopes):
            raise ValueError(f"Profile {name} must contain a non-empty list of submodule strings")
    knowledge = config.get("domain_knowledge")
    if not isinstance(knowledge, list) or not knowledge:
        raise ValueError("Config must contain a non-empty domain_knowledge list")
    config["domain_knowledge"] = [str(_markdown(path.parent, item)) for item in knowledge]
    config["bug_validator"] = str(_markdown(path.parent, config.get("bug_validator")))
    return config


def resolve_scope(repo: Path, scopes: list[str]) -> list[str]:
    """Validate and collapse overlapping source directories inside a repository.

    Args:
        repo: Resolved repository root.
        scopes: Relative or absolute source directories.

    Returns:
        Unique project-relative directories accepted by FM-Agent's --submodule.
    """
    directories = set()
    for value in scopes:
        if not value.strip():
            raise ValueError("Submodule paths cannot be empty")
        path = (repo / value).resolve()
        if not path.is_relative_to(repo) or path == repo or not path.is_dir():
            raise ValueError(f"Submodule must be an existing directory strictly inside {repo}: {value}")
        relative = path.relative_to(repo)
        if any(part.startswith(".") for part in relative.parts) or relative.parts[0].startswith("fm_agent"):
            raise ValueError(f"Select training source directories, not agent state: {value}")
        directories.add(relative.as_posix())
    collapsed: list[str] = []
    for directory in sorted(directories, key=lambda item: (item.count("/"), item)):
        if not any(directory.startswith(parent + "/") for parent in collapsed):
            collapsed.append(directory)
    if not collapsed:
        raise ValueError("Select at least one source directory")
    return collapsed


def _resolve_engine(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    engine = Path(raw_path).expanduser().resolve()
    if not (engine / "main.py").is_file():
        raise ValueError(f"FM-Agent checkout must contain main.py: {engine}")
    return engine


def _mode_flags(args: argparse.Namespace) -> list[str]:
    """Translate a validated execution mode into official engine CLI options."""
    if args.mode != "verify" and (args.resume or args.incremental):
        raise ValueError("--resume and --incremental require --mode verify")
    if args.mode == "estimate":
        return ["--estimate"]
    flags = ["--isolate", "--only-spec" if args.mode == "spec" else "--all-bugs"]
    if args.resume:
        flags.append("--resume")
    if args.incremental:
        intent = Path(args.incremental).expanduser().resolve()
        if not intent.is_file():
            raise ValueError(f"Incremental intent file does not exist: {intent}")
        flags.extend(["--incremental", str(intent)])
    return flags


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build an offline launch plan without importing or invoking a model backend.

    Args:
        args: Parsed CLI options.

    Returns:
        Repository, scope, input files and exact subprocess argument vector.
    """
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir() or Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise ValueError("--repo must name the root of a Git repository")
    config = load_config(Path(args.config))
    if args.profile not in config["profiles"]:
        raise ValueError(f"Unknown profile {args.profile!r}; choose from {', '.join(config['profiles'])}")
    profile = config["profiles"][args.profile]
    scope = resolve_scope(repo, args.submodule or profile["submodules"])
    engine = _resolve_engine(args.engine_root)
    flags = _mode_flags(args)
    knowledge = config["domain_knowledge"]
    if args.knowledge:
        knowledge = knowledge + [str(_markdown(Path.cwd(), item)) for item in args.knowledge]
    entry_point = str(engine / "main.py") if engine else "<FM-Agent>/main.py"
    command = [args.engine_python, entry_point, str(repo), *flags, "--submodule", *scope]
    command.extend(["--domain-knowledge", *knowledge, "--bug-validator", config["bug_validator"]])
    return {
        "framework": config["framework"], "profile": args.profile, "mode": args.mode,
        "repository": str(repo), "head": _git(repo, "rev-parse", "HEAD"),
        "has_local_changes": bool(_git(repo, "status", "--porcelain")),
        "submodules": scope, "focus": profile.get("focus", ""),
        "domain_knowledge": knowledge, "bug_validator": config["bug_validator"],
        "engine_root": str(engine) if engine else None, "command": command,
        "workspace": str(repo / ("fm_agent_estimate" if args.mode == "estimate" else "fm_agent")),
        "analysis_status": "not_run",
    }


def check_engine(plan: dict[str, Any]) -> None:
    """Check dependencies and CLI compatibility using the official --help entry point.

    Args:
        plan: Resolved launch plan with an installed FM-Agent checkout.
    """
    if not plan["engine_root"]:
        raise ValueError("Set FM_AGENT_HOME or --engine-root to an installed official FM-Agent checkout")
    help_result = subprocess.run(
        [plan["command"][0], str(Path(plan["engine_root"]) / "main.py"), "--help"],
        cwd=plan["engine_root"], capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_engine_environment(), check=False, timeout=60,
    )
    if help_result.returncode:
        raise ValueError(f"FM-Agent environment check failed; install its dependencies:\n{help_result.stderr[-3000:]}")
    missing = [item for item in plan["command"] if item.startswith("--") and item not in help_result.stdout]
    if missing:
        raise ValueError(f"FM-Agent CLI is incompatible; missing options: {', '.join(missing)}")


def _check_run_state(plan: dict[str, Any], args: argparse.Namespace) -> None:
    """Protect existing results and require a matching scope when reusing a run."""
    workspace = Path(plan["workspace"])
    if workspace.is_symlink():
        raise ValueError(f"FM-Agent workspace must not be a symlink: {workspace}")
    if args.mode == "estimate":
        return
    if args.resume or args.incremental:
        if not workspace.is_dir():
            raise ValueError("No previous fm_agent run to reuse; start with --mode verify")
        if args.incremental and not (workspace / "version.log").is_file():
            raise ValueError("Incremental mode requires fm_agent/version.log from a previous completed run")
        manifest = workspace / "training_adapter.json"
        if not manifest.is_file():
            raise ValueError(
                "Previous run has no adapter scope record; archive it or resume with the original FM-Agent CLI",
            )
        if json.loads(manifest.read_text(encoding="utf-8")) != _run_scope(plan):
            raise ValueError(
                "Previous run used a different framework, mode or source scope; archive it for a fresh run",
            )
    elif workspace.exists():
        raise ValueError("fm_agent already exists; archive it for a fresh run, or use --resume/--incremental")


def _engine_environment() -> dict[str, str]:
    # Upstream reads UTF-8 TOML/prompts with the locale default on Windows.
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    environment.pop("FM_AGENT_RESUME", None)
    return environment


def _run_scope(plan: dict[str, Any]) -> dict[str, Any]:
    return {name: plan[name] for name in ("framework", "mode", "submodules")}


def execute_plan(plan: dict[str, Any], args: argparse.Namespace) -> int:
    """Launch the official engine and retain its exit status and original report format.

    Args:
        plan: Launch plan.
        args: Parsed run options.

    Returns:
        Official process exit status, or 1 when no completion artifact was produced.
    """
    _check_run_state(plan, args)
    check_engine(plan)
    artifact_name = "estimate.json" if args.mode == "estimate" else "version.log"
    artifact = Path(plan["workspace"]) / artifact_name
    previous = (artifact.stat().st_mtime_ns, artifact.stat().st_size) if artifact.is_file() else None
    try:
        result = subprocess.run(plan["command"], cwd=plan["engine_root"], env=_engine_environment(), check=False)
    finally:
        if args.mode != "estimate" and Path(plan["workspace"]).is_dir():
            manifest = Path(plan["workspace"]) / "training_adapter.json"
            manifest.write_text(json.dumps(_run_scope(plan), indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        return result.returncode
    current = (artifact.stat().st_mtime_ns, artifact.stat().st_size) if artifact.is_file() else None
    if current is None or current == previous:
        LOGGER.error("FM-Agent exited without updating %s; this is not a successful analysis", artifact_name)
        return 1
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Define the adapter commands and their shared framework and engine options.

    Returns:
        Argument parser for CLI use and embedded invocation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "doctor", "run"):
        command = commands.add_parser(name)
        command.add_argument("--repo", default=str(REPO_ROOT))
        command.add_argument("--config", default=str(SKILL_ROOT / "profiles.json"))
        command.add_argument("--profile", default="dtensor")
        command.add_argument("--submodule", nargs="+", help="Override the profile's source directories")
        command.add_argument(
            "--knowledge", nargs="+", help="Additional Markdown inputs, relative to the launch directory",
        )
        command.add_argument("--engine-root", default=os.environ.get("FM_AGENT_HOME"))
        command.add_argument("--engine-python", default=sys.executable)
        command.add_argument("--mode", choices=("estimate", "spec", "verify"), default="verify")
        reuse = command.add_mutually_exclusive_group()
        reuse.add_argument("--resume", action="store_true")
        reuse.add_argument("--incremental", metavar="INTENT_FILE")
        if name == "run":
            command.add_argument("--dry-run", action="store_true", help="Print the plan without any engine calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the training-framework adapter CLI.

    Args:
        argv: Optional command-line argument list for embedding or tests.

    Returns:
        Zero on success, non-zero on setup or engine failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = create_parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.action == "doctor":
            check_engine(plan)
            LOGGER.info("Official FM-Agent dependencies and CLI options are compatible")
        elif args.action == "run" and not args.dry_run:
            return execute_plan(plan, args)
        else:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
