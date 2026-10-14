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
"""Put the sapp-parallel-strategy where a coding assistant looks for a skill, in one command.

The skill ships inside the repository, at ``.agent/skills/sapp-parallel-strategy``, and no assistant
looks there by itself. This script links it into the folders they do read, checks that the
checkout can price a layout, and prints the next command::

    python .agent/skills/sapp-parallel-strategy/install.py            # install, then report
    python .agent/skills/sapp-parallel-strategy/install.py --check    # report only, change nothing
    python .agent/skills/sapp-parallel-strategy/install.py --copy     # copy where a link cannot be made

It never downloads anything and never writes inside the repository except where asked.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

SKILL = Path(__file__).resolve().parent
NAME = SKILL.name
# Where each assistant reads skills from. "project" is inside the checkout, "user" is the
# home folder, which is what an assistant started in another folder finds.
PLACES: Dict[str, Dict[str, str]] = {
    "codex": {"user": ".agents/skills", "project": ".agents/skills", "ask": "$sapp-parallel-strategy"},
    "claude-code": {"user": ".claude/skills", "project": ".claude/skills", "ask": "/sapp-parallel-strategy"},
}
# What each assistant gets by default: the scope its own documentation names first.
DEFAULT_SCOPE = {"codex": "user", "claude-code": "project"}


class Target(NamedTuple):
    """One place the skill is installed into."""

    assistant: str
    scope: str
    path: Path

    @property
    def ask(self) -> str:
        """How a request reaches the skill once it is installed there."""
        return PLACES[self.assistant]["ask"]


def checkout_root(start: Path) -> Optional[Path]:
    """The repository checkout above ``start``, or None when there is none."""
    for folder in [start, *start.parents]:
        if (folder / "hyper_parallel" / "__init__.py").is_file():
            return folder
    return None


def targets(assistants: Sequence[str], scope: Optional[str], root: Path) -> List[Target]:
    """One target per assistant, at the scope asked for or the one its documentation names."""
    out = []
    for assistant in assistants:
        where = scope or DEFAULT_SCOPE[assistant]
        base = root if where == "project" else Path.home()
        out.append(Target(assistant, where, base / PLACES[assistant][where] / NAME))
    return out


def _link(target: Path, source: Path) -> str:
    """Make ``target`` point at ``source``, by symbolic link or, on Windows, a junction."""
    try:
        target.symlink_to(source, target_is_directory=True)
        return "linked"
    except (OSError, NotImplementedError) as problem:
        if os.name != "nt":
            raise OSError(f"{problem}") from problem
    # Windows asks for a privilege for a symbolic link, and for none for a directory junction.
    done = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(source)],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise OSError(done.stderr.strip() or "mklink failed")
    return "junctioned"


def _copy(target: Path, source: Path, root: Path) -> str:
    """Copy the skill to ``target`` and leave a pointer to the checkout its scripts must run."""
    shutil.copytree(source, target)
    (target / "checkout.txt").write_text(f"{root}\n", encoding="utf-8")
    return "copied"


def _linked(target: Path, source: Path) -> bool:
    """True when ``target`` is a link that stands for ``source``."""
    try:
        return target.exists() and target.resolve() == source
    except OSError:
        return False


def _our_copy(target: Path) -> bool:
    """True when ``target`` is a copy this script made, which it may replace with a fresh one."""
    return target.is_dir() and (target / "SKILL.md").is_file() and (target / "checkout.txt").is_file()


def _remove(path: Path) -> None:
    """Remove what sits at ``path``, dropping a link without following it."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def installed_state(target: Target, source: Path) -> str:
    """What ``--check`` says about one target: a link to the skill, a copy of it, or something else."""
    if _linked(target.path, source):
        return "installed"
    if _our_copy(target.path):
        return "installed as a copy"
    if target.path.exists() or target.path.is_symlink():
        return "something else is there"
    return "not installed"


def install_one(target: Target, source: Path, root: Path, copy: bool, force: bool) -> str:
    """Install the skill at one target. Returns what was done, for the report."""
    path = target.path
    if _linked(path, source) and not copy:
        return "already installed"
    ours = _linked(path, source) or _our_copy(path)
    there = path.exists() or path.is_symlink()
    if there and not (ours or force):
        return f"left alone: {path} exists and is not this skill, pass --force to replace it"
    _remove(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    note = "" if not there else (", refreshed" if ours else ", replacing what was there")
    if copy:
        return _copy(path, source, root) + note
    try:
        return _link(path, source) + note
    except OSError as problem:
        return f"not installed: no link could be made ({problem}), run again with --copy"


def dependency_report(root: Path) -> List[str]:
    """What the checkout can do, one line each: the Python, PyYAML, the cost model, the V4.1 model."""
    lines = [f"python {sys.version.split()[0]}, checkout {root}"]
    try:
        import yaml  # pylint: disable=import-outside-toplevel,unused-import
        lines.append("PyYAML: found")
    except ImportError:
        lines.append("PyYAML: MISSING, the scripts cannot read a template (pip install pyyaml)")
    model = root / "hyper_parallel" / "auto_parallel" / "sapp_nd" / "nd" / "run_nd.py"
    lines.append("cost model: found" if model.is_file() else f"cost model: MISSING at {model}")
    v41 = root / "hyper_parallel" / "models" / "deepseek_v41"
    lines.append("deepseek_v41 model: found, its crop is priced with a census of the layers"
                 if v41.is_dir() else
                 "deepseek_v41 model: not in this checkout, its crop is priced from the description instead")
    return lines


def self_test(source: Path) -> str:
    """Run the asking script's help through the installed path, which proves it finds its checkout."""
    script = source / "scripts" / "ask.py"
    done = subprocess.run([sys.executable, str(script), "--help"],
                          capture_output=True, text=True, check=False)
    if done.returncode == 0:
        return "self test: ask.py runs"
    tail = (done.stderr or done.stdout).strip().splitlines()
    return "self test: ask.py FAILED, " + (tail[-1] if tail else f"exit {done.returncode}")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """The command line of the installer."""
    parser = argparse.ArgumentParser(description="Install the sapp-parallel-strategy skill for a coding assistant.")
    parser.add_argument("--for", dest="assistants", default="all",
                        help="comma separated: codex, claude-code, or all (the default)")
    parser.add_argument("--scope", choices=("user", "project"), default=None,
                        help="install under the home folder or inside the checkout "
                             "(the default is each assistant's own first choice)")
    parser.add_argument("--copy", action="store_true",
                        help="copy the folder instead of linking it, for a filesystem without links")
    parser.add_argument("--force", action="store_true",
                        help="replace whatever already sits at the target")
    parser.add_argument("--check", action="store_true",
                        help="report what is installed and what the checkout can do, change nothing")
    return parser.parse_args(argv)


def chosen(assistants: str) -> List[str]:
    """The assistants named on the command line."""
    if assistants == "all":
        return list(PLACES)
    names = [name.strip() for name in assistants.split(",") if name.strip()]
    unknown = [name for name in names if name not in PLACES]
    if unknown:
        raise SystemExit(f"unknown assistant {unknown[0]!r}, pick from {', '.join(PLACES)} or all")
    return names


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Install the skill, or report, and print the first command to run."""
    arguments = parse_arguments(argv)
    root = checkout_root(SKILL)
    if root is None:
        raise SystemExit(f"no HyperParallel checkout above {SKILL}: run this from a checkout of the repository")

    lines = []
    for target in targets(chosen(arguments.assistants), arguments.scope, root):
        state = installed_state(target, SKILL) if arguments.check \
            else install_one(target, SKILL, root, arguments.copy, arguments.force)
        lines.append(f"{target.assistant} ({target.scope}): {state}")
        lines.append(f"    {target.path}")
        lines.append(f"    ask it with {target.ask}")

    print("\n".join(lines))
    print()
    print("\n".join(dependency_report(root)))
    print(self_test(SKILL))
    print()
    print("First question, from the root of this checkout:")
    print("    python .agent/skills/sapp-parallel-strategy/scripts/ask.py --family qwen3_5_moe --cluster a3 \\")
    print("        --layers 8 --seq 8192 --global-batch 64 --micro-batch 1 --dies 64 \\")
    print('        --request "which layout is fastest on 64 dies"')
    print("Or ask the assistant itself, which runs the same script and reads the answer back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
