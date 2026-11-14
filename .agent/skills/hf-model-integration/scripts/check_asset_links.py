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
"""Check both directions between a skill's prose and the assets it owns.

Two failure modes, equal and opposite:

* a **dangling reference** — doctrine names a script or document that does
  not exist, so an agent following it stops;
* an **unreferenced asset** — the file exists but no clause mentions it, so
  an agent following the doctrine never learns it is there and rebuilds it.

The first announces itself the moment someone runs the command. The second
is silent: nothing fails, capability is simply unreachable. Both are checked
here, because a review that only resolves references sees half the problem.

    python check_asset_links.py --skill-dir .agent/skills/<name>             [--also examples/training_demo/<kit>]

Exit 0 when every reference resolves and every asset is referenced; 2
otherwise, naming each violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REFERENCE = re.compile(r"[\w./-]*[\w-]+\.(?:py|sh|yaml|md)")
ASSET_SUFFIXES = (".py", ".sh", ".yaml")
SKIP_NAMES = {"__init__.py"}


def prose_files(skill_dir: Path) -> list[Path]:
    """Markdown that carries the doctrine."""
    return sorted(skill_dir.rglob("*.md"))


def asset_files(roots: list[Path]) -> list[Path]:
    """Executable and declarative assets the doctrine is supposed to name."""
    found: list[Path] = []
    for root in roots:
        for suffix in ASSET_SUFFIXES:
            found += [p for p in root.rglob(f"*{suffix}")
                      if p.name not in SKIP_NAMES]
    return sorted(set(found))


def mentioned_names(prose: list[Path]) -> set[str]:
    """Every file-like token the prose mentions, reduced to base names."""
    names: set[str] = set()
    for document in prose:
        text = document.read_text(encoding="utf-8", errors="ignore")
        for match in REFERENCE.findall(text):
            names.add(Path(match).name)
    return names


def dangling_references(prose: list[Path], roots: list[Path]) -> list[str]:
    """References into the skill's own asset roots that resolve to nothing.

    Scope matters more than reach here. Prose legitimately names files this
    check has no business validating: illustrations (`pkill -f
    "[r]un_x.sh"`), external toolchain paths (`.../set_env.sh`), and
    artifacts the workflow *produces* at run time
    (`analysis/optimization_plan.yaml`). Flagging those yields noise, and a
    check whose output is mostly noise is one readers learn to skip. So only
    a reference pointing into a directory this skill owns is treated as a
    claim it must honour.
    """
    owned = {root.name for root in roots} | {"scripts", "templates",
                                             "references"}
    problems = []
    for document in prose:
        text = document.read_text(encoding="utf-8", errors="ignore")
        for match in REFERENCE.findall(text):
            if "/" not in match or any(ch in match for ch in "[]*<>"):
                continue
            parts = Path(match).parts
            if not any(part in owned for part in parts[:-1]):
                continue
            if any((root / Path(match).name).exists()
                   or any(root.rglob(Path(match).name)) for root in roots):
                continue
            problems.append(f"{document.name} references missing {match}")
    return sorted(set(problems))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check doctrine<->asset references in both directions.")
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--also", action="append", default=[],
                        help="extra asset root the skill owns (a kit directory)")
    args = parser.parse_args(argv)

    skill_dir = Path(args.skill_dir)
    if not skill_dir.is_dir():
        print(f"[links] no such skill directory: {skill_dir}", flush=True)
        return 1
    roots = [skill_dir] + [Path(extra) for extra in args.also]
    missing_roots = [str(r) for r in roots if not r.is_dir()]
    if missing_roots:
        print(f"[links] no such asset root: {', '.join(missing_roots)}", flush=True)
        return 1

    prose = prose_files(skill_dir)
    assets = asset_files(roots)
    mentioned = mentioned_names(prose)

    unreferenced = [p for p in assets if p.name not in mentioned]
    dangling = dangling_references(prose, roots)

    print(f"[links] prose={len(prose)} assets={len(assets)}", flush=True)
    for problem in dangling:
        print(f"[links] FAIL dangling: {problem}", flush=True)
    for asset in unreferenced:
        print(f"[links] FAIL unreferenced: {asset.as_posix()} "
              "(exists but no clause names it)", flush=True)
    if not dangling and not unreferenced:
        print("[links] PASS: every reference resolves, every asset is named",
              flush=True)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
