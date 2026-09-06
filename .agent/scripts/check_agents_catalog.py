#!/usr/bin/env python3
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
"""Check AGENTS.md Skills/Agents tables against files under .agent/.

Also checks documentation links and RL navigation file/symbol references.
Text duplication checks are heuristic, not semantic validation. Scope conflicts,
configuration behavior and whether a test guards a feature still require review.
Exit 0 means these mechanical checks passed, not that all docs are correct.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def _table_names(section: str) -> set[str]:
    """Extract bold names from a markdown table section."""
    return set(re.findall(r"\|\s*\*\*([a-z0-9-]+)\*\*", section, flags=re.I))


def _links(text: str, base: Path) -> set[Path]:
    """Collect markdown link targets that resolve to local files.

    ``base`` is the directory the markdown file lives in; relative links resolve
    against it, so the returned paths are repo-relative (e.g. ``docs/api/x.md``).
    """
    links = set()
    for raw in re.findall(r"\]\(([^)]+)\)", text):
        target = raw.split("#")[0].split("?")[0]
        if not target or re.match(r"^[a-z]+://", target):
            continue
        resolved = (base / target).resolve()
        links.add(resolved)
    return links


def _broken_links(root: Path) -> list[str]:
    """Find markdown links that do not resolve to an existing file.

    Links are relative to the file that contains them, which is easy to get
    wrong for docs under ``.agent/rules/`` (a repo-root-relative path silently
    resolves into the rule directory). Returns ``"file -> target"`` strings.
    """
    broken = []
    targets = [root / "AGENTS.md"]
    targets += sorted((root / ".agent").rglob("*.md"))
    targets += sorted((root / "docs").rglob("*.md"))
    targets += sorted((root / "hyper_parallel/rl/docs").rglob("*.md"))
    for path in targets:
        if not path.is_file():
            continue
        for raw in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = raw.split("#")[0].split("?")[0]
            if not target or re.match(r"^[a-z]+:", target):
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(root)} -> {raw}")
    return broken


# Docs that are the single source of truth for a fact. If a fragment of one of
# these appears verbatim in another markdown file, that file is restating the
# fact instead of linking to it — the "one fact = one place" violation.
_DRIFT_SOURCES = ("rl-architecture.md", "rl-navigation.md")
# A normalized line shorter than this is a header or a single-token table cell,
# not a fact worth tracking.
_MIN_UNIT_CHARS = 30
# A copied line this long is specific enough that one occurrence already proves
# duplication; shorter lines need a second hit to rule out coincidental phrasing.
_LONG_UNIT_CHARS = 60
# Cap on how many drift lines we print, to keep failure output readable.
_MAX_REPORTED = 12


def _normalize(text: str) -> str:
    """Strip markdown/link noise and collapse whitespace, for fuzzy matching."""
    text = re.sub(r"(?m)^\s*(?:[-+*]|\d+[.)])\s+", "", text)  # list markers
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # [text](url) -> text
    text = re.sub(r"[*_`>#|]", " ", text)                     # emphasis / tables / code
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _doc_units(text: str) -> list[str]:
    """Split a source doc into fact-units: table rows and prose sentences.

    Args:
        text: Raw markdown of a canonical doc.

    Returns:
        One normalized string per retained line. Lines shorter than
        ``_MIN_UNIT_CHARS``, with fewer than two words, or with no letters are
        noise (headers, single-code-token rows) and are dropped.
    """
    units: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("<!--", "#", "---")):
            continue
        norm = _normalize(line)
        if (len(norm) < _MIN_UNIT_CHARS or len(norm.split()) < 2
                or not any(c.isalpha() for c in norm)):
            continue
        units.append(norm)
    return units


def _drift(root: Path) -> tuple[list[str], int]:
    """Find markdown files that restate facts from the canonical docs.

    A unit copied from ``docs/rl-architecture.md`` / ``docs/rl-navigation.md``
    into any other ``.md`` file is a duplication bug.

    Args:
        root: Repository root to scan.

    Returns:
        A ``(lines, total)`` pair: at most ``_MAX_REPORTED`` human-readable
        findings, and the total number of findings before truncation.
    """
    docs_dir = root / "docs"
    source_units: dict[str, list[str]] = {}
    for name in _DRIFT_SOURCES:
        doc = docs_dir / name
        if doc.exists():
            source_units[name] = _doc_units(doc.read_text(encoding="utf-8"))

    targets = sorted(
        p for p in root.rglob("*.md")
        if ".git" not in p.parts and ".agent" not in p.parts
        and p.name not in _DRIFT_SOURCES
    )

    reported: list[str] = []
    for path in targets:
        text = _normalize(path.read_text(encoding="utf-8"))
        for doc, units in source_units.items():
            found = [unit for unit in units if unit in text]
            has_long_unit = any(len(unit) >= _LONG_UNIT_CHARS for unit in found)
            # Two or more copied units, or a single long one, mean duplication
            # rather than coincidental phrase reuse.
            if len(found) >= 2 or has_long_unit:
                reported.append(f"{path.relative_to(root)} -> {doc}: {len(found)} unit(s) restated")
    return reported[:_MAX_REPORTED], len(reported)


def _symbol_error(path: Path, symbol: str) -> str:
    """Resolve a qualified definition within its lexical scope."""
    try:
        body = ast.parse(path.read_text(encoding="utf-8")).body
    except SyntaxError as error:
        return f"invalid Python: {error.msg}"
    for part in symbol.split("."):
        definition = next((node for node in body if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name == part), None)
        if definition is None:
            return f"missing symbol {symbol}"
        body = definition.body
    return ""


def _navigation_errors(root: Path) -> list[str]:
    """Check RL table references without importing training dependencies.

    Paths are relative to hyper_parallel/rl; bare test filenames resolve under
    rl_tests. Optional ::qualified.symbol references follow lexical definitions,
    so a method on the wrong class cannot satisfy a reference.
    """
    nav = root / "docs/rl-navigation.md"
    if not nav.is_file():
        return []  # main reports the missing navigation document.
    errors = []
    in_rl = False
    for number, line in enumerate(nav.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## 2. "):
            in_rl = True
        if not in_rl or not line.startswith("|"):
            continue
        for reference in re.findall(r"`([^`]+)`", line):
            match = re.fullmatch(r"([\w./-]+\.py)(?:::(\w+(?:\.\w+)*))?", reference)
            if not match:
                continue
            relative, symbol = match.groups()
            if "/" not in relative and relative.startswith("test_"):
                relative = "rl_tests/" + relative
            path = root / "hyper_parallel/rl" / relative
            label = f"docs/rl-navigation.md:{number} -> {reference}"
            if not path.is_file():
                errors.append(f"{label}: missing file")
                continue
            if not symbol:
                continue
            error = _symbol_error(path, symbol)
            if error:
                errors.append(f"{label}: {error}")
    return errors


def _doc_topology_errors(root: Path, agents_md: str) -> list[str]:
    """Validate the navigation entry and documentation index."""
    errors: list[str] = []
    # Single-source doc topology: navigation map + no orphan docs.
    docs_dir = root / "docs"
    nav = docs_dir / "rl-navigation.md"
    if not nav.exists():
        errors.append("docs: docs/rl-navigation.md missing (required traceability map)")
    else:
        if "rl-navigation.md" not in agents_md:
            errors.append("docs: AGENTS.md does not link docs/rl-navigation.md")

    index = docs_dir / "index.md"
    docs_reachable: set[Path] = set()
    if index.exists():
        docs_reachable = _links(index.read_text(encoding="utf-8"), docs_dir)
    on_disk = {p.resolve() for p in docs_dir.rglob("*.md") if "_" not in p.name}
    canonical = {"index.md", "rl-architecture.md", "rl-navigation.md"}
    orphans = sorted(
        p for p in on_disk if p not in docs_reachable and p.name not in canonical
    )
    if orphans:
        errors.append(f"docs orphan (not linked from docs/index.md): {[str(p) for p in orphans]}")

    return errors


def main() -> int:
    """Compare AGENTS.md catalogs to on-disk skills and agents, and check docs."""
    root = Path(__file__).resolve().parents[2]
    agents_md = (root / "AGENTS.md").read_text(encoding="utf-8")
    skills_sec = agents_md.split("### Skills")[1].split("### Commands")[0]
    agents_sec = agents_md.split("### Agents")[1].split("### Rules")[0]
    listed_skills = _table_names(skills_sec)
    listed_agents = _table_names(agents_sec)

    disk_skills = {p.parent.name for p in (root / ".agent/skills").glob("*/SKILL.md")}
    disk_agents = {
        p.stem
        for p in (root / ".agent/agents").glob("*.md")
        if not p.stem.endswith("-guide")
    }
    disk_rules = {p.stem for p in (root / ".agent/rules").glob("*.md")}

    errors: list[str] = []
    if listed_skills != disk_skills:
        errors.append(
            f"skills mismatch: only_in_AGENTS={sorted(listed_skills - disk_skills)} "
            f"only_on_disk={sorted(disk_skills - listed_skills)}"
        )
    if listed_agents != disk_agents:
        errors.append(
            f"agents mismatch: only_in_AGENTS={sorted(listed_agents - disk_agents)} "
            f"only_on_disk={sorted(disk_agents - listed_agents)}"
        )

    # Readability skill/rule must be registered wherever it is invoked.
    if "readability-first" not in listed_skills and "readability-first" in disk_skills:
        errors.append("skills: 'readability-first' exists on disk but is not listed in AGENTS.md")
    if "readability" not in disk_rules:
        errors.append("rules: 'readability' rule file is missing under .agent/rules/")

    errors.extend(_doc_topology_errors(root, agents_md))

    errors.extend(_navigation_errors(root))

    broken = _broken_links(root)
    if broken:
        errors.append(f"broken markdown links ({len(broken)}): {broken[:10]}")

    drift, drift_total = _drift(root)
    if drift:
        errors.append(
            f"doc drift: facts restated outside canonical docs "
            f"({drift_total} total, showing {len(drift)}):"
        )
        errors.extend(drift)

    if errors:
        print("AGENTS catalog / doc topology check FAILED:")
        for line in errors:
            print(" ", line)
        return 1
    print(
        f"AGENTS catalog OK ({len(disk_skills)} skills, {len(disk_agents)} agents, "
        f"{len(disk_rules)} rules); checked links and RL navigation references OK "
        "(semantic review still required)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
