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
"""Check optimizer structure and local links in every supporting Markdown file."""

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

SKILL_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = (
    "SKILL.md", "workflows/orchestration.md",
    "references/config-templates.md", "references/decision-rules.md",
    "references/psa-cost-model.md", "references/validation-metrics.md", "references/output-format.md",
    "scripts/run_optimization.py", "scripts/distributed_scheduler.py", "scripts/safety_checks.py",
    "scripts/metrics_output.py", "scripts/compare_benchmarks.py",
)


def markdown_targets(content: str) -> list:
    """Extract inline and reference destinations outside code fences/spans."""
    lines = []
    fence = None
    for line in content.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            lines.append(re.sub(r"(`+).*?\1", "", line))
    text = "\n".join(lines)
    destination = r'(<[^>\n]+>|(?:\\.|[^()\s]|\([^()]*\))+)'
    targets = re.findall(r'!?\[[^\]\n]*\]\(\s*' + destination, text)
    definitions = re.findall(r'^\s{0,3}\[[^\]\n]+\]:\s*' + destination, text, re.MULTILINE)
    return [target[1:-1] if target.startswith("<") else target for target in targets + definitions]


def broken_links(root: Path) -> list:
    """Return missing local destinations, resolved relative to each source file."""
    failures = []
    for source in sorted(root.rglob("*.md")):
        for target in set(markdown_targets(source.read_text(encoding="utf-8"))):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            decoded = unquote(parsed.path)
            decoded = re.sub(r"\\([() ])", r"\1", decoded)
            if not (source.parent / decoded).exists():
                failures.append((source, target))
    return failures


def check_internal_links(root: Path = SKILL_ROOT) -> bool:
    """Check all Markdown files under root, including nested reference documents."""
    failures = broken_links(root)
    for source, target in failures:
        print(f"FAIL: {source}: {target}")
    count = len(list(root.rglob("*.md")))
    print(f"Markdown links: {count} files, {len(failures)} missing targets")
    return not failures


def main() -> int:
    """Validate skill architecture, or validate links under an explicit root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links-root", type=Path, help="Check only Markdown links recursively under this directory")
    args = parser.parse_args()
    if args.links_root is not None:
        if not args.links_root.is_dir():
            parser.error("--links-root must be an existing directory")
        return 0 if check_internal_links(args.links_root.resolve()) else 1
    missing = [name for name in REQUIRED_FILES if not (SKILL_ROOT / name).is_file()]
    for name in missing:
        print(f"FAIL: missing {name}")
    content = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    frontmatter = content.split("---", 2)
    valid = len(frontmatter) == 3 and "name: strategy-optimizer" in frontmatter[1]
    valid = valid and "description:" in frontmatter[1]
    links_valid = check_internal_links()
    success = not missing and valid and links_valid
    print("Architecture: " + ("PASS" if success else "FAIL"))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
