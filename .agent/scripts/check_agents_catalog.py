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

Exit 0 if catalogs match; non-zero with a short report otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _table_names(section: str) -> set[str]:
    """Extract bold names from a markdown table section."""
    return set(re.findall(r"\|\s*\*\*([a-z0-9-]+)\*\*", section, flags=re.I))


def main() -> int:
    """Compare AGENTS.md catalogs to on-disk skills and agents."""
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

    if errors:
        print("AGENTS catalog check FAILED:")
        for line in errors:
            print(" ", line)
        return 1
    print(
        f"AGENTS catalog OK ({len(disk_skills)} skills, {len(disk_agents)} agents)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
