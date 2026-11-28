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
#!/usr/bin/env python3
"""
Quick validation script for strategy-validator skill structure.

Checks:
- All required files present
- Markdown files are valid
- No broken internal links
- Skill structure matches parallel-strategy-analyzer format
"""

import os
import re
from pathlib import Path

SKILL_ROOT = Path(__file__).parent

def check_structure():
    """Verify directory structure and required files."""
    required_files = [
        "SKILL.md",
        "README.md",
        "workflows/01-config-generation.md",
        "workflows/02-benchmark-execution.md",
        "workflows/03-metrics-extraction.md",
        "workflows/04-comparison-report.md",
        "references/config-templates.md",
        "references/benchmark-recipes.md",
        "references/metrics-schema.md",
    ]

    print("=== Structure Check ===")
    missing = []
    for file in required_files:
        path = SKILL_ROOT / file
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} MISSING")
            missing.append(file)

    return len(missing) == 0

def check_markdown_links():
    """Check for broken internal links in markdown files."""
    print("\n=== Link Check ===")

    md_files = list(SKILL_ROOT.rglob("*.md"))
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

    broken = []
    for md_file in md_files:
        with open(md_file) as f:
            content = f.read()

        for match in link_pattern.finditer(content):
            link_text, link_path = match.groups()

            # Skip external links
            if link_path.startswith(('http://', 'https://', '#')):
                continue

            # Resolve relative path
            target = (md_file.parent / link_path).resolve()

            if not target.exists():
                broken.append(f"{md_file.name}: [{link_text}]({link_path})")

    if broken:
        print("✗ Broken links found:")
        for link in broken:
            print(f"  - {link}")
        return False
    else:
        print("✓ All internal links valid")
        return True

def check_skill_metadata():
    """Verify SKILL.md frontmatter."""
    print("\n=== Skill Metadata Check ===")

    skill_md = SKILL_ROOT / "SKILL.md"
    with open(skill_md) as f:
        content = f.read()

    # Check for required frontmatter fields
    required_fields = ["name:", "description:"]

    for field in required_fields:
        if field in content[:500]:  # Check first 500 chars (frontmatter)
            print(f"✓ {field} present")
        else:
            print(f"✗ {field} MISSING in frontmatter")
            return False

    # Check name matches directory
    name_match = re.search(r'name:\s*(\S+)', content)
    if name_match and name_match.group(1) == "strategy-validator":
        print("✓ Skill name matches directory")
    else:
        print("✗ Skill name mismatch")
        return False

    return True

def check_workflow_phases():
    """Verify workflow files match SKILL.md phase definitions."""
    print("\n=== Workflow Phase Check ===")

    skill_md = SKILL_ROOT / "SKILL.md"
    with open(skill_md) as f:
        skill_content = f.read()

    # Extract phase definitions from SKILL.md
    phases = re.findall(r'(\d+)\s+([^\s]+.*?)\s*→\s*workflows/(\d+.*?\.md)', skill_content)

    if not phases:
        print("✗ No phase definitions found in SKILL.md")
        return False

    all_exist = True
    for phase_num, phase_name, workflow_file in phases:
        workflow_path = SKILL_ROOT / "workflows" / workflow_file
        if workflow_path.exists():
            print(f"✓ Phase {phase_num}: {phase_name} → {workflow_file}")
        else:
            print(f"✗ Phase {phase_num}: {workflow_file} MISSING")
            all_exist = False

    return all_exist

def main():
    print("Strategy-Validator Skill Validation\n")

    checks = [
        ("Structure", check_structure),
        ("Markdown Links", check_markdown_links),
        ("Skill Metadata", check_skill_metadata),
        ("Workflow Phases", check_workflow_phases),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with error: {e}")
            results.append((name, False))

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    all_pass = all(result for _, result in results)

    print("\n" + ("="*50))
    if all_pass:
        print("✓ ALL CHECKS PASSED - Skill structure valid")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Review output above")
        return 1

if __name__ == "__main__":
    exit(main())
