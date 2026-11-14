#!/usr/bin/env python3
"""
Validate moe-profiler skill structure

Checks:
1. SKILL.md exists and has required frontmatter
2. All references exist
3. All scripts exist
4. Scripts are importable (syntax check)
"""

import sys
from pathlib import Path
import re

SKILL_DIR = Path(__file__).parent


def check_skill_md():
    """Check SKILL.md structure"""
    skill_md = SKILL_DIR / "SKILL.md"

    if not skill_md.exists():
        print("❌ SKILL.md not found")
        return False

    with open(skill_md) as f:
        content = f.read()

    # Check frontmatter
    if not content.startswith("---"):
        print("❌ SKILL.md missing frontmatter")
        return False

    # Extract frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not frontmatter_match:
        print("❌ SKILL.md frontmatter malformed")
        return False

    frontmatter = frontmatter_match.group(1)

    # Check required fields
    required = ['name:', 'description:']
    for field in required:
        if field not in frontmatter:
            print(f"❌ SKILL.md missing required field: {field}")
            return False

    print("✓ SKILL.md structure valid")
    return True


def check_references():
    """Check references exist"""
    references_dir = SKILL_DIR / "references"

    if not references_dir.exists():
        print("❌ references/ directory not found")
        return False

    expected_files = [
        "moe-metrics.md",
        "load-imbalance-patterns.md",
        "communication-overhead-analysis.md",
    ]

    for file in expected_files:
        if not (references_dir / file).exists():
            print(f"❌ Missing reference: {file}")
            return False

    print(f"✓ All {len(expected_files)} reference files exist")
    return True


def check_scripts():
    """Check scripts exist and are importable"""
    scripts_dir = SKILL_DIR / "scripts"

    if not scripts_dir.exists():
        print("❌ scripts/ directory not found")
        return False

    expected_scripts = [
        "extract_moe_facts.py",
        "_moe_load_analysis.py",
        "_moe_comm_analysis.py",
        "_moe_routing_check.py",
    ]

    for script in expected_scripts:
        script_path = scripts_dir / script
        if not script_path.exists():
            print(f"❌ Missing script: {script}")
            return False

        # Syntax check (compile)
        try:
            with open(script_path) as f:
                compile(f.read(), script_path, 'exec')
        except SyntaxError as e:
            print(f"❌ Syntax error in {script}: {e}")
            return False

    print(f"✓ All {len(expected_scripts)} scripts exist and are valid Python")
    return True


def check_readme():
    """Check README exists"""
    readme = SKILL_DIR / "README.md"

    if not readme.exists():
        print("❌ README.md not found")
        return False

    print("✓ README.md exists")
    return True


def main():
    print("Validating moe-profiler skill structure...\n")

    checks = [
        check_skill_md,
        check_references,
        check_scripts,
        check_readme,
    ]

    results = [check() for check in checks]

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed")
        return 0
    else:
        print("❌ Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
