#!/usr/bin/env python3
"""
Comprehensive validation for routing-validator skill
"""

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent


def test_structure():
    """Test skill structure"""
    print("\n[Test 1] Skill Structure")
    print("=" * 60)

    required_files = {
        "SKILL.md": SKILL_DIR / "SKILL.md",
        "README.md": SKILL_DIR / "README.md",
        "scripts/validate_routing.py": SKILL_DIR / "scripts/validate_routing.py",
        "scripts/_token_integrity.py": SKILL_DIR / "scripts/_token_integrity.py",
        "scripts/_gradient_flow.py": SKILL_DIR / "scripts/_gradient_flow.py",
        "scripts/_expert_coverage.py": SKILL_DIR / "scripts/_expert_coverage.py",
        "references/correctness-invariants.md": SKILL_DIR / "references/correctness-invariants.md",
        "references/gradient-flow-patterns.md": SKILL_DIR / "references/gradient-flow-patterns.md",
    }

    for name, path in required_files.items():
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} MISSING")
            return False

    print(f"\n  ✓ All {len(required_files)} files present")
    return True


def test_syntax():
    """Test Python syntax"""
    print("\n[Test 2] Python Syntax")
    print("=" * 60)

    scripts = [
        "scripts/validate_routing.py",
        "scripts/_token_integrity.py",
        "scripts/_gradient_flow.py",
        "scripts/_expert_coverage.py",
    ]

    for script in scripts:
        path = SKILL_DIR / script
        try:
            with open(path) as f:
                compile(f.read(), path, 'exec')
            print(f"  ✓ {script} syntax valid")
        except SyntaxError as e:
            print(f"  ✗ {script} syntax error: {e}")
            return False

    print(f"\n  ✓ All scripts have valid syntax")
    return True


def test_skill_md():
    """Test SKILL.md content"""
    print("\n[Test 3] SKILL.md Content")
    print("=" * 60)

    with open(SKILL_DIR / "SKILL.md") as f:
        content = f.read()

    required = [
        "# Routing Validator",
        "## Three Core Checks",
        "Token Integrity",
        "Routing Correctness",
        "Gradient Flow",
        "## Integration with moe-optimizer",
    ]

    for section in required:
        if section in content:
            print(f"  ✓ Has '{section}'")
        else:
            print(f"  ✗ Missing '{section}'")
            return False

    print(f"\n  ✓ SKILL.md complete")
    return True


def test_references():
    """Test reference documents"""
    print("\n[Test 4] Reference Documents")
    print("=" * 60)

    # Check invariants
    invariants = SKILL_DIR / "references/correctness-invariants.md"
    with open(invariants) as f:
        content = f.read()

    if "Invariant 1" in content and "Invariant 8" in content:
        print("  ✓ correctness-invariants.md has 8 invariants")
    else:
        print("  ✗ correctness-invariants.md incomplete")
        return False

    # Check patterns
    patterns = SKILL_DIR / "references/gradient-flow-patterns.md"
    with open(patterns) as f:
        content = f.read()

    if "Pattern 1" in content and "Pattern 7" in content:
        print("  ✓ gradient-flow-patterns.md has 7 patterns")
    else:
        print("  ✗ gradient-flow-patterns.md incomplete")
        return False

    print(f"\n  ✓ References complete")
    return True


def main():
    print("=" * 60)
    print("Routing Validator Skill Validation")
    print("=" * 60)

    tests = [
        ("Structure", test_structure),
        ("Python Syntax", test_syntax),
        ("SKILL.md", test_skill_md),
        ("References", test_references),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test failed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  ✅ All validations passed!")
        return 0
    else:
        print(f"\n  ❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
