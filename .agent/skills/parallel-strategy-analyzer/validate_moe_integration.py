#!/usr/bin/env python3
"""
Validate MoE integration in parallel-strategy-analyzer
"""

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent

def test_moe_rules_exist():
    """Test MoE rules file exists"""
    print("\n[Test 1] MoE Rules File")
    print("=" * 60)

    moe_rules = SKILL_DIR / "references/moe-strategy-rules.md"

    if not moe_rules.exists():
        print(f"  ✗ moe-strategy-rules.md not found")
        return False

    # Check content
    with open(moe_rules) as f:
        content = f.read()

    required = [
        "# MoE-Specific Strategy Rules",
        "## Memory Estimation Adjustments",
        "## Communication Cost",
        "## Strategy Search Rules",
        "expert_activation_memory",
        "alltoall_cost",
        "load_imbalance_factor",
    ]

    for item in required:
        if item in content:
            print(f"  ✓ Has '{item}'")
        else:
            print(f"  ✗ Missing '{item}'")
            return False

    # Check file size
    size_kb = len(content) / 1024
    print(f"  ✓ File size: {size_kb:.1f} KB")

    if size_kb < 5:
        print(f"  ⚠ Warning: File seems small ({size_kb:.1f} KB)")

    return True


def test_skill_md_updated():
    """Test SKILL.md references MoE rules"""
    print("\n[Test 2] SKILL.md Integration")
    print("=" * 60)

    skill_md = SKILL_DIR / "SKILL.md"

    with open(skill_md) as f:
        content = f.read()

    if "moe-strategy-rules.md" in content:
        print("  ✓ SKILL.md references moe-strategy-rules.md")
    else:
        print("  ✗ SKILL.md doesn't reference moe-strategy-rules.md")
        return False

    if "is_moe" in content or "MoE-specific" in content:
        print("  ✓ SKILL.md mentions MoE-specific handling")
    else:
        print("  ⚠ SKILL.md doesn't explicitly mention MoE handling")

    return True


def test_workflow_updated():
    """Test workflow references MoE rules"""
    print("\n[Test 3] Workflow Integration")
    print("=" * 60)

    strategy_search = SKILL_DIR / "workflows/03-strategy-search.md"

    with open(strategy_search) as f:
        content = f.read()

    if "moe-strategy-rules.md" in content:
        print("  ✓ 03-strategy-search.md references MoE rules")
    else:
        print("  ✗ 03-strategy-search.md doesn't reference MoE rules")
        return False

    return True


def test_formulas():
    """Test key formulas present"""
    print("\n[Test 4] Key Formulas")
    print("=" * 60)

    moe_rules = SKILL_DIR / "references/moe-strategy-rules.md"

    with open(moe_rules) as f:
        content = f.read()

    formulas = [
        "expert_activation_memory",
        "alltoall_buffer_memory",
        "load_imbalance_factor",
        "select_ep_size",
        "select_capacity_factor",
    ]

    for formula in formulas:
        if formula in content:
            print(f"  ✓ Formula present: {formula}")
        else:
            print(f"  ✗ Missing formula: {formula}")
            return False

    return True


def main():
    print("=" * 60)
    print("MoE Integration Validation for PSA")
    print("=" * 60)

    tests = [
        ("MoE Rules File", test_moe_rules_exist),
        ("SKILL.md Integration", test_skill_md_updated),
        ("Workflow Integration", test_workflow_updated),
        ("Key Formulas", test_formulas),
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
        print("\n  ✅ MoE integration validated!")
        return 0
    else:
        print(f"\n  ❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
