#!/usr/bin/env python3
"""
Validate moe-load-balancer agent
"""

import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent

def test_agent_file():
    """Test agent file exists and has required sections"""
    print("\n[Test 1] Agent File Structure")
    print("=" * 60)

    agent_file = AGENT_DIR / "moe-load-balancer.md"

    if not agent_file.exists():
        print(f"  ✗ moe-load-balancer.md not found")
        return False

    with open(agent_file) as f:
        content = f.read()

    # Check frontmatter
    if not content.startswith("---"):
        print(f"  ✗ Missing frontmatter")
        return False

    required_sections = [
        "name: moe-load-balancer",
        "description:",
        "# MoE Load Balancer Agent",
        "## Diagnosis Logic",
        "## Fix Implementation Guide",
        "## Output Format",
    ]

    for section in required_sections:
        if section in content:
            print(f"  ✓ Has '{section}'")
        else:
            print(f"  ✗ Missing '{section}'")
            return False

    size_kb = len(content) / 1024
    print(f"  ✓ File size: {size_kb:.1f} KB")

    return True


def test_diagnosis_logic():
    """Test diagnosis logic is comprehensive"""
    print("\n[Test 2] Diagnosis Logic")
    print("=" * 60)

    agent_file = AGENT_DIR / "moe-load-balancer.md"

    with open(agent_file) as f:
        content = f.read()

    required_patterns = [
        "HEAVY_TAIL",
        "BIMODAL",
        "LAYER_STRATIFIED",
    ]

    for pattern in required_patterns:
        if pattern in content:
            print(f"  ✓ Pattern: {pattern}")
        else:
            print(f"  ✗ Missing pattern: {pattern}")
            return False

    return True


def test_recommendations():
    """Test recommendations cover key fixes"""
    print("\n[Test 3] Recommendations")
    print("=" * 60)

    agent_file = AGENT_DIR / "moe-load-balancer.md"

    with open(agent_file) as f:
        content = f.read()

    required_fixes = [
        "capacity_factor",
        "expert_bias",
        "load_balance_loss",
        "top_k",
    ]

    for fix in required_fixes:
        if fix in content:
            print(f"  ✓ Fix: {fix}")
        else:
            print(f"  ✗ Missing fix: {fix}")
            return False

    return True


def test_references():
    """Test references to skills"""
    print("\n[Test 4] Skill References")
    print("=" * 60)

    agent_file = AGENT_DIR / "moe-load-balancer.md"

    with open(agent_file) as f:
        content = f.read()

    required_refs = [
        "moe-optimizer",
        "moe-profiler",
        "optimization-strategies.md",
    ]

    for ref in required_refs:
        if ref in content:
            print(f"  ✓ References: {ref}")
        else:
            print(f"  ✗ Missing reference: {ref}")
            return False

    return True


def main():
    print("=" * 60)
    print("MoE Load Balancer Agent Validation")
    print("=" * 60)

    tests = [
        ("Agent File Structure", test_agent_file),
        ("Diagnosis Logic", test_diagnosis_logic),
        ("Recommendations", test_recommendations),
        ("Skill References", test_references),
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
        print("\n  ✅ Agent validated!")
        return 0
    else:
        print(f"\n  ❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
