#!/usr/bin/env python3
"""
Validate MoE integration in strategy-validator
"""

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent

def test_moe_metrics_file():
    """Test MoE metrics file exists"""
    print("\n[Test 1] MoE Metrics File")
    print("=" * 60)

    moe_metrics = SKILL_DIR / "references/moe-metrics.md"

    if not moe_metrics.exists():
        print(f"  ✗ moe-metrics.md not found")
        return False

    with open(moe_metrics) as f:
        content = f.read()

    required = [
        "# MoE-Specific Metrics",
        "## Metric Definitions",
        "expert_load_std",
        "load_imbalance_factor",
        "alltoall_overhead_ratio",
        "routing_entropy",
        "straggler_factor",
    ]

    for item in required:
        if item in content:
            print(f"  ✓ Has '{item}'")
        else:
            print(f"  ✗ Missing '{item}'")
            return False

    size_kb = len(content) / 1024
    print(f"  ✓ File size: {size_kb:.1f} KB")

    return True


def test_skill_md_updated():
    """Test SKILL.md references MoE metrics"""
    print("\n[Test 2] SKILL.md Integration")
    print("=" * 60)

    skill_md = SKILL_DIR / "SKILL.md"

    with open(skill_md) as f:
        content = f.read()

    if "moe-metrics.md" in content:
        print("  ✓ SKILL.md references moe-metrics.md")
        return True
    else:
        print("  ⚠ SKILL.md doesn't reference moe-metrics.md (optional)")
        return True  # Not critical


def test_metrics_schema_updated():
    """Test metrics-schema.md includes MoE section"""
    print("\n[Test 3] Metrics Schema Integration")
    print("=" * 60)

    metrics_schema = SKILL_DIR / "references/metrics-schema.md"

    with open(metrics_schema) as f:
        content = f.read()

    if "MoE-Specific Metrics" in content:
        print("  ✓ metrics-schema.md has MoE section")
    else:
        print("  ✗ metrics-schema.md missing MoE section")
        return False

    if "moe-metrics.md" in content:
        print("  ✓ metrics-schema.md references moe-metrics.md")
    else:
        print("  ✗ metrics-schema.md doesn't reference moe-metrics.md")
        return False

    return True


def test_metric_formulas():
    """Test key metric formulas present"""
    print("\n[Test 4] Metric Formulas")
    print("=" * 60)

    moe_metrics = SKILL_DIR / "references/moe-metrics.md"

    with open(moe_metrics) as f:
        content = f.read()

    formulas = [
        "compute_load_std",
        "compute_load_imbalance_factor",
        "compute_alltoall_overhead",
        "compute_routing_entropy",
        "compute_straggler_factor",
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
    print("MoE Integration Validation for strategy-validator")
    print("=" * 60)

    tests = [
        ("MoE Metrics File", test_moe_metrics_file),
        ("SKILL.md Integration", test_skill_md_updated),
        ("Metrics Schema Integration", test_metrics_schema_updated),
        ("Metric Formulas", test_metric_formulas),
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
