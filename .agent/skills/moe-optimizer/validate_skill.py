#!/usr/bin/env python3
"""
Comprehensive validation for moe-optimizer skill

Tests:
1. Skill structure completeness
2. Workflow file consistency
3. Reference document integrity
4. Integration with moe-profiler
5. Simulated optimization flow
"""

import sys
import json
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
PROFILER_DIR = SKILL_DIR.parent / "moe-profiler"


def test_structure():
    """Test 1: Skill structure completeness"""
    print("\n[Test 1] Skill Structure Completeness")
    print("=" * 60)

    required_files = {
        "SKILL.md": SKILL_DIR / "SKILL.md",
        "README.md": SKILL_DIR / "README.md",
        "workflows/01-baseline-profiling.md": SKILL_DIR / "workflows/01-baseline-profiling.md",
        "workflows/02-problem-diagnosis.md": SKILL_DIR / "workflows/02-problem-diagnosis.md",
        "workflows/03-strategy-generation.md": SKILL_DIR / "workflows/03-strategy-generation.md",
        "workflows/04-validation.md": SKILL_DIR / "workflows/04-validation.md",
        "workflows/05-iterative-refinement.md": SKILL_DIR / "workflows/05-iterative-refinement.md",
        "references/optimization-strategies.md": SKILL_DIR / "references/optimization-strategies.md",
        "references/case-studies.md": SKILL_DIR / "references/case-studies.md",
        "references/workflow.md": SKILL_DIR / "references/workflow.md",
    }

    all_present = True
    for name, path in required_files.items():
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {name} ({size} bytes)")
        else:
            print(f"  ✗ {name} MISSING")
            all_present = False

    assert all_present, "Some required files are missing"
    print(f"\n  ✓ All {len(required_files)} required files present")
    return True


def test_skill_md_structure():
    """Test 2: SKILL.md has required sections"""
    print("\n[Test 2] SKILL.md Structure")
    print("=" * 60)

    skill_md = SKILL_DIR / "SKILL.md"
    with open(skill_md) as f:
        content = f.read()

    required_sections = [
        "# MoE Optimizer",
        "## When to use",
        "## Dependencies",
        "## Execution Flow",
        "## Workflow References",
        "## References",
        "## Example Usage",
        "## Output Format",
        "## Handoff",
    ]

    for section in required_sections:
        if section in content:
            print(f"  ✓ {section}")
        else:
            print(f"  ✗ {section} MISSING")
            return False

    # Check frontmatter
    if content.startswith("---"):
        print("  ✓ Has frontmatter")
        if "name: moe-optimizer" in content:
            print("  ✓ Correct skill name")
        if "description:" in content:
            print("  ✓ Has description")
    else:
        print("  ✗ Missing frontmatter")
        return False

    print(f"\n  ✓ SKILL.md structure valid")
    return True


def test_workflow_consistency():
    """Test 3: Workflow files are consistent"""
    print("\n[Test 3] Workflow Consistency")
    print("=" * 60)

    workflows = [
        ("01-baseline-profiling.md", ["Phase 1", "moe-profiler", "baseline_metrics"]),
        ("02-problem-diagnosis.md", ["Phase 2", "load_imbalance", "communication_bottleneck"]),
        ("03-strategy-generation.md", ["Phase 3", "parallel-strategy-analyzer", "Top-3"]),
        ("04-validation.md", ["Phase 4", "strategy-validator", "routing-validator"]),
        ("05-iterative-refinement.md", ["Phase 5", "refinement", "iteration"]),
    ]

    for filename, required_terms in workflows:
        filepath = SKILL_DIR / "workflows" / filename
        with open(filepath) as f:
            content = f.read()

        all_present = all(term in content for term in required_terms)
        if all_present:
            print(f"  ✓ {filename} has required terms")
        else:
            missing = [t for t in required_terms if t not in content]
            print(f"  ✗ {filename} missing: {missing}")
            return False

    print(f"\n  ✓ All {len(workflows)} workflows consistent")
    return True


def test_references_completeness():
    """Test 4: Reference documents are complete"""
    print("\n[Test 4] Reference Documents Completeness")
    print("=" * 60)

    # Check optimization-strategies.md
    strategies = SKILL_DIR / "references/optimization-strategies.md"
    with open(strategies) as f:
        content = f.read()

    required_problems = [
        "Load Imbalance",
        "Communication Bottleneck",
        "Stragglers",
        "Memory Pressure",
    ]

    for problem in required_problems:
        if problem in content:
            print(f"  ✓ Strategy for '{problem}'")
        else:
            print(f"  ✗ Missing strategy for '{problem}'")
            return False

    # Check case-studies.md
    case_studies = SKILL_DIR / "references/case-studies.md"
    with open(case_studies) as f:
        content = f.read()

    if "Case Study" in content:
        num_cases = content.count("## Case Study")
        print(f"  ✓ {num_cases} case studies documented")
    else:
        print(f"  ✗ No case studies found")
        return False

    print(f"\n  ✓ References complete")
    return True


def test_moe_profiler_integration():
    """Test 5: Integration with moe-profiler"""
    print("\n[Test 5] moe-profiler Integration")
    print("=" * 60)

    # Check if moe-profiler exists
    if not PROFILER_DIR.exists():
        print("  ✗ moe-profiler skill not found")
        return False

    print(f"  ✓ moe-profiler skill found at {PROFILER_DIR}")

    # Check if moe-optimizer references moe-profiler correctly
    skill_md = SKILL_DIR / "SKILL.md"
    with open(skill_md) as f:
        content = f.read()

    if "moe-profiler" in content:
        print("  ✓ moe-optimizer references moe-profiler")
    else:
        print("  ✗ moe-optimizer doesn't reference moe-profiler")
        return False

    # Check baseline profiling workflow
    baseline_wf = SKILL_DIR / "workflows/01-baseline-profiling.md"
    with open(baseline_wf) as f:
        content = f.read()

    if "moe-profiler" in content and "extract_moe_facts.py" in content:
        print("  ✓ Baseline workflow correctly invokes moe-profiler")
    else:
        print("  ✗ Baseline workflow doesn't properly invoke moe-profiler")
        return False

    print(f"\n  ✓ Integration verified")
    return True


def test_simulated_flow():
    """Test 6: Simulated optimization flow"""
    print("\n[Test 6] Simulated Optimization Flow")
    print("=" * 60)

    # Simulate the 5-phase workflow
    print("\n  Simulating optimization workflow:")

    # Phase 1: Baseline Profiling
    print("    Phase 1: Baseline Profiling")
    baseline_metrics = {
        "throughput": 94000,
        "load_imbalance_factor": 1.68,
        "alltoall_ratio": 0.32,
    }
    print(f"      → Extracted metrics: {baseline_metrics}")

    # Phase 2: Problem Diagnosis
    print("    Phase 2: Problem Diagnosis")
    problems = []
    if baseline_metrics["load_imbalance_factor"] > 1.5:
        problems.append("load_imbalance")
    if baseline_metrics["alltoall_ratio"] > 0.3:
        problems.append("communication_bottleneck")
    print(f"      → Identified problems: {problems}")

    # Phase 3: Strategy Generation
    print("    Phase 3: Strategy Generation")
    candidates = [
        {"tp": 2, "ep": 4, "dp": 8, "estimated_throughput": 115000},
        {"tp": 4, "ep": 4, "dp": 4, "estimated_throughput": 108000},
        {"tp": 4, "ep": 2, "dp": 8, "estimated_throughput": 105000},
    ]
    print(f"      → Generated {len(candidates)} candidates")

    # Phase 4: Validation
    print("    Phase 4: Validation")
    best_candidate = candidates[0]
    actual_throughput = 119000  # Simulated
    target = 113000

    if actual_throughput >= target:
        print(f"      → Best: {actual_throughput} tok/s (target: {target}) ✓")
        decision = "DEPLOY"
    else:
        print(f"      → Best: {actual_throughput} tok/s (target: {target}) ✗")
        decision = "REFINE"

    # Phase 5: Decision
    print("    Phase 5: Decision")
    if decision == "DEPLOY":
        print(f"      → Deploy optimal config")
        improvement = (actual_throughput / baseline_metrics["throughput"] - 1) * 100
        print(f"      → Improvement: +{improvement:.1f}%")
    else:
        print(f"      → Refine constraints and iterate")

    print(f"\n  ✓ Simulated flow completed successfully")
    return True


def test_agents_md_registration():
    """Test 7: Check AGENTS.md registration"""
    print("\n[Test 7] AGENTS.md Registration")
    print("=" * 60)

    agents_md = SKILL_DIR.parents[2] / "AGENTS.md"

    if not agents_md.exists():
        print("  ✗ AGENTS.md not found")
        return False

    with open(agents_md) as f:
        content = f.read()

    # Check for moe-optimizer registration
    if "moe-optimizer" in content:
        print("  ✓ moe-optimizer registered in AGENTS.md")
    else:
        print("  ✗ moe-optimizer not found in AGENTS.md")
        return False

    # Check for moe-profiler registration
    if "moe-profiler" in content:
        print("  ✓ moe-profiler registered in AGENTS.md")
    else:
        print("  ✗ moe-profiler not found in AGENTS.md")
        return False

    print(f"\n  ✓ Both skills properly registered")
    return True


def main():
    print("=" * 60)
    print("MoE Optimizer Skill Validation")
    print("=" * 60)

    tests = [
        ("Structure Completeness", test_structure),
        ("SKILL.md Structure", test_skill_md_structure),
        ("Workflow Consistency", test_workflow_consistency),
        ("Reference Completeness", test_references_completeness),
        ("moe-profiler Integration", test_moe_profiler_integration),
        ("Simulated Flow", test_simulated_flow),
        ("AGENTS.md Registration", test_agents_md_registration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

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
