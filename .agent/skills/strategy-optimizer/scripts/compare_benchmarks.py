#!/usr/bin/env python3
"""
Compare two benchmark results.

Usage:
    python compare_benchmarks.py \
        --baseline metrics1.json \
        --candidate metrics2.json
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline metrics JSON")
    parser.add_argument("--candidate", type=str, required=True, help="Candidate metrics JSON")
    parser.add_argument("--output", type=str, help="Output comparison to file")
    return parser.parse_args()


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_metrics(baseline: dict, candidate: dict) -> dict:
    """Compare two sets of metrics."""
    comparison = {}

    # Throughput
    throughput_delta = candidate["throughput_tokens_per_sec"] - baseline["throughput_tokens_per_sec"]
    throughput_ratio = candidate["throughput_tokens_per_sec"] / baseline["throughput_tokens_per_sec"]

    comparison["throughput"] = {
        "baseline": baseline["throughput_tokens_per_sec"],
        "candidate": candidate["throughput_tokens_per_sec"],
        "delta": throughput_delta,
        "ratio": throughput_ratio,
        "improvement_pct": (throughput_ratio - 1) * 100,
    }

    # Step time
    step_time_delta = candidate["step_time_ms"] - baseline["step_time_ms"]
    step_time_ratio = candidate["step_time_ms"] / baseline["step_time_ms"]

    comparison["step_time"] = {
        "baseline": baseline["step_time_ms"],
        "candidate": candidate["step_time_ms"],
        "delta": step_time_delta,
        "ratio": step_time_ratio,
        "improvement_pct": (1 - step_time_ratio) * 100,  # Lower is better
    }

    # Memory
    memory_delta = candidate["memory_peak_gb"] - baseline["memory_peak_gb"]
    memory_ratio = candidate["memory_peak_gb"] / baseline["memory_peak_gb"]

    comparison["memory"] = {
        "baseline": baseline["memory_peak_gb"],
        "candidate": candidate["memory_peak_gb"],
        "delta": memory_delta,
        "ratio": memory_ratio,
        "change_pct": (memory_ratio - 1) * 100,
    }

    return comparison


def print_comparison(comparison: dict):
    """Print comparison results."""
    print("="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    # Throughput
    t = comparison["throughput"]
    print(f"\n📊 Throughput:")
    print(f"  Baseline:  {t['baseline']:.1f} tokens/s")
    print(f"  Candidate: {t['candidate']:.1f} tokens/s")
    print(f"  Delta:     {t['delta']:+.1f} tokens/s ({t['improvement_pct']:+.1f}%)")

    if t['improvement_pct'] > 5:
        print(f"  ✅ Candidate is SIGNIFICANTLY FASTER")
    elif t['improvement_pct'] > 0:
        print(f"  ✓ Candidate is slightly faster")
    elif t['improvement_pct'] > -5:
        print(f"  ≈ Candidate is comparable")
    else:
        print(f"  ✗ Candidate is SLOWER")

    # Step Time
    st = comparison["step_time"]
    print(f"\n⏱️  Step Time:")
    print(f"  Baseline:  {st['baseline']:.1f} ms")
    print(f"  Candidate: {st['candidate']:.1f} ms")
    print(f"  Delta:     {st['delta']:+.1f} ms ({st['improvement_pct']:+.1f}%)")

    # Memory
    m = comparison["memory"]
    print(f"\n💾 Memory:")
    print(f"  Baseline:  {m['baseline']:.2f} GB")
    print(f"  Candidate: {m['candidate']:.2f} GB")
    print(f"  Delta:     {m['delta']:+.2f} GB ({m['change_pct']:+.1f}%)")

    if abs(m['change_pct']) < 5:
        print(f"  ≈ Memory usage is comparable")
    elif m['change_pct'] > 0:
        print(f"  ⚠️  Candidate uses MORE memory")
    else:
        print(f"  ✓ Candidate uses LESS memory")

    # Overall Decision
    print(f"\n{'='*80}")
    print("DECISION")
    print(f"{'='*80}")

    if t['improvement_pct'] > 5 and abs(m['change_pct']) < 20:
        print("\n✅ RECOMMEND: Deploy candidate (faster, acceptable memory)")
    elif t['improvement_pct'] > 2 and m['change_pct'] < 10:
        print("\n✓ CONSIDER: Candidate is better (slight improvement)")
    elif t['improvement_pct'] < -5:
        print("\n✗ REJECT: Candidate is significantly slower")
    elif m['change_pct'] > 30:
        print("\n⚠️  CAUTION: Candidate uses significantly more memory")
    else:
        print("\n≈ NEUTRAL: Performance is comparable, choose based on other factors")


def main():
    args = parse_args()

    # Load metrics
    baseline = load_metrics(args.baseline)
    candidate = load_metrics(args.candidate)

    # Compare
    comparison = compare_metrics(baseline, candidate)

    # Print
    print_comparison(comparison)

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Comparison saved: {args.output}")


if __name__ == "__main__":
    main()
