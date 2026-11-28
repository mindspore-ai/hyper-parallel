# Phase 4: Comparison Report

Compare actual validation metrics against PSA estimates and baseline configurations.

## 4.1 Input Sources

**Three comparison dimensions:**

1. **PSA Estimates** (from parallel-strategy-analyzer output)
2. **Actual Measurements** (from Phase 3 metrics extraction)
3. **Baseline** (reference config, e.g., pure DP or user's current setup)

```python
comparison_input = {
    "psa_estimates": {
        "memory_per_device_gb": 58.0,
        "throughput_tokens_per_sec": 125000,
        "mfu_percent": 45,
        "bubble_ratio": 0.053,
        "step_time_ms": 1100
    },
    "actual_metrics": {
        "memory_per_device_gb": 61.3,
        "throughput_tokens_per_sec": 118234,
        "mfu_percent": 42.1,
        "bubble_ratio": 0.061,
        "step_time_ms": 1203.5
    },
    "baseline": {
        "config": "dp=64, tp=1, pp=1",
        "throughput_tokens_per_sec": 80000,  # From user or prior run
        "memory_per_device_gb": 72.0,
        "status": "OOM at seq_len=4096"
    }
}
```

## 4.2 Accuracy Assessment

### 4.2.1 Compute Deltas

```python
def compute_delta(actual, estimate, metric_name):
    """Compute % difference: (actual - estimate) / estimate * 100"""
    if estimate == 0:
        return {"error": "Division by zero"}

    delta = (actual - estimate) / estimate * 100

    # Determine tolerance (metric-dependent)
    tolerances = {
        "memory": 20,      # ±20% acceptable
        "throughput": 15,  # ±15%
        "mfu": 10,         # ±10 percentage points (absolute)
        "bubble": 5,       # ±5 percentage points
        "step_time": 15    # ±15%
    }

    tolerance = tolerances.get(metric_name, 20)
    within_tolerance = abs(delta) <= tolerance

    assessment = "✓ PASS" if within_tolerance else "⚠ DEVIATION"

    return {
        "actual": actual,
        "estimate": estimate,
        "delta_percent": round(delta, 1),
        "within_tolerance": within_tolerance,
        "tolerance_threshold": tolerance,
        "assessment": assessment
    }

# Example:
# Memory: actual=61.3, estimate=58.0
# → delta = +5.7%, within ±20% → ✓ PASS
```

### 4.2.2 Generate Comparison Table

```python
def generate_comparison_table(psa, actual, baseline=None):
    table = []

    # Core metrics
    metrics = [
        ("Memory/device", "memory_per_device_gb", "GB"),
        ("Throughput", "throughput_tokens_per_sec", "tokens/s"),
        ("MFU", "mfu_percent", "%"),
        ("Step time", "step_time_ms", "ms"),
    ]

    # Add bubble only if PP > 1
    if psa.get("strategy", {}).get("pp", 1) > 1:
        metrics.append(("Bubble (PP)", "bubble_ratio", "%"))

    for display_name, key, unit in metrics:
        psa_val = psa.get(key, None)
        actual_val = actual.get(key, None)

        if psa_val is None or actual_val is None:
            continue

        # Format values
        if "tokens_per_sec" in key:
            psa_str = f"{psa_val/1000:.1f}K"
            actual_str = f"{actual_val/1000:.1f}K"
        elif "ratio" in key:
            psa_str = f"{psa_val*100:.1f}%"
            actual_str = f"{actual_val*100:.1f}%"
        else:
            psa_str = f"{psa_val:.1f}"
            actual_str = f"{actual_val:.1f}"

        delta = compute_delta(actual_val, psa_val, key.split("_")[0])

        table.append({
            "metric": display_name,
            "psa_estimate": f"{psa_str} {unit}",
            "actual": f"{actual_str} {unit}",
            "delta": f"{delta['delta_percent']:+.1f}%",
            "status": delta["assessment"]
        })

    return table
```

**Rendered output:**

```
┌──────────────────┬────────────┬───────────┬────────┬────────────┐
│ Metric           │ PSA Est.   │ Actual    │ Delta  │ Status     │
├──────────────────┼────────────┼───────────┼────────┼────────────┤
│ Memory/device    │ 58.0 GB    │ 61.3 GB   │ +5.7%  │ ✓ PASS     │
│ Throughput       │ 125.0K t/s │ 118.2K t/s│ -5.4%  │ ✓ PASS     │
│ MFU              │ 45.0 %     │ 42.1 %    │ -2.9 pt│ ✓ PASS     │
│ Step time        │ 1100 ms    │ 1203.5 ms │ +9.4%  │ ✓ PASS     │
│ Bubble (PP)      │ 5.3 %      │ 6.1 %     │ +0.8 pt│ ✓ PASS     │
└──────────────────┴────────────┴───────────┴────────┴────────────┘

Overall Assessment: ✓ VALIDATED (5/5 metrics within tolerance)
```

## 4.3 Baseline Comparison

If baseline provided, show improvement:

```python
def compare_to_baseline(actual, baseline):
    if not baseline:
        return None

    throughput_improvement = (
        (actual["throughput_tokens_per_sec"] - baseline["throughput_tokens_per_sec"])
        / baseline["throughput_tokens_per_sec"] * 100
    )

    memory_improvement = (
        (baseline["memory_per_device_gb"] - actual["memory_per_device_gb"])
        / baseline["memory_per_device_gb"] * 100
    )

    return {
        "baseline_config": baseline["config"],
        "throughput_improvement_percent": round(throughput_improvement, 1),
        "memory_reduction_percent": round(memory_improvement, 1),
        "baseline_status": baseline.get("status", "Unknown")
    }
```

**Rendered:**

```
Comparison to Baseline (dp=64, tp=1, pp=1):
  ✓ Throughput: +47.8% (80K → 118K tokens/s)
  ✓ Memory: -14.9% (72GB → 61.3GB, baseline had OOM)

Baseline status: OOM at seq_len=4096
New strategy: Fits comfortably with 19% headroom
```

## 4.4 Diagnostic Analysis

### 4.4.1 Memory Breakdown

If actual memory significantly exceeds estimate:

```python
def diagnose_memory_overage(actual_gb, estimate_gb, threshold=10):
    delta_pct = (actual_gb - estimate_gb) / estimate_gb * 100

    if delta_pct < threshold:
        return None  # Within acceptable range

    # Common causes
    diagnosis = []

    if delta_pct > 20:
        diagnosis.append({
            "cause": "Activation memory underestimated",
            "solution": "Enable activation checkpoint or reduce micro-batch size"
        })

    if delta_pct > 15:
        diagnosis.append({
            "cause": "FSDP overhead (parameter prefetch buffers)",
            "solution": "Normal for FSDP, upgrade to level3 if tight"
        })

    diagnosis.append({
        "cause": "Fragmentation or CUDA graph overhead",
        "solution": "Use torch.cuda.empty_cache() between steps (dev only)"
    })

    return {
        "overage_percent": round(delta_pct, 1),
        "likely_causes": diagnosis
    }
```

### 4.4.2 Throughput Gap

If actual throughput << estimate:

```python
def diagnose_throughput_gap(actual_tps, estimate_tps, comm_breakdown):
    gap_pct = (estimate_tps - actual_tps) / estimate_tps * 100

    if gap_pct < 10:
        return None

    diagnosis = []

    # Check if communication dominates
    comm_time_ms = comm_breakdown.get("total_comm_time_ms", 0)
    step_time_ms = comm_breakdown.get("step_time_ms", 1)
    comm_ratio = comm_time_ms / step_time_ms

    if comm_ratio > 0.3:  # >30% time in communication
        diagnosis.append({
            "cause": "Communication bottleneck",
            "details": f"Communication: {comm_ratio*100:.1f}% of step time",
            "solution": "Reduce TP/CP degree or check network bandwidth"
        })

    if gap_pct > 20:
        diagnosis.append({
            "cause": "Compute inefficiency",
            "solution": "Check for kernel launch overhead, enable compile_model"
        })

    return {
        "gap_percent": round(gap_pct, 1),
        "likely_causes": diagnosis
    }
```

## 4.5 Multi-Candidate Ranking

If validating Top-3 strategies:

```python
def rank_candidates(candidates):
    """
    Rank by:
    1. Passes validation (within tolerance)
    2. Highest actual throughput
    3. Memory headroom (prefer 70-80% utilization)
    """
    ranked = []

    for candidate in candidates:
        score = 0

        # Pass/fail gate
        if candidate.get("validation_status") == "PASS":
            score += 1000
        else:
            score = -1  # Rejected
            ranked.append({"candidate": candidate, "score": score, "rank": None})
            continue

        # Throughput (higher is better)
        score += candidate["actual"]["throughput_tokens_per_sec"] / 1000

        # Memory fitness (penalize if >90% or <60%)
        mem_util = candidate["actual"]["memory_per_device_gb"] / candidate["hardware"]["memory_per_device_gb"]
        if 0.70 <= mem_util <= 0.85:
            score += 50  # Optimal range
        elif mem_util > 0.90:
            score -= 100  # Too tight

        # Simplicity bonus (fewer parallel dimensions)
        strategy = candidate["strategy"]
        active_dims = sum([
            strategy["dp"] > 1,
            strategy["tp"] > 1,
            strategy["pp"] > 1,
            strategy["cp"] > 1,
            strategy["ep"] > 1
        ])
        score -= active_dims * 5

        ranked.append({"candidate": candidate, "score": score})

    # Sort by score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks
    for i, item in enumerate(ranked):
        if item["score"] >= 0:
            item["rank"] = i + 1

    return ranked
```

**Rendered:**

```
Multi-Candidate Validation Results:

┌────┬──────────────────┬──────────┬──────────┬────────────┐
│ #  │ Config           │ Tokens/s │ Memory   │ Status     │
├────┼──────────────────┼──────────┼──────────┼────────────┤
│ 1  │ dp=4,tp=8,pp=2   │ 118K     │ 61/64 GB │ ✓ Best     │
│ 2  │ dp=2,tp=8,pp=4   │ 95K      │ 45/64 GB │ ✓ Fallback │
│ 3  │ dp=8,tp=8,pp=1   │ —        │ OOM @s12 │ ✗ Rejected │
└────┴──────────────────┴──────────┴──────────┴────────────┘

Recommendation: Deploy candidate #1 (dp=4,tp=8,pp=2)
  - Highest throughput (118K tokens/s, 1.48x baseline)
  - Healthy memory utilization (95%)
  - All metrics validated within tolerance
```

## 4.6 Final Report Structure

**Complete validation report:**

```markdown
# Strategy Validation Report

**Generated:** 2026-10-16 14:35:22 UTC
**Model:** Qwen3-30B-A3B (cropped to 24 layers)
**Hardware:** 64× Ascend 910B (64GB each)
**Workload:** seq_len=4096, batch_size=128, 50 steps

---

## 1. Validated Strategy

**Configuration:** dp=4, tp=8, pp=2, FSDP=level2, activation_checkpoint=True

```python
mesh = init_device_mesh("npu", (4, 2, 8), mesh_dim_names=("dp", "pp", "tp"))
```

---

## 2. Performance Comparison

[Comparison table from 4.2.2]

**Overall Assessment:** ✓ VALIDATED (5/5 metrics within tolerance)

---

## 3. Baseline Improvement

[Baseline comparison from 4.3]

---

## 4. Detailed Metrics

**Throughput:**
- Mean: 118,234 tokens/s (std: 1,632 tokens/s, 1.4% variance)
- Samples/s: 28.9 (batch_size=1, seq_len=4096)

**Memory:**
- Peak: 61.3 GB per device (95% utilization of 64GB)
- Headroom: 2.7 GB (4.2%)
- Variance across ranks: 0.2 GB (stable)

**Compute Efficiency:**
- MFU: 42.1% (actual: 134.7 TFLOPS / peak: 320 TFLOPS)
- Step time: 1203.5 ms ± 12.4 ms

**Communication (from profiler):**
- Total: 142.3 ms per step (11.8% of step time)
- Breakdown:
  - TP AllReduce: 85.2 ms (96 ops)
  - FSDP AllGather: 32.1 ms (48 ops)
  - FSDP ReduceScatter: 25.0 ms (48 ops)

**Pipeline Bubble:**
- Ratio: 6.1% (idle: 73.5ms / compute: 1130ms)
- Schedule: 1F1B
- Recommendation: Acceptable, or use interleaved_1f1b for ~3% bubble

---

## 5. Diagnostics

[Any warnings from 4.4.1/4.4.2, or "No issues detected"]

---

## 6. Production Recommendation

✓ **APPROVED for production deployment**

- All metrics validated within tolerance
- 1.48x throughput improvement over baseline
- Stable performance (low variance across steps)
- Healthy memory headroom (4.2%)

**Suggested next steps:**
1. Run extended validation (500+ steps) to check for memory growth
2. Test with real data (synthetic data used in benchmark)
3. Monitor first production runs for OOM or communication anomalies

---

## 7. Reproducibility

**Logs:** `validation_run_20261016_143052.log`
**Config:** `config/generated_validation_config.yaml`
**Metrics:** `validation_metrics.json`
**Launch script:** `scripts/run_validation_20261016_143052.sh`
```

## 4.7 Exit Codes

```python
EXIT_CODES = {
    0: "VALIDATED - all metrics within tolerance, approved",
    1: "PARTIAL - some metrics exceeded tolerance, review required",
    2: "REJECTED - strategy failed (OOM/hang/timeout)",
    3: "ERROR - validation process error (log missing, parse failure)"
}
```

## 4.8 Output Artifacts

```
validation_report_20261016_143522.md        # Human-readable report
validation_comparison.json                  # Machine-readable results
validation_plots/                           # Optional: throughput/memory curves
  throughput_over_steps.png
  memory_timeline.png
```
