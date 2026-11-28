# Phase 2: Problem Diagnosis

**Goal:** Classify problems from baseline metrics and derive optimization strategies.

## Input from Phase 1

```json
{
  "baseline_metrics": {...},
  "problems_identified": [...],
  "constraints_for_next_phase": {...}
}
```

## Diagnosis Decision Tree

### Branch 1: Load Imbalance

**Trigger:** `load_imbalance_factor > 1.5`

#### Root Cause Analysis

Read `expert_load_per_layer.csv` and `token_distribution_summary.csv`:

**Pattern A: Heavy-Tail Distribution**
- 20% experts receive 2-3× average load
- High Gini coefficient (> 0.3)
- **Cause:** Routing learned to prefer certain experts
- **Solution:** Enable `expert_bias`, increase `capacity_factor`

**Pattern B: Bimodal Distribution**
- Two clusters: "active" vs "dormant" experts
- Some experts rarely activated
- **Cause:** Over-capacity model or poor initialization
- **Solution:** Reduce `num_experts`, or retrain with load-balance loss

**Pattern C: Layer-Stratified**
- Imbalance increases with layer depth
- Early layers balanced, late layers skewed
- **Cause:** Specialization in deeper layers
- **Solution:** Per-layer capacity adjustment (advanced)

#### Strategy Hints

```python
if load_imbalance_factor > 2.0:
    # Severe: aggressive correction
    strategy_hints = {
        "capacity_factor": 1.5,
        "expert_bias": True,
        "load_balance_loss_weight": 0.01
    }
elif load_imbalance_factor > 1.5:
    # Moderate: mild correction
    strategy_hints = {
        "capacity_factor": 1.25,
        "expert_bias": True
    }
```

---

### Branch 2: Communication Bottleneck

**Trigger:** `alltoall_ratio > 0.3`

#### Root Cause Analysis

Read `alltoall_timing.csv` and `communication_breakdown.csv`:

**Pattern A: Large EP Size**
- `ep_size >= 8`
- Each All-to-All touches many ranks
- **Cause:** Over-parallelized experts
- **Solution:** Reduce `ep_size`, increase expert replication

**Pattern B: Small Hidden Size**
- `hidden_size < 2048`
- Compute too fast, communication dominates
- **Cause:** Model architecture
- **Solution:** Increase TP to add more compute, or accept comm overhead

**Pattern C: Slow Interconnect**
- Effective bandwidth < 50% theoretical
- High latency per All-to-All
- **Cause:** Network topology or congestion
- **Solution:** Reduce `ep_size`, optimize placement

#### Strategy Hints

```python
current_ep_size = baseline_config.get("ep_size", 8)

if alltoall_ratio > 0.4:
    # Severe: halve EP size
    max_ep_size = max(2, current_ep_size // 2)
elif alltoall_ratio > 0.3:
    # Moderate: reduce by 25%
    max_ep_size = max(2, int(current_ep_size * 0.75))

strategy_hints = {
    "max_ep_size": max_ep_size,
    "prefer_tp_over_ep": True,
    "min_tp_size": 2  # Force some TP to balance
}
```

---

### Branch 3: Stragglers

**Trigger:** `straggler_factor > 1.2`

#### Root Cause Analysis

Read `rank_timing_distribution.csv` and `straggler_analysis.csv`:

**Pattern A: Load Imbalance → Stragglers**
- Straggler ranks hold hot experts
- Correlation between expert load and rank time
- **Cause:** Static expert-to-rank mapping
- **Solution:** Same as load imbalance (capacity, bias)

**Pattern B: Hardware Heterogeneity**
- Straggler ranks consistently slow, independent of load
- No correlation with expert assignment
- **Cause:** Degraded device or network link
- **Solution:** Exclude straggler ranks, or use dynamic rebalancing

**Pattern C: Memory Pressure**
- Straggler ranks have high memory usage
- Frequent cache evictions
- **Cause:** Uneven memory distribution
- **Solution:** FSDP adjustment, reduce batch size

#### Strategy Hints

```python
# Check correlation with load imbalance
if load_imbalance_factor > 1.5 and straggler_factor > 1.2:
    # Likely load-induced stragglers
    strategy_hints["address_load_imbalance_first"] = True
else:
    # Hardware or memory issue
    strategy_hints["check_hardware_health"] = True
    strategy_hints["consider_fsdp"] = True
```

---

### Branch 4: Multiple Problems (Common Case)

**Trigger:** 2+ problems identified

#### Priority Order

1. **Correctness issues** (CRITICAL — must fix first)
2. **Stragglers** (can mask other issues)
3. **Load imbalance** (often causes stragglers)
4. **Communication bottleneck** (last to address)

#### Combined Strategy

```python
combined_hints = {}

# Address in priority order
if correctness_fail:
    # Abort optimization, fix correctness first
    return {"action": "abort", "reason": "correctness_failure"}

if stragglers and load_imbalance:
    # Load imbalance likely causing stragglers
    combined_hints.update(load_imbalance_hints)
    combined_hints["expect_straggler_improvement"] = True

if load_imbalance and comm_bottleneck:
    # Tradeoff: reducing EP helps comm but may worsen load balance
    # Prefer capacity/bias over EP reduction
    combined_hints["capacity_factor"] = 1.5  # More aggressive
    combined_hints["max_ep_size"] = max(4, current_ep_size // 2)

if only_comm_bottleneck:
    # Pure communication issue, safe to reduce EP
    combined_hints.update(comm_hints)
```

---

## Step 1: Classify Problems

For each detected problem, determine:
- Severity (moderate / high / severe)
- Root cause pattern
- Recommended strategy hints

## Step 2: Generate Constraint Updates

Update constraints from Phase 1 with diagnosis results:

```json
{
  "constraints_updated": {
    "capacity_factor": 1.25,
    "expert_bias": true,
    "max_ep_size": 4,
    "min_tp_size": 2,
    "prefer_tp_over_ep": true,
    "enable_fsdp": false
  },
  "diagnosis": {
    "primary_problem": "communication_bottleneck",
    "secondary_problem": "load_imbalance",
    "root_cause_pattern": "large_ep_size + heavy_tail_load",
    "expected_improvement": "20-40%"
  }
}
```

## Step 3: Estimate Achievable Improvement

Based on problem severity, estimate realistic improvement range:

```python
def estimate_improvement(problems, severities):
    # Conservative estimates
    improvements = {
        "load_imbalance": {
            "moderate": 1.05,  # 5%
            "high": 1.15,      # 15%
            "severe": 1.25     # 25%
        },
        "communication_bottleneck": {
            "moderate": 1.10,  # 10%
            "high": 1.20,      # 20%
            "severe": 1.35     # 35%
        },
        "stragglers": {
            "moderate": 1.08,  # 8%
            "high": 1.18,      # 18%
            "severe": 1.30     # 30%
        }
    }

    # Compound improvements (multiplicative)
    total_improvement = 1.0
    for problem, severity in zip(problems, severities):
        total_improvement *= improvements[problem][severity]

    # Cap at 1.5x (50%) to be conservative
    return min(1.5, total_improvement)
```

## Step 4: Set Realistic Target

```python
estimated_improvement = estimate_improvement(...)
conservative_target = baseline_throughput * estimated_improvement * 0.9  # 90% of estimate

user_target = USER_INPUT.target_throughput
if user_target > baseline_throughput * 2.0:
    warn("User target (2x+) may be unrealistic given diagnosed problems")
    suggest_target = conservative_target
else:
    suggest_target = user_target
```

## Output

```json
{
  "phase": "problem_diagnosis",
  "status": "complete",
  "diagnosis_summary": {
    "primary_problem": "communication_bottleneck",
    "severity": "moderate",
    "root_cause": "ep_size=8 with alltoall_ratio=0.32",
    "secondary_problem": "load_imbalance",
    "severity": "moderate"
  },
  "strategy_hints": {
    "max_ep_size": 4,
    "capacity_factor": 1.25,
    "expert_bias": true,
    "prefer_tp_over_ep": true
  },
  "estimated_improvement": 1.26,
  "suggested_target": 118000,
  "user_target": 113000,
  "target_achievable": true
}
```

---

**Next Phase:** [03-strategy-generation.md](03-strategy-generation.md)
