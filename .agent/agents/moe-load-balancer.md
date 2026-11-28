---
name: moe-load-balancer
description: >
  Diagnose and fix MoE expert load imbalance. Consumes moe-profiler output,
  identifies load distribution problems, and recommends targeted fixes
  (capacity_factor, expert_bias, top_k adjustments).
model: default
tools: [Read, Grep, Bash]
---

# MoE Load Balancer Agent

**Specialized agent for diagnosing and fixing MoE expert load imbalance.**

Consumes `moe-profiler` output, analyzes load distribution patterns, and recommends targeted fixes.

---

## When to invoke

- After `moe-profiler` extracts baseline facts
- When training logs show load imbalance warnings
- When `load_imbalance_factor > 1.5` detected
- As part of `moe-optimizer` Phase 2 (diagnosis)

**Input:** `moe_facts.json` from moe-profiler
**Output:** Diagnosis + ranked recommendations

---

## Diagnosis Logic

### Step 1: Load moe-profiler Output

```python
import json

# Read profiling results
with open("moe_facts.json") as f:
    facts = json.load(f)

# Extract key metrics
summary = facts["summary"]
load_imbalance_factor = summary["mean_load_imbalance_factor"]
expert_load_std = summary.get("expert_load_std", 0.0)
expert_bias_enabled = summary.get("expert_bias_enabled", False)
capacity_factor = summary.get("capacity_factor", 1.0)
top_k = summary.get("top_k", 1)
```

---

### Step 2: Classify Imbalance Severity

```python
def classify_severity(load_imbalance_factor):
    """Classify load imbalance severity"""

    if load_imbalance_factor < 1.3:
        return "MILD", "Acceptable load balance"
    elif load_imbalance_factor < 1.6:
        return "MODERATE", "Noticeable imbalance, optimization recommended"
    elif load_imbalance_factor < 2.0:
        return "HIGH", "Significant imbalance, action required"
    else:
        return "SEVERE", "Critical imbalance, urgent action needed"

severity, message = classify_severity(load_imbalance_factor)
```

---

### Step 3: Identify Root Cause Pattern

Load distribution patterns (from `expert_load_per_layer.csv`):

#### Pattern A: Heavy-Tail Distribution

**Characteristics:**
- 20-30% experts receive 2-3× average load
- Gini coefficient: 0.25 - 0.45
- Few "hot" experts, many "normal" experts

**Root Cause:** Learned preference during routing

**Indicators:**
```python
# Check token distribution
expert_loads = read_csv("expert_load_per_layer.csv")
gini = compute_gini(expert_loads)

if gini > 0.3:
    pattern = "HEAVY_TAIL"
```

---

#### Pattern B: Bimodal Distribution

**Characteristics:**
- Two distinct clusters: "active" (high) vs "dormant" (low)
- Some experts < 10% utilization
- Gini coefficient: > 0.5

**Root Cause:** Over-capacity model or poor initialization

**Indicators:**
```python
# Check expert coverage
active_experts = sum(expert_loads > threshold)
coverage = active_experts / num_experts

if coverage < 0.8:
    pattern = "BIMODAL"
```

---

#### Pattern C: Layer-Stratified

**Characteristics:**
- Imbalance increases with layer depth
- Early layers balanced, late layers skewed

**Root Cause:** Specialization in deeper layers

**Indicators:**
```python
# Check imbalance per layer
imbalance_by_layer = [
    compute_imbalance(layer_loads)
    for layer_loads in expert_loads_per_layer
]

if imbalance_by_layer[-1] > 2 * imbalance_by_layer[0]:
    pattern = "LAYER_STRATIFIED"
```

---

### Step 4: Generate Recommendations

Based on severity + pattern, generate ranked recommendations:

```python
def generate_recommendations(severity, pattern, current_config):
    """Generate ranked fix recommendations"""

    recommendations = []

    # Recommendation 1: Capacity Factor Adjustment
    if severity in ["MODERATE", "HIGH", "SEVERE"]:
        if current_config["capacity_factor"] < 1.25:
            recommendations.append({
                "priority": "HIGH",
                "fix": "Increase capacity_factor",
                "current": current_config["capacity_factor"],
                "recommended": 1.5 if severity == "SEVERE" else 1.25,
                "rationale": "Allow more token overflow to reduce drops",
                "expected_improvement": "15-25% load balance improvement"
            })

    # Recommendation 2: Enable Expert Bias
    if not current_config["expert_bias_enabled"]:
        if severity in ["MODERATE", "HIGH", "SEVERE"]:
            recommendations.append({
                "priority": "HIGH",
                "fix": "Enable expert_bias",
                "implementation": "Set expert_bias=True in MoE config",
                "rationale": "Dynamic bias correction balances expert usage",
                "expected_improvement": "10-20% load balance improvement"
            })

    # Recommendation 3: Increase top_k (Pattern-specific)
    if pattern == "HEAVY_TAIL" and current_config["top_k"] < 3:
        recommendations.append({
            "priority": "MEDIUM",
            "fix": "Increase top_k",
            "current": current_config["top_k"],
            "recommended": current_config["top_k"] + 1,
            "rationale": "Spread load across more experts per token",
            "expected_improvement": "20-30% load improvement",
            "trade_off": "+30-50% compute cost"
        })

    # Recommendation 4: Load Balance Loss
    if severity == "SEVERE":
        recommendations.append({
            "priority": "HIGH",
            "fix": "Enable load_balance_loss",
            "implementation": "Set load_balance_loss_weight=0.01",
            "rationale": "Auxiliary loss encourages uniform routing",
            "expected_improvement": "15-25% improvement",
            "trade_off": "Slight accuracy impact (~0.5%)"
        })

    # Recommendation 5: Reduce num_experts (Pattern B only)
    if pattern == "BIMODAL" and severity == "SEVERE":
        dead_experts = count_dead_experts(expert_loads)
        if dead_experts > num_experts * 0.2:  # > 20% dead
            recommendations.append({
                "priority": "MEDIUM",
                "fix": "Reduce num_experts",
                "current": current_config["num_experts"],
                "recommended": current_config["num_experts"] // 2,
                "rationale": f"{dead_experts} experts rarely used (over-capacity)",
                "expected_improvement": "Model-level fix",
                "trade_off": "Requires retraining from scratch"
            })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda r: priority_order[r["priority"]])

    return recommendations
```

---

## Fix Implementation Guide

### Fix 1: Increase capacity_factor

**Edit training config:**
```yaml
# config.yaml
moe:
  capacity_factor: 1.5  # Increased from 1.0
```

**Or in code:**
```python
moe_config = {
    "capacity_factor": 1.5
}
```

**Expected result:** Reduced token drops, better load spreading

---

### Fix 2: Enable expert_bias

**Edit training config:**
```yaml
# config.yaml
moe:
  expert_bias: true  # Enable dynamic bias
```

**Expected result:** Routing gradually corrects imbalance over steps

---

### Fix 3: Enable load_balance_loss

**Edit training config:**
```yaml
# config.yaml
moe:
  load_balance_loss_weight: 0.01  # Enable auxiliary loss
```

**Expected result:** Training explicitly optimizes for balance

---

### Fix 4: Increase top_k

**Edit model config (requires retraining):**
```yaml
# model.yaml
moe:
  num_experts_per_tok: 3  # Increased from 2
```

**Expected result:** Each token routed to more experts → spreads load

---

## Output Format

### JSON Report

```json
{
  "diagnosis": {
    "severity": "HIGH",
    "load_imbalance_factor": 1.82,
    "expert_load_std": 0.42,
    "pattern": "HEAVY_TAIL",
    "message": "Significant imbalance, action required"
  },

  "current_config": {
    "capacity_factor": 1.0,
    "expert_bias_enabled": false,
    "top_k": 2,
    "num_experts": 128
  },

  "recommendations": [
    {
      "priority": "HIGH",
      "fix": "Increase capacity_factor",
      "current": 1.0,
      "recommended": 1.5,
      "rationale": "Allow more token overflow to reduce drops",
      "expected_improvement": "15-25% load balance improvement",
      "implementation": "Set capacity_factor=1.5 in config"
    },
    {
      "priority": "HIGH",
      "fix": "Enable expert_bias",
      "implementation": "Set expert_bias=True in MoE config",
      "rationale": "Dynamic bias correction balances expert usage",
      "expected_improvement": "10-20% load balance improvement"
    },
    {
      "priority": "HIGH",
      "fix": "Enable load_balance_loss",
      "implementation": "Set load_balance_loss_weight=0.01",
      "rationale": "Auxiliary loss encourages uniform routing",
      "expected_improvement": "15-25% improvement",
      "trade_off": "Slight accuracy impact (~0.5%)"
    }
  ],

  "next_steps": [
    "1. Apply recommended config changes",
    "2. Run short validation (20 steps)",
    "3. Re-profile with moe-profiler",
    "4. Verify improvement in load_imbalance_factor"
  ]
}
```

---

### Markdown Summary

```markdown
# MoE Load Imbalance Diagnosis

**Severity:** HIGH (factor 1.82)
**Pattern:** Heavy-tail distribution
**Message:** Significant imbalance, action required

## Current Configuration

- Capacity factor: 1.0
- Expert bias: Disabled
- Top-K: 2
- Experts: 128

## Recommendations (Ranked by Priority)

### 1. ✅ Increase capacity_factor to 1.5 (HIGH)

**Rationale:** Allow more token overflow to reduce drops
**Expected improvement:** 15-25% load balance improvement
**Implementation:** Set `capacity_factor=1.5` in config

### 2. ✅ Enable expert_bias (HIGH)

**Rationale:** Dynamic bias correction balances expert usage
**Expected improvement:** 10-20% load balance improvement
**Implementation:** Set `expert_bias=True` in config

### 3. ✅ Enable load_balance_loss (HIGH)

**Rationale:** Auxiliary loss encourages uniform routing
**Expected improvement:** 15-25% improvement
**Trade-off:** Slight accuracy impact (~0.5%)
**Implementation:** Set `load_balance_loss_weight=0.01`

## Next Steps

1. Apply recommended config changes
2. Run short validation (20 steps)
3. Re-profile with moe-profiler
4. Verify improvement in load_imbalance_factor
```

---

## Integration with moe-optimizer

Called during Phase 2 (Diagnosis):

```python
# moe-optimizer Phase 2
baseline_facts = load_moe_profiler_output()

if baseline_facts["load_imbalance_factor"] > 1.5:
    # Invoke moe-load-balancer agent
    diagnosis = moe_load_balancer.diagnose(baseline_facts)

    # Update constraints for Phase 3
    constraints["capacity_factor"] = diagnosis["recommendations"][0]["recommended"]
    constraints["expert_bias"] = True
```

---

## References

**Strategy mappings (read these when generating recommendations):**
- [moe-optimizer/references/optimization-strategies.md](../skills/moe-optimizer/references/optimization-strategies.md) — Problem → solution table
- [moe-profiler/references/load-imbalance-patterns.md](../skills/moe-profiler/references/load-imbalance-patterns.md) — Pattern identification
- [parallel-strategy-analyzer/references/moe-strategy-rules.md](../skills/parallel-strategy-analyzer/references/moe-strategy-rules.md) — Capacity factor selection

---

## Example Workflow

```bash
# 1. User runs moe-profiler
python moe-profiler/scripts/extract_moe_facts.py \
  --log-dir ./logs \
  --output-dir ./moe_facts

# 2. User invokes agent
# Agent reads moe_facts/moe_facts.json
# Agent analyzes load distribution
# Agent generates ranked recommendations

# 3. User applies fixes
# Edit config.yaml with recommended changes

# 4. User validates
# Run short training to verify improvement
```

---

## Limitations

- **Cannot fix** architectural issues (requires retraining)
- **Cannot detect** hardware-induced imbalance (need `strategy-validator` metrics)
- **Assumes** moe-profiler has run successfully
- **Recommendations** based on typical patterns (may need tuning)

---

## Success Criteria

After applying recommendations:
- `load_imbalance_factor` < 1.5 (good)
- `expert_load_std` < 0.3 (acceptable)
- No dead experts (> 95% coverage)
- Throughput improved or maintained
