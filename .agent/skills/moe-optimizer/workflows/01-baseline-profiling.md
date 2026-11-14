# Phase 1: Baseline Profiling

**Goal:** Extract baseline MoE training facts to identify optimization opportunities.

## Prerequisites Check

Before running profiling:
- [ ] Training logs exist and contain MoE metrics
- [ ] Logs include at least 20 steps (preferably 50+)
- [ ] If available: profiler output, checkpoints

## Step 1: Invoke moe-profiler

Call the `moe-profiler` skill with baseline training logs:

```bash
python .agent/skills/moe-profiler/scripts/extract_moe_facts.py \
  --log-dir ${USER_INPUT.baseline_logs} \
  --profile-dir ${USER_INPUT.profiler_output} \
  --checkpoint-dir ${USER_INPUT.checkpoint} \
  --output-dir ./baseline_moe_facts
```

**Expected output:** 15 artifacts in `./baseline_moe_facts/`

## Step 2: Load and Parse moe_facts.json

```python
import json

with open('./baseline_moe_facts/moe_facts.json') as f:
    baseline_facts = json.load(f)

baseline_metrics = baseline_facts['summary']
```

**Key metrics to extract:**
- `steps_analyzed`
- `layers_analyzed`
- `experts_per_layer`
- `mean_load_imbalance_factor`
- `mean_alltoall_ratio`
- `max_straggler_slowdown` (if available)
- `routing_integrity`
- `gradient_flow`

## Step 3: Identify Problem Types

Based on extracted metrics, classify problems:

### Problem Type 1: Load Imbalance

**Condition:**
```python
load_imbalance_factor > 1.5
```

**Severity:**
- 1.5 - 1.8: Moderate
- 1.8 - 2.2: High
- > 2.2: Severe

**Symptoms:**
- Some experts receive far more tokens than others
- Check `expert_load_per_layer.csv` for hot/cold experts
- May correlate with stragglers

---

### Problem Type 2: Communication Bottleneck

**Condition:**
```python
alltoall_ratio > 0.3
```

**Severity:**
- 0.3 - 0.4: Moderate (30-40% of step time)
- 0.4 - 0.5: High
- > 0.5: Severe (communication-bound)

**Symptoms:**
- High fraction of time in All-to-All collectives
- Check `alltoall_timing.csv` for timing breakdown
- Often caused by large `ep_size`

---

### Problem Type 3: Stragglers

**Condition:**
```python
max_rank_time / avg_rank_time > 1.2
```

**Severity:**
- 1.2 - 1.3: Moderate (20-30% slowdown)
- 1.3 - 1.5: High
- > 1.5: Severe

**Symptoms:**
- Some ranks consistently slower
- Check `rank_timing_distribution.csv`
- Often caused by load imbalance + static expert mapping

---

### Problem Type 4: Correctness Issues

**Condition:**
```python
routing_integrity != "PASS" or gradient_flow != "PASS"
```

**Severity:** CRITICAL (must fix before optimization)

**Symptoms:**
- Token count mismatch
- Missing gradients for some experts
- Check `routing_integrity.csv` and `gradient_flow_check.csv`

## Step 4: Compute Baseline Throughput

Extract baseline performance:

```python
# From logs or user input
baseline_throughput = baseline_metrics.get('baseline_throughput', USER_INPUT.throughput)

# From step time if available
if 'mean_step_time_ms' in baseline_metrics:
    step_time_sec = baseline_metrics['mean_step_time_ms'] / 1000.0
    global_batch_size = USER_INPUT.global_batch_size
    seq_length = USER_INPUT.seq_length
    baseline_throughput = global_batch_size * seq_length / step_time_sec
```

## Step 5: Set Optimization Constraints

Based on problem diagnosis, set initial constraints for strategy generation:

```python
constraints = {
    "problems": [],
    "strategy_hints": {}
}

# Load imbalance → suggest capacity/bias
if load_imbalance_factor > 1.5:
    constraints["problems"].append("load_imbalance")
    constraints["strategy_hints"]["capacity_factor"] = 1.25  # Allow overflow
    constraints["strategy_hints"]["expert_bias"] = True      # Enable bias correction

# Communication bottleneck → reduce EP
if alltoall_ratio > 0.3:
    constraints["problems"].append("communication_bottleneck")
    max_ep_size = min(8, hardware.device_count // 4)
    constraints["strategy_hints"]["max_ep_size"] = max_ep_size
    constraints["strategy_hints"]["prefer_tp_over_ep"] = True

# Stragglers → need rebalancing
if straggler_factor > 1.2:
    constraints["problems"].append("stragglers")
    constraints["strategy_hints"]["dynamic_expert_mapping"] = True
```

## Step 6: Output Baseline Report

Generate a structured baseline report:

```json
{
  "phase": "baseline_profiling",
  "status": "complete",
  "baseline_metrics": {
    "throughput": 94000,
    "step_time_ms": 175.3,
    "memory_per_device_gb": 52.1,
    "load_imbalance_factor": 1.68,
    "alltoall_ratio": 0.32,
    "straggler_factor": 1.15
  },
  "problems_identified": [
    "load_imbalance (moderate)",
    "communication_bottleneck (moderate)"
  ],
  "constraints_for_next_phase": {
    "capacity_factor": 1.25,
    "expert_bias": true,
    "max_ep_size": 8,
    "prefer_tp_over_ep": true
  },
  "artifacts_location": "./baseline_moe_facts/"
}
```

## Step 7: Validation

Before proceeding to Phase 2:

- [ ] Baseline metrics successfully extracted
- [ ] At least one problem identified (otherwise no optimization needed)
- [ ] No critical correctness issues (must fix first if present)
- [ ] Constraints set for strategy generation

If validation fails:
- Check moe-profiler output for errors
- Ensure logs contain MoE-specific metrics
- Verify HP_MOE_PROFILING was enabled during training

---

**Next Phase:** [02-problem-diagnosis.md](02-problem-diagnosis.md)
