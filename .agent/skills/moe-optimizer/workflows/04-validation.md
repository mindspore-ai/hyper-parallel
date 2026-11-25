# Phase 4: Validation

**Goal:** Validate Top-3 candidates through actual training runs and correctness checks.

## Dual Validation Strategy

Each candidate undergoes two independent validations:
1. **Performance Validation** (strategy-validator)
2. **Correctness Validation** (routing-validator)

Both must pass for candidate to be acceptable.

---

## Part A: Performance Validation

### Step 1: Prepare Validation Runs

For each candidate, generate training script:

```python
def generate_validation_script(candidate, model_config, hardware):
    config = candidate["config"]

    script = f'''#!/bin/bash
# Validation run for Candidate {candidate["rank"]}

export WORLD_SIZE={hardware.device_count}
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

# MoE profiling
export HP_MOE_PROFILING=1
export HP_MOE_PROFILING_DIR=./validation_candidate_{candidate["rank"]}/moe_profile

# Device mesh config
export TP_SIZE={config["tp_size"]}
export EP_SIZE={config["ep_size"]}
export DP_SIZE={config["dp_size"]}

# MoE-specific
export MOE_CAPACITY_FACTOR={config.get("capacity_factor", 1.0)}
export MOE_EXPERT_BIAS={str(config.get("expert_bias", False)).lower()}

# Run validation (50 steps)
torchrun --nproc_per_node={hardware.device_count} \\
  train.py \\
  --model {model_config.model_path} \\
  --steps 50 \\
  --global_batch_size {model_config.global_batch_size} \\
  --seq_length {model_config.seq_length} \\
  --output_dir ./validation_candidate_{candidate["rank"]}
'''
    return script
```

### Step 2: Invoke strategy-validator

```yaml
skill: strategy-validator
inputs:
  model_path: ${model_config.model_path}
  config: ${candidate.config}
  validation_steps: 50
  output_dir: ./validation_candidate_${candidate.rank}
parallel: true  # Run all 3 candidates in parallel if resources allow
```

**Runtime:** ~5-10 minutes per candidate

### Step 3: Parse Validation Results

```python
def parse_validation_result(candidate_rank):
    result_file = f"./validation_candidate_{candidate_rank}/metrics.json"

    with open(result_file) as f:
        metrics = json.load(f)

    return {
        "throughput": metrics["throughput"]["tokens_per_sec"],
        "memory_per_device_gb": metrics["memory"]["max_allocated_gb"],
        "step_time_ms": metrics["throughput"]["mean_step_time_ms"],
        "step_time_std_ms": metrics["throughput"]["std_step_time_ms"],
        "stability": metrics["throughput"]["std"] / metrics["throughput"]["mean"],  # CV
        "oom": metrics.get("oom", False),
        "errors": metrics.get("errors", [])
    }
```

### Step 4: Check Performance Criteria

```python
def check_performance(actual, estimated, user_target):
    # Criterion 1: No OOM
    if actual["oom"]:
        return {"pass": False, "reason": "OOM"}

    # Criterion 2: No errors
    if actual["errors"]:
        return {"pass": False, "reason": f"Errors: {actual['errors']}"}

    # Criterion 3: Stable throughput (CV < 15%)
    if actual["stability"] > 0.15:
        return {"pass": False, "reason": f"Unstable: CV={actual['stability']:.2%}"}

    # Criterion 4: Memory within limits
    if actual["memory_per_device_gb"] > hardware.memory_per_device_gb * 0.95:
        return {"pass": False, "reason": "Memory too high"}

    # Criterion 5: Throughput meets target (with 5% tolerance)
    if actual["throughput"] < user_target * 0.95:
        return {
            "pass": False,
            "reason": f"Throughput {actual['throughput']} < target {user_target}"
        }

    # Criterion 6: Not too far from estimate (sanity check)
    deviation = abs(actual["throughput"] - estimated["throughput_adjusted"]) / estimated["throughput_adjusted"]
    if deviation > 0.3:  # 30% deviation
        return {
            "pass": False,
            "reason": f"Large deviation from estimate: {deviation:.1%}"
        }

    return {"pass": True, "reason": "All criteria met"}
```

---

## Part B: Correctness Validation

### Step 1: Extract Validation Artifacts

After performance validation run completes, extract MoE-specific artifacts:

```bash
# Extract MoE profiling data
python .agent/skills/moe-profiler/scripts/extract_moe_facts.py \
  --log-dir ./validation_candidate_${rank}/logs \
  --checkpoint-dir ./validation_candidate_${rank}/checkpoint \
  --output-dir ./validation_candidate_${rank}/moe_facts
```

### Step 2: Invoke routing-validator

```yaml
skill: routing-validator
inputs:
  checkpoint_dir: ./validation_candidate_${rank}/checkpoint
  log_dir: ./validation_candidate_${rank}/logs
  output_dir: ./validation_candidate_${rank}/correctness
```

### Step 3: Parse Correctness Results

```python
def parse_correctness_result(candidate_rank):
    result_file = f"./validation_candidate_{candidate_rank}/correctness/report.json"

    with open(result_file) as f:
        report = json.load(f)

    return {
        "routing_integrity": report["routing"]["status"],  # PASS/FAIL
        "routing_details": report["routing"].get("details", ""),
        "gradient_flow": report["gradients"]["status"],
        "gradient_details": report["gradients"].get("details", ""),
        "expert_coverage": report["coverage"]["status"],
        "dead_experts": report["coverage"].get("dead_experts", [])
    }
```

### Step 4: Check Correctness Criteria

```python
def check_correctness(correctness):
    # Criterion 1: Routing integrity (CRITICAL)
    if correctness["routing_integrity"] != "PASS":
        return {
            "pass": False,
            "reason": f"Routing integrity failed: {correctness['routing_details']}"
        }

    # Criterion 2: Gradient flow (CRITICAL)
    if correctness["gradient_flow"] != "PASS":
        return {
            "pass": False,
            "reason": f"Gradient flow failed: {correctness['gradient_details']}"
        }

    # Criterion 3: Expert coverage (WARNING, not blocking)
    if correctness["dead_experts"]:
        # Dead experts are concerning but not blocking
        return {
            "pass": True,
            "reason": "Passed with warnings",
            "warnings": f"Dead experts detected: {correctness['dead_experts']}"
        }

    return {"pass": True, "reason": "All correctness checks passed"}
```

---

## Step 5: Aggregate Validation Results

Combine performance and correctness results for each candidate:

```python
validation_results = []

for candidate in candidates:
    rank = candidate["rank"]

    # Run validations
    perf_actual = parse_validation_result(rank)
    perf_check = check_performance(perf_actual, candidate["estimated_performance"], user_target)

    correctness = parse_correctness_result(rank)
    correctness_check = check_correctness(correctness)

    # Overall status
    overall_pass = perf_check["pass"] and correctness_check["pass"]

    validation_results.append({
        "candidate_rank": rank,
        "config": candidate["config"],
        "performance": {
            "actual": perf_actual,
            "check": perf_check
        },
        "correctness": {
            "results": correctness,
            "check": correctness_check
        },
        "overall_pass": overall_pass,
        "failure_reasons": [
            perf_check.get("reason") if not perf_check["pass"] else None,
            correctness_check.get("reason") if not correctness_check["pass"] else None
        ]
    })

# Filter to only passing candidates
passing_candidates = [r for r in validation_results if r["overall_pass"]]
```

## Step 6: Select Best Candidate

If multiple candidates pass, select the best one:

```python
def select_best(passing_candidates):
    if not passing_candidates:
        return None

    # Primary: highest throughput
    # Secondary: lowest memory usage
    # Tertiary: best stability (lowest CV)

    def score(candidate):
        perf = candidate["performance"]["actual"]
        return (
            perf["throughput"],
            -perf["memory_per_device_gb"],  # Negative to minimize
            -perf["stability"]              # Negative to minimize
        )

    best = max(passing_candidates, key=score)
    return best
```

## Output

```json
{
  "phase": "validation",
  "status": "complete",
  "candidates_tested": 3,
  "candidates_passed": 2,
  "validation_results": [
    {
      "candidate_rank": 1,
      "config": {
        "tp_size": 2,
        "ep_size": 4,
        "dp_size": 8,
        "capacity_factor": 1.25,
        "expert_bias": true
      },
      "performance": {
        "actual": {
          "throughput": 119000,
          "memory_per_device_gb": 58.1,
          "step_time_ms": 172.5,
          "stability": 0.08,
          "oom": false
        },
        "check": {
          "pass": true,
          "reason": "All criteria met"
        }
      },
      "correctness": {
        "results": {
          "routing_integrity": "PASS",
          "gradient_flow": "PASS",
          "expert_coverage": "PASS"
        },
        "check": {
          "pass": true,
          "reason": "All correctness checks passed"
        }
      },
      "overall_pass": true
    },
    {
      "candidate_rank": 2,
      "config": {...},
      "overall_pass": true,
      "performance": {...}
    },
    {
      "candidate_rank": 3,
      "overall_pass": false,
      "failure_reasons": ["Throughput 104000 < target 113000"]
    }
  ],
  "best_candidate": {
    "rank": 1,
    "throughput": 119000,
    "improvement_vs_baseline": 1.266,
    "target_met": true
  },
  "validation_time_minutes": 23
}
```

---

## Decision Points

### Scenario A: At least one candidate passes

→ **Select best passing candidate**
→ **Skip to Phase 5 (Report)**

### Scenario B: No candidates pass, iteration < 3

→ **Analyze failure modes**
→ **Go to Phase 4B (Iterative Refinement)**

### Scenario C: No candidates pass, iteration == 3

→ **Deploy best-effort candidate (highest throughput even if < target)**
→ **Generate report with limitations**

---

**Next Phase (if needed):** [05-iterative-refinement.md](05-iterative-refinement.md)
