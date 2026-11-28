# Phase 5: Iterative Refinement

**Goal:** Analyze validation failures and refine constraints for next iteration.

**Trigger:** No candidates passed validation AND iteration < max_iterations

---

## Step 1: Analyze Failure Modes

Categorize failures from Phase 4 validation:

```python
def categorize_failures(validation_results):
    failure_types = {
        "oom": [],
        "throughput_below_target": [],
        "unstable": [],
        "correctness_routing": [],
        "correctness_gradient": [],
        "other": []
    }

    for result in validation_results:
        if not result["overall_pass"]:
            for reason in result["failure_reasons"]:
                if reason and "OOM" in reason:
                    failure_types["oom"].append(result)
                elif "Throughput" in reason and "< target" in reason:
                    failure_types["throughput_below_target"].append(result)
                elif "Unstable" in reason:
                    failure_types["unstable"].append(result)
                elif "Routing integrity" in reason:
                    failure_types["correctness_routing"].append(result)
                elif "Gradient flow" in reason:
                    failure_types["correctness_gradient"].append(result)
                else:
                    failure_types["other"].append(result)

    return failure_types
```

---

## Failure Mode 1: OOM (Out of Memory)

**Symptoms:**
- One or more candidates ran out of memory
- `oom: true` in validation results

**Root Causes:**
- TP/EP too small (not enough sharding)
- Batch size too large
- Capacity factor too high (more active experts)

**Refinement Actions:**

```python
def refine_for_oom(current_constraints, oom_candidates):
    refined = current_constraints.copy()

    # Action 1: Increase sharding
    if refined.get("min_tp_size", 1) < 4:
        refined["min_tp_size"] = 4
        refined["reason"] = "Increase TP to reduce memory per device"

    # Action 2: Enable FSDP if not already
    if not refined.get("enable_fsdp", False):
        refined["enable_fsdp"] = True
        refined["reason"] = "Enable FSDP to shard optimizer states"

    # Action 3: Reduce capacity factor
    if refined.get("capacity_factor", 1.0) > 1.0:
        refined["capacity_factor"] = max(1.0, refined["capacity_factor"] - 0.25)
        refined["reason"] = "Reduce capacity to limit active experts"

    # Action 4: Enable activation checkpointing (last resort)
    if refined.get("enable_fsdp") and refined.get("min_tp_size") >= 4:
        refined["enable_activation_checkpoint"] = True
        refined["checkpoint_every_n_layers"] = 4
        refined["reason"] = "Enable activation checkpoint to trade compute for memory"

    return refined
```

---

## Failure Mode 2: Throughput Below Target

**Symptoms:**
- Candidates run successfully but throughput < target
- Large gap between actual and target (> 10%)

**Root Causes:**
- Communication overhead still too high
- Compute under-utilized
- Sub-optimal parallelism balance

**Refinement Actions:**

```python
def refine_for_low_throughput(current_constraints, low_throughput_candidates):
    refined = current_constraints.copy()

    # Analyze why throughput is low
    avg_alltoall_ratio = np.mean([
        c["performance"]["actual"].get("alltoall_ratio", 0.3)
        for c in low_throughput_candidates
    ])

    # If still communication-bound
    if avg_alltoall_ratio > 0.25:
        # Further reduce EP
        current_max_ep = refined.get("max_ep_size", 8)
        refined["max_ep_size"] = max(2, current_max_ep // 2)
        refined["min_tp_size"] = max(2, refined.get("min_tp_size", 1))
        refined["reason"] = "Further reduce EP to lower communication"

    # If compute-bound but inefficient
    else:
        # Try different TP/EP balance
        refined["explore_alternative_balance"] = True
        refined["reason"] = "Explore different TP/EP configurations"

    # Lower target if unrealistic
    gap_ratio = user_target / baseline_throughput
    if gap_ratio > 1.5:  # Expecting > 50% improvement
        refined["realistic_target"] = baseline_throughput * 1.3
        refined["reason"] = "Original target may be unrealistic"

    return refined
```

---

## Failure Mode 3: Unstable Performance

**Symptoms:**
- High variance in step time (CV > 15%)
- Throughput fluctuates across steps

**Root Causes:**
- Load imbalance causing periodic stragglers
- Memory allocation jitter
- Network congestion

**Refinement Actions:**

```python
def refine_for_instability(current_constraints, unstable_candidates):
    refined = current_constraints.copy()

    # Action 1: More aggressive load balancing
    refined["capacity_factor"] = max(1.5, refined.get("capacity_factor", 1.25) + 0.25)
    refined["expert_bias"] = True
    refined["load_balance_loss_weight"] = 0.01
    refined["reason"] = "Aggressive load balancing to reduce variance"

    # Action 2: Reduce batch size variance
    refined["use_fixed_batch_size"] = True
    refined["reason"] = "Fix batch size to reduce allocation jitter"

    return refined
```

---

## Failure Mode 4: Correctness Issues

**Symptoms:**
- Routing integrity FAIL
- Gradient flow FAIL

**Root Causes:**
- Implementation bugs (critical)
- Incorrect configuration
- Numerical instability

**Refinement Actions:**

```python
def refine_for_correctness(current_constraints, correctness_failures):
    # Correctness failures are CRITICAL
    # Should not proceed with optimization until fixed

    refined = {
        "action": "ABORT",
        "reason": "Correctness failures detected",
        "details": [c["failure_reasons"] for c in correctness_failures],
        "recommendation": "Fix routing implementation before optimization"
    }

    # However, some issues may be config-related
    if any("capacity_factor" in str(c) for c in correctness_failures):
        # Token overflow due to capacity
        refined = current_constraints.copy()
        refined["capacity_factor"] = min(2.0, refined.get("capacity_factor", 1.0) + 0.5)
        refined["reason"] = "Increase capacity to prevent token drops"
        refined["action"] = "RETRY"

    return refined
```

---

## Step 2: Apply Refinements

Combine refinements from all failure modes:

```python
def apply_refinements(current_constraints, failure_types, iteration):
    refined = current_constraints.copy()
    actions_taken = []

    # Priority 1: Correctness (abort if unfixable)
    if failure_types["correctness_routing"] or failure_types["correctness_gradient"]:
        correctness_refinement = refine_for_correctness(
            refined,
            failure_types["correctness_routing"] + failure_types["correctness_gradient"]
        )
        if correctness_refinement.get("action") == "ABORT":
            return correctness_refinement
        refined.update(correctness_refinement)
        actions_taken.append("correctness_fix")

    # Priority 2: OOM (must fix to proceed)
    if failure_types["oom"]:
        oom_refinement = refine_for_oom(refined, failure_types["oom"])
        refined.update(oom_refinement)
        actions_taken.append("oom_fix")

    # Priority 3: Instability
    if failure_types["unstable"]:
        stability_refinement = refine_for_instability(refined, failure_types["unstable"])
        refined.update(stability_refinement)
        actions_taken.append("stability_fix")

    # Priority 4: Low throughput (if no other issues)
    if failure_types["throughput_below_target"] and not actions_taken:
        throughput_refinement = refine_for_low_throughput(refined, failure_types["throughput_below_target"])
        refined.update(throughput_refinement)
        actions_taken.append("throughput_optimization")

    refined["iteration"] = iteration + 1
    refined["actions_taken"] = actions_taken

    return refined
```

---

## Step 3: Check Convergence

Before returning to Phase 2, check if refinement is making progress:

```python
def check_convergence(iteration_history):
    # If last 2 iterations produced same constraints → stuck
    if len(iteration_history) >= 2:
        last_two = iteration_history[-2:]
        if last_two[0]["constraints"] == last_two[1]["constraints"]:
            return {
                "converged": True,
                "reason": "Constraints not changing, optimization stuck",
                "action": "DEPLOY_BEST_EFFORT"
            }

    # If throughput is improving but slowly
    if len(iteration_history) >= 3:
        throughputs = [h.get("best_throughput", 0) for h in iteration_history]
        improvements = [
            (throughputs[i+1] - throughputs[i]) / throughputs[i]
            for i in range(len(throughputs) - 1)
        ]

        if all(imp < 0.05 for imp in improvements[-2:]):  # < 5% improvement
            return {
                "converged": True,
                "reason": "Diminishing returns, optimization converging",
                "action": "DEPLOY_BEST_CURRENT"
            }

    return {"converged": False}
```

---

## Step 4: Decide Next Action

```python
def decide_next_action(refined_constraints, iteration, max_iterations):
    # Check iteration limit
    if iteration >= max_iterations:
        return {
            "action": "DEPLOY_BEST_EFFORT",
            "reason": f"Max iterations ({max_iterations}) reached"
        }

    # Check for abort conditions
    if refined_constraints.get("action") == "ABORT":
        return {
            "action": "ABORT",
            "reason": refined_constraints["reason"]
        }

    # Check convergence
    convergence = check_convergence(iteration_history)
    if convergence["converged"]:
        return {
            "action": convergence["action"],
            "reason": convergence["reason"]
        }

    # Otherwise, continue to next iteration
    return {
        "action": "CONTINUE",
        "reason": f"Retry with refined constraints (iteration {iteration + 1})"
    }
```

---

## Output

```json
{
  "phase": "iterative_refinement",
  "status": "complete",
  "current_iteration": 1,
  "failure_analysis": {
    "failure_types": {
      "oom": 0,
      "throughput_below_target": 2,
      "unstable": 0,
      "correctness": 0
    },
    "primary_issue": "throughput_below_target",
    "root_cause": "Communication overhead still high (alltoall_ratio=0.28)"
  },
  "refinements_applied": {
    "max_ep_size": "8 → 4 (halved)",
    "min_tp_size": "1 → 2 (doubled)",
    "reason": "Further reduce EP to lower communication"
  },
  "next_action": {
    "action": "CONTINUE",
    "reason": "Retry with refined constraints (iteration 2)",
    "go_to_phase": "03-strategy-generation"
  },
  "refined_constraints": {
    "max_ep_size": 4,
    "min_tp_size": 2,
    "capacity_factor": 1.25,
    "expert_bias": true,
    "prefer_tp_over_ep": true,
    "iteration": 2
  }
}
```

---

## Loop Back

**Return to Phase 2** (Problem Diagnosis) with refined constraints

**OR**

**Proceed to Phase 5** (Report) if:
- Converged (deploy best effort)
- Max iterations reached
- Abort condition met

---

**Completion:** Return to [SKILL.md](../SKILL.md) for final report generation.
