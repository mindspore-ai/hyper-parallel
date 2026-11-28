# Strategy Optimizer - Orchestration Workflow

## Overview

This workflow orchestrates PSA (theory) and Validator (practice) in a closed loop to find optimal parallelism strategy.

## Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Iteration Loop (Max 3)                        │
└─────────────────────────────────────────────────────────────────┘

Iteration N:
  ↓
Step 1: Call PSA (parallel-strategy-analyzer)
  Input: model + hardware + constraints_N
  Output: Top-3 candidates (sorted by est. throughput)
  ↓
Step 2: Call Validator for each candidate
  For each of Top-3:
    - Generate config
    - Run 50-step benchmark
    - Extract metrics
  Output: Validation results (SUCCESS/FAILED)
  ↓
Step 3: Decision
  ┌────────────────────────────────────────┐
  │ Any candidate meets target?            │
  │   YES → Deploy winner ✓                │
  │   NO  → Analyze failures               │
  └────────────────────────────────────────┘
  ↓
Step 4: Refinement (if NO and iteration < 3)
  - Identify failure patterns (OOM, FSDP, Comm, Bubble)
  - Update constraints
  - Go to Iteration N+1
  ↓
Final: Deploy best performer
```

## Step-by-Step Implementation

### Initialization

```python
def initialize_optimizer(model, hardware, objective):
    """Setup initial state."""

    state = {
        "iteration": 0,
        "max_iterations": 3,
        "best_so_far": None,
        "history": [],
        "constraints": {
            # Initial constraints (permissive)
            "tp_range": [1, 2, 4, 8],
            "cp_range": [1, 2, 4],
            "ep_range": [2, 4, 8],  # MoE: EP >= 2
            "pp_range": [1, 2, 4],
            "dp_min": 2,  # Avoid FSDP conflicts
            "fsdp_level": 2,
            "activation_checkpoint": False,
            "pp_schedule": "1F1B",
        },
        "target": objective.target_throughput,
    }

    return state
```

### Step 1: PSA Analysis

```python
def call_psa(model, hardware, constraints, iteration):
    """Invoke parallel-strategy-analyzer skill."""

    print(f"\n{'='*70}")
    print(f"Iteration {iteration + 1}: PSA Analysis")
    print(f"{'='*70}")

    # Prepare PSA input
    psa_input = {
        "model": {
            "name": model.name,
            "num_layers": model.num_layers,
            "hidden_size": model.hidden_size,
            "seq_len": model.seq_len,
            "is_moe": model.is_moe,
            "num_experts": model.num_experts if model.is_moe else 0,
        },
        "hardware": {
            "device_type": hardware.device_type,
            "device_count": hardware.device_count,
            "memory_per_device": hardware.memory_per_device_gb,
            "interconnect": hardware.interconnect,
        },
        "constraints": constraints,
    }

    # Call PSA skill
    psa_result = Skill(
        skill="parallel-strategy-analyzer",
        args=json.dumps(psa_input)
    )

    # Parse PSA output
    candidates = parse_psa_output(psa_result)

    print(f"\nPSA Top-3 Candidates:")
    for i, cand in enumerate(candidates[:3], 1):
        print(f"  {i}. TP={cand.tp}, CP={cand.cp}, EP={cand.ep}, PP={cand.pp}, DP={cand.dp}")
        print(f"     Est. throughput: {cand.est_throughput} tokens/s")
        print(f"     Est. memory: {cand.est_memory} GB")

    return candidates[:3]


def parse_psa_output(psa_result):
    """Parse PSA output into structured candidates."""

    # Extract from PSA output (JSON or text)
    # Expected format: list of strategies with estimates
    candidates = []

    for strategy in psa_result.strategies:
        candidates.append({
            "tp": strategy.tp_size,
            "cp": strategy.cp_size,
            "ep": strategy.ep_size,
            "pp": strategy.pp_size,
            "dp": strategy.dp_size,
            "est_throughput": strategy.estimated_throughput,
            "est_memory": strategy.estimated_memory,
            "est_step_time": strategy.estimated_step_time,
        })

    return candidates
```

### Step 2: Validation

```python
def validate_candidates(candidates, model, hardware):
    """Run strategy-validator on each candidate."""

    print(f"\n{'='*70}")
    print(f"Validation Phase")
    print(f"{'='*70}")

    results = []

    for i, candidate in enumerate(candidates, 1):
        print(f"\n--- Validating Candidate {i} ---")
        print(f"Config: TP={candidate.tp}, CP={candidate.cp}, EP={candidate.ep}, "
              f"PP={candidate.pp}, DP={candidate.dp}")

        # Generate config file
        config_path = generate_validation_config(candidate, model, f"optimizer_iter{iteration}_cand{i}")

        # Generate launch script
        script_path = generate_launch_script(config_path, f"optimizer_iter{iteration}_cand{i}")

        print(f"  Config: {config_path}")
        print(f"  Script: {script_path}")
        print(f"  Status: Ready for execution on target node")

        # Store for batch execution
        results.append({
            "candidate_id": f"iter{iteration}_cand{i}",
            "config": candidate,
            "config_path": config_path,
            "script_path": script_path,
            "status": "PENDING",
        })

    print(f"\n{'='*70}")
    print(f"All validation configs generated")
    print(f"{'='*70}")
    print(f"\n📋 Manual Execution Required:")
    print(f"  1. SSH to the target node")
    print(f"  2. Run each script:")
    for r in results:
        print(f"     bash {r['script_path']}")
    print(f"  3. After completion, continue this workflow")

    # Wait for user to run validations
    user_input = input("\n✋ Have all validations completed? (yes/no): ")
    if user_input.lower() != "yes":
        print("⏸️  Paused. Run validations and resume.")
        return None

    # Extract metrics from logs
    print(f"\n📊 Extracting metrics...")
    for result in results:
        metrics = extract_validation_metrics(result["candidate_id"])
        if metrics:
            result["status"] = "SUCCESS"
            result["throughput"] = metrics["throughput"]
            result["memory"] = metrics["memory"]
            result["step_time"] = metrics["step_time"]
            result["delta_vs_psa"] = {
                "throughput": (metrics["throughput"] - result["config"]["est_throughput"]) / result["config"]["est_throughput"],
                "memory": metrics["memory"] - result["config"]["est_memory"],
            }
        else:
            result["status"] = "FAILED"
            result["error"] = detect_error_type(result["candidate_id"])

    # Print results
    print(f"\n{'='*70}")
    print(f"Validation Results")
    print(f"{'='*70}")
    for r in results:
        if r["status"] == "SUCCESS":
            print(f"✓ {r['candidate_id']}: {r['throughput']:.0f} tokens/s "
                  f"(memory: {r['memory']:.1f} GB)")
        else:
            print(f"✗ {r['candidate_id']}: {r['error']}")

    return results


def extract_validation_metrics(candidate_id):
    """Extract metrics from validation log."""
    log_path = f"validation_experiment/{candidate_id}/validation_run.log"

    if not Path(log_path).exists():
        return None

    # Use strategy-validator's extraction logic
    pattern = re.compile(r'tokens/s=([\d.]+),?\s+step_time=([\d.]+)s.*memory/device_max_allocated_gb=([\d.]+)')

    tokens_per_sec = []
    step_times_ms = []
    memory_gb = []

    with open(log_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                tps, step_time, mem = match.groups()
                tokens_per_sec.append(float(tps))
                step_times_ms.append(float(step_time) * 1000)
                memory_gb.append(float(mem))

    if not tokens_per_sec:
        return None

    # Use last 30 steps
    tokens_per_sec = tokens_per_sec[-30:]
    step_times_ms = step_times_ms[-30:]

    return {
        "throughput": statistics.mean(tokens_per_sec),
        "step_time": statistics.mean(step_times_ms),
        "memory": max(memory_gb),
    }


def detect_error_type(candidate_id):
    """Detect error type from log."""
    log_path = f"validation_experiment/{candidate_id}/validation_run.log"

    with open(log_path) as f:
        log_content = f.read()

    if "out of memory" in log_content.lower():
        return "OOM"
    elif "A view was created in no_grad mode" in log_content:
        return "FSDP_CONFLICT"
    elif "fused MoE expert weights incompatible" in log_content:
        return "MOE_INCOMPATIBLE"
    elif "timeout" in log_content.lower() or "hang" in log_content.lower():
        return "HANG"
    else:
        return "UNKNOWN"
```

### Step 3: Decision

```python
def make_decision(validation_results, target_throughput, iteration, max_iterations):
    """Decide: deploy, iterate, or give up."""

    print(f"\n{'='*70}")
    print(f"Decision Phase")
    print(f"{'='*70}")

    # Filter successful results
    successful = [r for r in validation_results if r["status"] == "SUCCESS"]

    if not successful:
        print(f"❌ All candidates failed")
        if iteration < max_iterations - 1:
            print(f"➡️  Will refine and retry (iteration {iteration + 2}/{max_iterations})")
            return "ITERATE", None
        else:
            print(f"❌ Max iterations reached, no successful candidate")
            return "FAILURE", None

    # Find best performer
    best = max(successful, key=lambda x: x["throughput"])

    print(f"\n🏆 Best Candidate: {best['candidate_id']}")
    print(f"   Config: TP={best['config']['tp']}, CP={best['config']['cp']}, "
          f"EP={best['config']['ep']}, PP={best['config']['pp']}, DP={best['config']['dp']}")
    print(f"   Throughput: {best['throughput']:.0f} tokens/s")
    print(f"   Memory: {best['memory']:.1f} GB")

    # Check if meets target
    if best["throughput"] >= target_throughput:
        print(f"✅ Meets target ({best['throughput']:.0f} >= {target_throughput:.0f})")
        print(f"➡️  Decision: DEPLOY")
        return "DEPLOY", best
    else:
        improvement_gap = target_throughput / best["throughput"] - 1
        print(f"⚠️  Below target ({best['throughput']:.0f} < {target_throughput:.0f})")
        print(f"   Gap: {improvement_gap*100:.1f}%")

        if iteration < max_iterations - 1:
            print(f"➡️  Decision: ITERATE (try to close gap)")
            return "ITERATE", best
        else:
            print(f"➡️  Max iterations reached")
            print(f"➡️  Decision: DEPLOY BEST EFFORT")
            return "DEPLOY_BEST_EFFORT", best
```

### Step 4: Refinement

```python
def refine_constraints(validation_results, old_constraints):
    """Refine search space based on validation results."""

    print(f"\n{'='*70}")
    print(f"Refinement Phase")
    print(f"{'='*70}")

    new_constraints = old_constraints.copy()
    refinements = []

    # Analyze failures
    failures = [r for r in validation_results if r["status"] == "FAILED"]

    # Rule 1: OOM → Increase sharding
    oom_count = sum(1 for r in failures if r.get("error") == "OOM")
    if oom_count > 0:
        print(f"\n🔍 Detected {oom_count} OOM failures")
        new_constraints["fsdp_level"] = min(old_constraints["fsdp_level"] + 1, 3)
        new_constraints["activation_checkpoint"] = True
        new_constraints["dp_min"] = max(old_constraints["dp_min"], 2)
        refinements.append("Increase FSDP level + enable activation checkpoint")

    # Rule 2: FSDP conflict → Force DP ≥ 2
    fsdp_count = sum(1 for r in failures if r.get("error") == "FSDP_CONFLICT")
    if fsdp_count > 0:
        print(f"\n🔍 Detected {fsdp_count} FSDP conflicts")
        new_constraints["dp_min"] = 2
        refinements.append("Force DP >= 2")

    # Rule 3: Communication bottleneck → Reduce TP/CP
    successful = [r for r in validation_results if r["status"] == "SUCCESS"]
    comm_bottleneck_count = 0
    for r in successful:
        # Actual >> PSA estimate → communication underestimated
        if r["step_time"] > r["config"]["est_step_time"] * 2:
            comm_bottleneck_count += 1

    if comm_bottleneck_count > 0:
        print(f"\n🔍 Detected {comm_bottleneck_count} communication bottlenecks")
        # Reduce max TP/CP
        new_constraints["tp_range"] = [x for x in old_constraints["tp_range"] if x <= 4]
        new_constraints["cp_range"] = [x for x in old_constraints["cp_range"] if x <= 2]
        refinements.append("Reduce TP/CP limits (communication dominates)")

    # Rule 4: All below target but no clear pattern → Try extremes
    if successful and not refinements:
        best_throughput = max(r["throughput"] for r in successful)
        print(f"\n🔍 Best throughput {best_throughput:.0f}, but no failure pattern")

        # Try more aggressive DP (less communication per device)
        new_constraints["dp_min"] = max(old_constraints["dp_min"] * 2, 4)
        new_constraints["tp_range"] = [1, 2]  # Minimize TP
        new_constraints["cp_range"] = [1]     # Minimize CP
        refinements.append("Maximize DP, minimize TP/CP (reduce per-device comm)")

    print(f"\n📝 Refinements:")
    for i, ref in enumerate(refinements, 1):
        print(f"  {i}. {ref}")

    print(f"\n🔧 Updated Constraints:")
    print(f"  TP range: {new_constraints['tp_range']}")
    print(f"  CP range: {new_constraints['cp_range']}")
    print(f"  EP range: {new_constraints['ep_range']}")
    print(f"  PP range: {new_constraints['pp_range']}")
    print(f"  DP min: {new_constraints['dp_min']}")
    print(f"  FSDP level: {new_constraints['fsdp_level']}")
    print(f"  Activation checkpoint: {new_constraints['activation_checkpoint']}")

    return new_constraints
```

### Main Loop

```python
def strategy_optimizer_main(model, hardware, objective):
    """Main orchestration loop."""

    print(f"\n{'='*70}")
    print(f"STRATEGY OPTIMIZER")
    print(f"{'='*70}")
    print(f"Model: {model.name} ({model.num_layers} layers)")
    print(f"Hardware: {hardware.device_count}× {hardware.device_type}")
    print(f"Target: {objective.target_throughput} tokens/s (vs baseline)")
    print(f"Max iterations: 3")

    # Initialize
    state = initialize_optimizer(model, hardware, objective)

    # Iteration loop
    while state["iteration"] < state["max_iterations"]:
        iteration = state["iteration"]

        print(f"\n{'#'*70}")
        print(f"# ITERATION {iteration + 1} / {state['max_iterations']}")
        print(f"{'#'*70}")

        # Step 1: PSA
        psa_candidates = call_psa(model, hardware, state["constraints"], iteration)

        # Step 2: Validation
        validation_results = validate_candidates(psa_candidates, model, hardware)

        if validation_results is None:
            # User hasn't run validations yet
            return "PAUSED"

        # Record history
        state["history"].append({
            "iteration": iteration + 1,
            "psa_candidates": psa_candidates,
            "validation_results": validation_results,
        })

        # Update best
        successful = [r for r in validation_results if r["status"] == "SUCCESS"]
        if successful:
            current_best = max(successful, key=lambda x: x["throughput"])
            if not state["best_so_far"] or current_best["throughput"] > state["best_so_far"]["throughput"]:
                state["best_so_far"] = current_best

        # Step 3: Decision
        decision, winner = make_decision(
            validation_results,
            state["target"],
            iteration,
            state["max_iterations"]
        )

        if decision == "DEPLOY":
            return deploy(winner, state)
        elif decision == "DEPLOY_BEST_EFFORT":
            return deploy_best_effort(state["best_so_far"], state)
        elif decision == "FAILURE":
            return failure_report(state)
        elif decision == "ITERATE":
            # Step 4: Refine
            state["constraints"] = refine_constraints(validation_results, state["constraints"])
            state["iteration"] += 1
            continue

    # Should not reach here
    if state["best_so_far"]:
        return deploy_best_effort(state["best_so_far"], state)
    else:
        return failure_report(state)
```

### Deployment

```python
def deploy(winner, state):
    """Generate deployment artifacts."""

    print(f"\n{'='*70}")
    print(f"✅ DEPLOYMENT")
    print(f"{'='*70}")

    config = winner["config"]

    print(f"\n🏆 Optimal Strategy:")
    print(f"   TP={config['tp']}, CP={config['cp']}, EP={config['ep']}, "
          f"PP={config['pp']}, DP={config['dp']}")
    print(f"\n📊 Performance:")
    print(f"   Throughput: {winner['throughput']:.0f} tokens/s")
    print(f"   Memory: {winner['memory']:.1f} GB")
    print(f"   Step Time: {winner['step_time']:.1f} ms")
    print(f"\n🔄 Iterations: {state['iteration'] + 1}")

    # Generate production config
    prod_config_path = "optimizer_output/production_config.yaml"
    generate_production_config(config, prod_config_path)

    # Generate deployment guide
    guide_path = "optimizer_output/DEPLOYMENT_GUIDE.md"
    generate_deployment_guide(winner, state, guide_path)

    # Generate final report
    report_path = "optimizer_output/OPTIMIZATION_REPORT.md"
    generate_final_report(state, winner, report_path)

    print(f"\n📄 Generated Files:")
    print(f"   - {prod_config_path}")
    print(f"   - {guide_path}")
    print(f"   - {report_path}")

    print(f"\n✅ Ready for production deployment!")

    return {
        "status": "SUCCESS",
        "config": config,
        "performance": {
            "throughput": winner["throughput"],
            "memory": winner["memory"],
        },
        "iterations": state["iteration"] + 1,
        "files": [prod_config_path, guide_path, report_path],
    }
```

---

**This workflow provides the complete orchestration logic for strategy-optimizer.**
