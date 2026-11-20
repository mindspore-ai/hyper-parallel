# MoE Optimizer Workflow

**Detailed orchestration logic** for the moe-optimizer skill.

This document supplements the phase-specific workflow files with cross-phase logic.

---

## State Management

### Execution State Structure

```python
execution_state = {
    "workflow_id": "moe_opt_20240120_143022",
    "model_config": {...},
    "hardware_config": {...},
    "user_objective": {...},

    "current_phase": "baseline_profiling",  # or "strategy_generation", etc.
    "current_iteration": 0,
    "max_iterations": 3,

    "phases_completed": [],
    "phases_failed": [],

    "baseline_metrics": None,  # Set in Phase 1
    "problems_identified": [],  # Set in Phase 2
    "current_constraints": {},  # Updated in Phase 2, 5
    "candidates": [],  # Set in Phase 3
    "validation_results": [],  # Set in Phase 4
    "best_candidate": None,  # Set in Phase 4 or 5

    "iteration_history": [],  # Track all iterations
}
```

### State Persistence

Save state after each phase:

```python
import json
from pathlib import Path

state_dir = Path(f"./moe_optimization_{execution_state['workflow_id']}")
state_dir.mkdir(exist_ok=True)

def save_state(execution_state):
    state_file = state_dir / "execution_state.json"
    with open(state_file, 'w') as f:
        json.dump(execution_state, f, indent=2)

def load_state(workflow_id):
    state_file = Path(f"./moe_optimization_{workflow_id}/execution_state.json")
    with open(state_file) as f:
        return json.load(f)
```

**Benefits:**
- Resume after interruption
- Audit trail for debugging
- Report generation

---

## Phase Transitions

### Transition Logic

```python
def next_phase(execution_state):
    current = execution_state["current_phase"]
    iteration = execution_state["current_iteration"]

    # Normal flow: 1 → 2 → 3 → 4
    if current == "baseline_profiling":
        # Check if any problems found
        if not execution_state["problems_identified"]:
            return {"phase": "report", "reason": "No optimization needed"}
        return {"phase": "problem_diagnosis"}

    elif current == "problem_diagnosis":
        # Check for critical correctness issues
        if any(p.startswith("correctness") for p in execution_state["problems_identified"]):
            return {"phase": "abort", "reason": "Fix correctness before optimization"}
        return {"phase": "strategy_generation"}

    elif current == "strategy_generation":
        return {"phase": "validation"}

    elif current == "validation":
        # Check if any candidates passed
        passing = [r for r in execution_state["validation_results"] if r["overall_pass"]]

        if passing:
            # Success path
            execution_state["best_candidate"] = select_best(passing)
            return {"phase": "report", "reason": "Candidate passed"}

        elif iteration < execution_state["max_iterations"]:
            # Refinement path
            return {"phase": "iterative_refinement"}

        else:
            # Max iterations reached, deploy best-effort
            execution_state["best_candidate"] = select_best_effort(
                execution_state["validation_results"]
            )
            return {"phase": "report", "reason": "Max iterations reached"}

    elif current == "iterative_refinement":
        # Check refinement decision
        decision = execution_state.get("refinement_decision", {})

        if decision.get("action") == "ABORT":
            return {"phase": "abort", "reason": decision["reason"]}

        elif decision.get("action") == "DEPLOY_BEST_EFFORT":
            return {"phase": "report", "reason": "Optimization converged"}

        else:
            # Continue to next iteration
            execution_state["current_iteration"] += 1
            return {"phase": "problem_diagnosis"}  # Re-diagnose with refined constraints
```

---

## Skill Invocation Helpers

### Wrapper for moe-profiler

```python
def invoke_moe_profiler(log_dir, profile_dir=None, checkpoint_dir=None):
    """Wrapper for moe-profiler skill invocation"""

    # Prepare input
    profiler_input = {
        "log_dir": log_dir,
        "profile_dir": profile_dir,
        "checkpoint_dir": checkpoint_dir,
        "output_dir": f"./baseline_moe_facts"
    }

    # Invoke skill (actual invocation depends on harness)
    # Option 1: Via Skill tool
    # result = Skill("moe-profiler", args=profiler_input)

    # Option 2: Via script
    import subprocess
    cmd = [
        "python",
        ".agent/skills/moe-profiler/scripts/extract_moe_facts.py",
        "--log-dir", log_dir,
        "--output-dir", profiler_input["output_dir"]
    ]
    if profile_dir:
        cmd.extend(["--profile-dir", profile_dir])
    if checkpoint_dir:
        cmd.extend(["--checkpoint-dir", checkpoint_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"moe-profiler failed: {result.stderr}")

    # Parse output
    import json
    with open(f"{profiler_input['output_dir']}/moe_facts.json") as f:
        return json.load(f)
```

### Wrapper for parallel-strategy-analyzer

```python
def invoke_parallel_strategy_analyzer(model_config, hardware_config, constraints):
    """Wrapper for PSA skill invocation"""

    psa_input = {
        "model": model_config,
        "hardware": hardware_config,
        "constraints": constraints,
        "output": "./strategy_candidates.json"
    }

    # Invoke PSA (implementation depends on how PSA is exposed)
    # This is a placeholder - actual invocation method depends on PSA's interface

    # For now, assume PSA outputs to JSON file
    with open("./strategy_candidates.json") as f:
        return json.load(f)
```

### Wrapper for strategy-validator

```python
def invoke_strategy_validator(model_path, candidate_config, validation_steps=50):
    """Wrapper for strategy-validator skill invocation"""

    validator_input = {
        "model_path": model_path,
        "config": candidate_config,
        "validation_steps": validation_steps,
        "output_dir": f"./validation_candidate_{candidate_config['rank']}"
    }

    # Invoke validator
    # Implementation depends on validator's interface

    # Parse results
    with open(f"{validator_input['output_dir']}/metrics.json") as f:
        return json.load(f)
```

---

## Error Handling

### Error Types

```python
class MoEOptimizerError(Exception):
    """Base exception for moe-optimizer"""
    pass

class CorrectnessError(MoEOptimizerError):
    """Critical correctness issue detected"""
    pass

class ResourceError(MoEOptimizerError):
    """Resource unavailable (hardware, time, etc.)"""
    pass

class ConvergenceError(MoEOptimizerError):
    """Optimization not converging"""
    pass
```

### Error Recovery

```python
def handle_error(error, execution_state):
    """Handle errors with appropriate recovery"""

    if isinstance(error, CorrectnessError):
        # Correctness errors are critical - abort
        execution_state["status"] = "ABORTED"
        execution_state["abort_reason"] = str(error)
        return {"action": "abort"}

    elif isinstance(error, ResourceError):
        # Try to salvage with reduced resources
        if "device" in str(error):
            # Reduce device count
            execution_state["hardware_config"]["device_count"] //= 2
            return {"action": "retry"}
        else:
            return {"action": "abort"}

    elif isinstance(error, ConvergenceError):
        # Deploy best-effort
        return {"action": "deploy_best_effort"}

    else:
        # Unknown error - log and abort
        import traceback
        execution_state["error_trace"] = traceback.format_exc()
        return {"action": "abort"}
```

---

## Reporting

### Final Report Generation

```python
def generate_final_report(execution_state):
    """Generate comprehensive final report"""

    report = {
        "workflow_id": execution_state["workflow_id"],
        "status": execution_state.get("status", "SUCCESS"),
        "iterations_completed": execution_state["current_iteration"],

        "baseline": execution_state["baseline_metrics"],
        "problems_identified": execution_state["problems_identified"],

        "candidates_tested": len(execution_state["validation_results"]),
        "candidates_passed": sum(
            1 for r in execution_state["validation_results"] if r["overall_pass"]
        ),

        "optimal_config": execution_state["best_candidate"]["config"],
        "performance": {
            "baseline_throughput": execution_state["baseline_metrics"]["throughput"],
            "final_throughput": execution_state["best_candidate"]["performance"]["actual"]["throughput"],
            "improvement": execution_state["best_candidate"]["performance"]["actual"]["throughput"] /
                          execution_state["baseline_metrics"]["throughput"],
            "target_met": execution_state["best_candidate"]["performance"]["actual"]["throughput"] >=
                         execution_state["user_objective"]["target_throughput"]
        },

        "correctness": execution_state["best_candidate"]["correctness"],

        "total_time_minutes": (datetime.now() - execution_state["start_time"]).total_seconds() / 60,

        "deployment_instructions": generate_deployment_instructions(execution_state["best_candidate"])
    }

    # Save report
    report_file = Path(execution_state["state_dir"]) / "final_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate markdown summary
    markdown = generate_markdown_report(report)
    with open(Path(execution_state["state_dir"]) / "final_report.md", 'w') as f:
        f.write(markdown)

    return report
```

---

## Time Budget Management

### Time Estimation

```python
def estimate_remaining_time(execution_state):
    """Estimate time to completion"""

    # Time per validation: ~10 minutes
    # 3 candidates × 10 min = 30 min per iteration
    # Max 3 iterations = 90 min worst case

    iteration = execution_state["current_iteration"]
    max_iterations = execution_state["max_iterations"]

    remaining_iterations = max_iterations - iteration
    time_per_iteration_minutes = 30  # Conservative estimate

    estimated_remaining = remaining_iterations * time_per_iteration_minutes

    return {
        "estimated_remaining_minutes": estimated_remaining,
        "worst_case_minutes": max_iterations * time_per_iteration_minutes
    }
```

### Timeout Handling

```python
def check_timeout(execution_state):
    """Check if user timeout exceeded"""

    if "timeout_minutes" not in execution_state["user_objective"]:
        return False  # No timeout set

    elapsed = (datetime.now() - execution_state["start_time"]).total_seconds() / 60
    timeout = execution_state["user_objective"]["timeout_minutes"]

    if elapsed > timeout:
        # Timeout exceeded - deploy best so far
        execution_state["status"] = "TIMEOUT"
        return True

    return False
```

---

## Parallelization

### Parallel Candidate Validation

If hardware allows, validate multiple candidates in parallel:

```python
def validate_candidates_parallel(candidates, execution_state):
    """Validate all candidates in parallel if resources allow"""

    hardware = execution_state["hardware_config"]
    devices_per_candidate = hardware["device_count"]

    # Check if we have multiple nodes or enough GPUs
    if hardware.get("num_nodes", 1) >= len(candidates):
        # Can run in parallel across nodes
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            futures = [
                executor.submit(invoke_strategy_validator, model_path, c["config"])
                for c in candidates
            ]
            results = [f.result() for f in futures]

        return results

    else:
        # Sequential validation
        return [
            invoke_strategy_validator(model_path, c["config"])
            for c in candidates
        ]
```

---

**See also:** Individual phase workflows in `workflows/` directory.
