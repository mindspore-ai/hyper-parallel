# MoE Optimizer

**End-to-end MoE training optimization skill** — orchestrates multiple skills to optimize MoE models.

## Overview

The `moe-optimizer` skill automates the complete MoE optimization workflow:
1. Profile baseline training (via `moe-profiler`)
2. Diagnose problems (load imbalance, communication, stragglers)
3. Generate strategy candidates (via `parallel-strategy-analyzer`)
4. Validate candidates (performance + correctness)
5. Iteratively refine (up to 3 iterations)
6. Deploy optimal configuration

**Key Features:**
- **Orchestration-only**: Combines existing skills, doesn't do analysis itself
- **Closed-loop**: Iterative refinement based on validation feedback
- **Dual validation**: Both performance and correctness checks
- **Automated**: Minimal user intervention required

## Quick Start

### Prerequisites

- MoE model training logs (baseline)
- Target hardware available
- 1-3 hours for complete optimization

### Example Usage

```bash
# User provides baseline logs and target
user_input = {
  "model": "Qwen/Qwen3-30B-A3B",
  "baseline_logs": "/path/to/training_logs",
  "hardware": {
    "device_type": "Ascend910B",
    "device_count": 64
  },
  "target_throughput": 113000  # tokens/s
}

# Agent invokes moe-optimizer
# → Runs 1-3 iterations automatically
# → Returns optimal configuration
```

## Workflow Overview

```
Input: Baseline logs + Target
  ↓
Phase 1: Baseline Profiling (moe-profiler)
  → Extract load, communication, correctness metrics
  ↓
Phase 2: Problem Diagnosis
  → Classify: load imbalance / comm bottleneck / stragglers
  ↓
Phase 3: Strategy Generation (parallel-strategy-analyzer)
  → Generate Top-3 candidates with MoE adjustments
  ↓
Phase 4: Validation (strategy-validator + routing-validator)
  → Performance: 50-step benchmark
  → Correctness: routing integrity + gradient flow
  ↓
Decision: Pass? → Deploy ✓
         Fail? → Iterate (Phase 2-4, max 3 times)
  ↓
Phase 5: Report
  → Optimal config + improvement + deployment instructions
```

## Dependencies

### Skills Called

1. **moe-profiler** (required)
   - Extracts baseline MoE facts
   - Generates 15 artifacts

2. **parallel-strategy-analyzer** (required)
   - Generates strategy candidates
   - Provides performance estimates

3. **strategy-validator** (required)
   - Validates candidates via actual training
   - Measures throughput, memory, stability

4. **routing-validator** (required for correctness)
   - Validates routing integrity
   - Checks gradient flow

## Typical Results

### Example: Qwen3-30B-A3B Optimization

**Input:**
- Baseline: 94K tokens/s
- Target: 113K tokens/s (+20%)

**Output (after 2 iterations):**
- Achieved: 119K tokens/s (+26.6%)
- Config: TP=2, EP=4, DP=8, capacity=1.5, expert_bias=true
- Correctness: PASS (all checks)
- Time: 1.5 hours

**Key Changes:**
- Reduced EP: 8 → 4 (lower communication)
- Increased capacity: 1.0 → 1.5 (better load balance)
- Enabled expert_bias (dynamic load correction)

## Documentation

### Workflows (load before each phase)

- [01-baseline-profiling.md](workflows/01-baseline-profiling.md)
- [02-problem-diagnosis.md](workflows/02-problem-diagnosis.md)
- [03-strategy-generation.md](workflows/03-strategy-generation.md)
- [04-validation.md](workflows/04-validation.md)
- [05-iterative-refinement.md](workflows/05-iterative-refinement.md)

### References (on-demand)

- [optimization-strategies.md](references/optimization-strategies.md) — Problem → Solution mappings
- [case-studies.md](references/case-studies.md) — Verified real-world examples
- [workflow.md](references/workflow.md) — Detailed orchestration logic

## Output Structure

### Success Case

```json
{
  "status": "SUCCESS",
  "iterations": 2,
  "optimal_config": {
    "tp_size": 2,
    "ep_size": 4,
    "dp_size": 8,
    "capacity_factor": 1.5,
    "expert_bias": true
  },
  "performance": {
    "baseline_throughput": 94000,
    "final_throughput": 119000,
    "improvement": 1.266,
    "target_met": true
  },
  "correctness": {
    "routing_integrity": "PASS",
    "gradient_flow": "PASS"
  },
  "total_time_minutes": 92
}
```

### Failure/Best-Effort Case

```json
{
  "status": "PARTIAL_SUCCESS",
  "iterations": 3,
  "optimal_config": {...},
  "performance": {
    "final_throughput": 108000,
    "improvement": 1.149,
    "target_met": false
  },
  "limitations": [
    "Target (113K) not achieved",
    "Recommend hardware upgrade or code-level optimization"
  ]
}
```

## Common Issues

### Issue: All candidates fail validation

**Cause:** Target too ambitious or correctness issues

**Solution:**
- Check if target is realistic (< 50% improvement)
- Verify baseline correctness first
- May need code-level optimization

### Issue: Optimization takes too long

**Cause:** 3 iterations × 3 candidates × 10 min = 90 min

**Solution:**
- Set `max_iterations: 2` to reduce time
- Run validation in parallel if hardware allows
- Use shorter validation runs (30 steps instead of 50)

### Issue: Memory OOM during validation

**Cause:** Candidates have insufficient sharding

**Solution:**
- Enable FSDP in constraints
- Increase TP size minimum
- Reduce batch size

## Limitations

- **MoE-only**: Assumes pure MoE model (not hybrid MoE-dense)
- **Hardware-dependent**: Requires actual hardware for validation
- **Time-intensive**: 1-3 hours for complete optimization
- **Target bounds**: Typically achieves 10-40% improvement, not 2x+

## Integration

### With Other Skills

- **Before moe-optimizer**: Run `hf-model-integration` if model not integrated
- **Before moe-optimizer**: Run `hf-model-precision-validation` for baseline correctness
- **After moe-optimizer**: Use `autogit` to commit optimized config

### With Commands

```bash
# Typical workflow
/integrate-model Qwen3-30B-A3B      # hf-model-integration
/validate-precision                 # hf-model-precision-validation
/optimize-moe                       # This skill
/commit                            # autogit
```

## Development

### Adding New Problem Types

1. Update `02-problem-diagnosis.md` with new pattern
2. Add solution to `optimization-strategies.md`
3. Update `05-iterative-refinement.md` with refinement logic

### Adding New Metrics

1. Extend `moe-profiler` to extract new metric
2. Update diagnosis in Phase 2 to use it
3. Add threshold checks in Phase 4 validation

## License

Apache 2.0 (same as HyperParallel project)
