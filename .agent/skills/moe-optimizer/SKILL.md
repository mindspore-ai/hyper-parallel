---
name: moe-optimizer
description: >
  End-to-end MoE training optimization: diagnose load imbalance,
  All-to-All overhead, stragglers; validate correctness; iteratively
  refine EP/TP/capacity/routing strategies. Orchestrates moe-profiler,
  parallel-strategy-analyzer, and strategy-validator in a closed loop.
---

# MoE Optimizer

**Orchestration-only skill** — Combines other skills to optimize MoE training. Does not perform analysis itself.

## When to use

- MoE training performance issues (low throughput, high memory)
- Load imbalance detected (some experts overloaded)
- High All-to-All communication overhead (> 30% of step time)
- Straggler ranks slowing down training
- Need automated MoE optimization with correctness validation

**Prerequisites:**
- MoE model training logs or ability to run training
- Target hardware available (for validation runs)
- Baseline throughput known or measurable
- 1-3 hours for complete optimization cycle

## Trigger patterns

- "Optimize MoE training performance"
- "Reduce MoE load imbalance"
- "Reduce All-to-All communication overhead"
- "MoE training optimization"
- User explicitly requests `/optimize-moe`

## Dependencies (skills called by this orchestrator)

### Required (pre-loaded)
- **moe-profiler**: Extract baseline MoE facts (load, communication, correctness)

### Called during workflow
- **parallel-strategy-analyzer**: Generate strategy candidates with MoE-specific rules
- **strategy-validator**: Validate candidate configs via actual training runs
- **routing-validator**: Verify routing correctness (token integrity, gradient flow)

## Execution Flow

**Closed-loop optimization with 1-3 iterations:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  MoE Optimizer Workflow                          │
└─────────────────────────────────────────────────────────────────┘

Input: Model + Training Logs/Config + Hardware + Objective
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Baseline Profiling                                     │
│  - Call moe-profiler on training logs                           │
│  - Extract: load imbalance, AllToAll ratio, routing integrity   │
│  - Identify problem types:                                      │
│    * Load imbalance (factor > 1.5)                              │
│    * Communication bottleneck (ratio > 0.3)                     │
│    * Stragglers (rank variance > 20%)                           │
│  Output: Baseline metrics + problem diagnosis                   │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Strategy Generation                                    │
│  - Call parallel-strategy-analyzer with:                        │
│    * Model config (num_experts, top_k, etc.)                    │
│    * Hardware config                                            │
│    * Constraints from Phase 1 diagnosis:                        │
│      - If load_imbalance: suggest capacity_factor, expert_bias  │
│      - If comm_bottleneck: reduce ep_size, increase TP          │
│      - If stragglers: rebalance expert-to-rank mapping          │
│  Output: Top-3 candidate configs                                │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Dual Validation                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 3a. Performance Validation                                 │ │
│  │  - Call strategy-validator for each Top-3 candidate       │ │
│  │  - Run 50-step training benchmark                         │ │
│  │  - Extract: throughput, memory, step_time, stability      │ │
│  │  Output: Performance metrics per candidate                │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 3b. Correctness Validation                                 │ │
│  │  - Call routing-validator on validation checkpoints       │ │
│  │  - Check:                                                  │ │
│  │    * Token conservation (input == output count)           │ │
│  │    * Routing integrity (expert-major order)               │ │
│  │    * Gradient flow (all experts receive gradients)        │ │
│  │  Output: PASS / FAIL per candidate                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Decision & Refinement                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ if any(performance_ok AND correctness_ok):              │   │
│  │   → Select best candidate → Deploy ✓                    │   │
│  │                                                          │   │
│  │ elif iteration < max_iterations (default: 3):           │   │
│  │   → Analyze failure modes:                              │   │
│  │      - OOM → increase FSDP, enable activation ckpt      │   │
│  │      - Comm bottleneck → reduce ep_size                 │   │
│  │      - Load imbalance → enable expert_bias, adjust capacity │
│  │      - Correctness fail → revert to safer config        │   │
│  │   → Update constraints                                  │   │
│  │   → Go back to Phase 2                                  │   │
│  │                                                          │   │
│  │ else:                                                    │   │
│  │   → Deploy best-effort (even if < target)              │   │
│  │   → Report limitations                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Report Generation                                      │
│  - Baseline metrics                                             │
│  - All candidates tested (with pass/fail status)                │
│  - Selected config + rationale                                  │
│  - Performance improvement (actual vs baseline)                 │
│  - Correctness validation results                               │
│  - Deployment instructions                                      │
└─────────────────────────────────────────────────────────────────┘
  ↓
Output: Optimized Config + Validation Report
```

## Workflow References (load before each phase)

Each phase has detailed instructions in `workflows/`:

1. [01-baseline-profiling.md](workflows/01-baseline-profiling.md) — Extract baseline facts via moe-profiler
2. [02-problem-diagnosis.md](workflows/02-problem-diagnosis.md) — Classify problem types from metrics
3. [03-strategy-generation.md](workflows/03-strategy-generation.md) — Generate candidates via PSA
4. [04-validation.md](workflows/04-validation.md) — Dual validation (performance + correctness)
5. [05-iterative-refinement.md](workflows/05-iterative-refinement.md) — Failure analysis & constraint updates

## References (on-demand)

Load these when interpreting results or making decisions:

- [references/optimization-strategies.md](references/optimization-strategies.md) — Problem → Solution mapping
- [references/case-studies.md](references/case-studies.md) — Verified optimization examples
- [references/workflow.md](references/workflow.md) — Detailed orchestration logic

## Example Usage

### Scenario 1: Optimize Qwen3-30B-A3B on 64× Ascend 910B

**Input:**
```yaml
model:
  name: "Qwen3-30B-A3B"
  num_layers: 48
  num_experts: 128
  top_k: 8

hardware:
  device_type: "Ascend910B"
  device_count: 64
  memory_per_device_gb: 64

baseline:
  log_dir: "/path/to/baseline_logs"
  throughput: 94000  # tokens/s

objective:
  target_throughput: 113000  # 1.2x improvement
  max_memory_per_device: 60
```

**Execution:**
```
User: Optimize Qwen3-30B-A3B training on 64 Ascend 910B devices

Agent: Loading moe-optimizer skill...

[Iteration 1]
Phase 1: Baseline profiling
  - Load imbalance: 1.68 (moderate)
  - AllToAll ratio: 0.32 (high)
  - Identified problem: Communication bottleneck

Phase 2: Strategy generation
  - PSA Top-3:
    1. TP=4, EP=4, DP=4
    2. TP=2, EP=8, DP=4
    3. TP=8, EP=2, DP=4

Phase 3: Validation
  - Candidate 1: 108K tok/s, PASS correctness
  - Candidate 2: 102K tok/s, PASS correctness
  - Candidate 3: 95K tok/s, PASS correctness

  Best: 108K < 113K (target not met)

Phase 4: Refinement
  - Analysis: Candidates 2&3 have higher ep_size → more comm
  - Constraint: Force ep_size ≤ 4
  - Go to Iteration 2

[Iteration 2]
Phase 2: Strategy generation (with ep_size ≤ 4)
  - PSA Top-3:
    1. TP=4, EP=4, DP=4, capacity=1.25
    2. TP=4, EP=2, DP=8, capacity=1.25
    3. TP=2, EP=4, DP=8, expert_bias=True

Phase 3: Validation
  - Candidate 1: 116K tok/s ✓, PASS correctness
  - Candidate 2: 111K tok/s, PASS correctness
  - Candidate 3: 119K tok/s ✓, PASS correctness

  Best: 119K > 113K ✓ (target met!)

Phase 5: Report
  ✅ Optimal config: TP=2, EP=4, DP=8, expert_bias=True
  ✅ Throughput: 119K tok/s (+26.6% vs baseline)
  ✅ Correctness: PASS (all checks)
  ✅ Iterations: 2
```

## Output Format

**Final Report Structure:**

```json
{
  "status": "SUCCESS",
  "iterations": 2,
  "baseline": {
    "throughput": 94000,
    "load_imbalance_factor": 1.68,
    "alltoall_ratio": 0.32,
    "problems_identified": ["communication_bottleneck", "moderate_load_imbalance"]
  },
  "optimal_config": {
    "tp_size": 2,
    "ep_size": 4,
    "dp_size": 8,
    "capacity_factor": 1.25,
    "expert_bias": true
  },
  "performance": {
    "throughput": 119000,
    "improvement": 1.266,
    "memory_per_device_gb": 58.3,
    "load_imbalance_factor": 1.22,
    "alltoall_ratio": 0.18
  },
  "correctness": {
    "routing_integrity": "PASS",
    "gradient_flow": "PASS",
    "token_conservation": "PASS"
  },
  "candidates_tested": 6,
  "total_validation_time_minutes": 48
}
```

## Handoff to Other Skills/Agents

### When this skill should NOT be used

→ **Handoff to `/code-review`**:
- Need to review MoE implementation code quality
- Suspect bugs in routing logic

→ **Handoff to `parallel-strategy-analyzer`**:
- Only need strategy analysis, no actual optimization
- Non-MoE parallelism optimization

→ **Handoff to `ep-dev-expert` agent**:
- Need to modify EP implementation itself
- Add new routing algorithms

→ **Handoff to `pangu-perf` + `profile-explorer`**:
- Need deep profiling beyond MoE metrics
- Investigate kernel-level performance

## Safety & Limitations

- **Resource intensive**: Each validation run takes ~5-10 minutes
- **Hardware requirement**: Must have target hardware available
- **Time budget**: Full optimization takes 1-3 hours (3 iterations × 3 candidates × 10 min)
- **No guarantees**: Target may not be achievable within hardware limits
- **Correctness first**: Never deploys config that fails correctness checks

**Pre-flight checks:**
- Verify model is actually MoE (has num_experts > 0)
- Verify baseline logs are valid and parseable
- Verify target hardware is accessible
- Warn user about estimated time and cost

**Abort conditions:**
- All candidates fail correctness checks
- All candidates OOM
- Hardware unavailable
- User timeout exceeded

## Success Metrics

- ✅ **Target met**: Final throughput ≥ target
- ✅ **Best-effort**: Improved throughput even if < target
- ✅ **Correctness**: All checks pass
- ✅ **Documented**: Complete report with rationale

---

**Next Steps:**
1. Load appropriate workflow file for current phase
2. Follow workflow instructions precisely
3. Aggregate results and make decisions
4. Generate final report for user
