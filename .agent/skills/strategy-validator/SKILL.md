---
name: strategy-validator
description: >
  Validate and optimize parallel training strategies through iterative benchmark runs.
  Generates candidate configurations, executes distributed training, extracts metrics,
  and identifies optimal parallelism settings. Can work standalone or with PSA.
---

# Strategy Validator

**Iterative strategy optimization through empirical validation.**
Automatically generates and tests multiple parallelism candidates, identifies bottlenecks,
and converges to optimal configuration.

## When to use

- **After baseline training** to find optimal parallelism strategy
- After `/parallel-strategy-analyzer` to validate recommendations (optional)
- To optimize existing configurations (e.g., remove unnecessary dimensions)
- To explore parallelism space for new model/hardware combinations
- **MoE-specific**: Validate MoE strategies with expert load and All-to-All metrics

**Prerequisites:**
- Baseline training run completed (for comparison) or target throughput known
- Target hardware available (NPU/GPU cluster, e.g., 8× Ascend 910B)
- Model checkpoint accessible (e.g., `/path/to/Qwen3-30B-A3B`)
- 5-10 minutes per validation run (50 training steps)
- Patience for iterative exploration (typically 3-5 candidates)
- **For MoE models**: Load [references/moe-metrics.md](references/moe-metrics.md) for additional metrics

## Pre-flight Safety Checks

**Before running validation, the following checks are performed automatically:**

### 1. Device Availability
```bash
# NPU check
npu-smi info

# GPU check
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

**Validation**: Must have required number of devices (e.g., 8 for world_size=8)
**Failure action**: Abort with error message

### 2. Resource Isolation
```bash
# Check for conflicting training jobs
pgrep -f torch.distributed
```

**Validation**: Warn if other distributed jobs detected
**Failure action**: Warning only, user can proceed

### 3. Runtime Estimation
```python
# Estimated time = warmup (10 steps × 1.5 × avg_step_time)
#                  + training (40 steps × avg_step_time)
#                  + overhead (60s)
# Typical: 50 steps × 0.7s = ~5-7 minutes
```

**Display**: Show estimated completion time
**User action**: Confirm if first run or --confirm flag

### 4. User Confirmation (First Run)

On first validation run in a session, user is prompted:
```
VALIDATION SAFETY CHECK
======================================================================
Configuration:
  Strategy: TP=1, CP=1, EP=4, DP=2
  Devices: 8
  Steps: 50

Estimated runtime:
  Total: ~6.5 minutes

⚠️  First validation run - approval required

Proceed with validation? (yes/no):
```

**Bypass**: Use `--auto-approve` flag for automated workflows
**Force**: Use `--confirm` flag to always prompt

### 5. Output Format

Validation outputs **standardized JSON metrics** to stdout for parsing:
```json
{
  "step": 50,
  "throughput_tokens_per_sec": 4486.2,
  "memory_allocated_gb": 7.12,
  "step_time_ms": 174.5,
  "mfu_percent": 42.1,
  "bubble_ratio": 0.053
}
```

**Tools available**:
- `tests/benchmarks/strategy_validation/safety_checks.py` - Pre-flight checks
- `tests/benchmarks/strategy_validation/metrics_output.py` - Standardized output

## Validation flow (iterative, load workflow before each phase)

### Mode 1: Iterative Optimization (Recommended)

```text
Phase 1: Baseline Analysis
  - Extract baseline metrics (throughput, memory, step time)
  - Identify bottlenecks (communication vs computation)
  → workflows/01-baseline-analysis.md

Phase 2: Candidate Generation
  - Generate 3-5 candidates based on:
    * Model size (layers, parameters)
    * Sequence length
    * Hardware constraints
    * Baseline bottlenecks
  - Apply heuristics (e.g., avoid CP for seq<512, avoid TP for <10 layers)
  → workflows/02-candidate-generation.md

Phase 3: Batch Validation
  - Run all candidates (parallel or sequential)
  - Monitor for OOM, FSDP errors, hangs
  - Collect metrics automatically
  → workflows/03-batch-validation.md

Phase 4: Comparison & Iteration
  - Compare all candidates vs baseline
  - Identify winner or iterate (if no improvement)
  - Generate deployment-ready config
  → workflows/04-comparison-iteration.md
```

### Mode 2: Single Strategy Validation (Legacy)

```text
For validating a single PSA recommendation or manual config:
  1. Config generation   → workflows/01-config-generation.md
  2. Benchmark execution → workflows/02-benchmark-execution.md
  3. Metrics extraction  → workflows/03-metrics-extraction.md
  4. PSA comparison      → workflows/04-comparison-report.md
```

**Iterative cycle:** Baseline → Generate candidates → Validate all → Pick best → Deploy (or iterate)

## References (on demand)

- [references/config-templates.md](references/config-templates.md) — Training config templates
- [references/benchmark-recipes.md](references/benchmark-recipes.md) — Standard workloads
- [references/metrics-schema.md](references/metrics-schema.md) — Metric definitions & extraction

## Output format

### Iterative Optimization Report

**Final Summary:**

```
Model: Qwen3-30B-A3B (4 layers)
Hardware: 8× Ascend 910B
Sequence Length: 128
Goal: 1.2x throughput improvement

┌───┬─────────────────────┬───────────┬───────────┬──────────┬────────┐
│ # │ Strategy            │ Tokens/s  │ vs Base   │ Memory   │ Status │
├───┼─────────────────────┼───────────┼───────────┼──────────┼────────┤
│ 0 │ Baseline (ref)      │ 1,081     │ 1.00x     │ 6.81 GB  │ Ref    │
│ 1 │ TP=2,CP=1,EP=2,DP=2 │ 2,305     │ 2.13x     │ 6.83 GB  │ ✓ Good │
│ 2 │ TP=2,CP=1,EP=4,DP=1 │ —         │ —         │ —        │ ✗ FSDP │
│ 3 │ TP=4,CP=1,EP=2,DP=1 │ —         │ —         │ —        │ ✗ FSDP │
│ 4 │ TP=1,CP=1,EP=2,DP=4 │ 4,274     │ 3.95x ⭐  │ 6.18 GB  │ ✓ BEST │
└───┴─────────────────────┴───────────┴───────────┴──────────┴────────┘

🏆 Winner: Candidate 4 (TP=1, CP=1, EP=2, DP=4)
  - Throughput: 4274 tokens/s (+295% vs baseline)
  - Training time: -75% (4x faster)
  - Memory: -9% (6.18 GB, more efficient)

Key Insights:
  • CP harmful for short sequences (seq_len < 512)
  • TP harmful for small models (<10 layers)
  • Communication >> Computation (main bottleneck)
  • Simple config (fewer dimensions) often wins

Deployment Config:
  accelerator:
    tp_size: 1  # No tensor parallel
    cp_size: 1  # No context parallel
    ep_size: 2  # Keep expert parallel (MoE requirement)
    pp_size: 1  # No pipeline parallel
  fsdp_config:
    dp_shard_size: 4
    edp_shard_size: 4
```

### Single Strategy Validation (Legacy)

```
Strategy: TP=8, CP=1, EP=2, DP=4
Model: Qwen3-30B-A3B (24 layers)
Hardware: 64× Ascend 910B

┌──────────────────┬────────────┬───────────┬────────┐
│ Metric           │ PSA Est.   │ Actual    │ Delta  │
├──────────────────┼────────────┼───────────┼────────┤
│ Memory/device    │ 58.0 GB    │ 61.3 GB   │ +5.7%  │
│ Throughput       │ 125K tok/s │ 118K tok/s│ -5.6%  │
│ MFU              │ 45%        │ 42%       │ -3 pts │
└──────────────────┴────────────┴───────────┴────────┘

Status: ✓ PASS (within ±20% tolerance)
```

## Key Heuristics (learned from validation experiments)

### Model Size Rules

- **≤10 layers**: TP=1 (communication > computation benefit)
- **10-30 layers**: TP=2
- **30-60 layers**: TP=4
- **>60 layers**: TP=8 or consider PP

### Sequence Length Rules

- **seq_len < 512**: CP=1 (communication dominates)
- **512 ≤ seq_len < 2048**: CP=2 (marginal benefit)
- **2048 ≤ seq_len < 4096**: CP=4
- **seq_len ≥ 4096**: CP=8 (amortizes communication)

### MoE Constraints

- **Always EP ≥ 2** (framework requirement for fused weights)
- EP degree should divide total experts evenly
- Avoid DP=1 (triggers FSDP conflicts in current framework)

### Failure Patterns

- **FSDP + no_grad conflict**: Keep DP ≥ 2
- **MoE weight incompatibility**: Keep EP > 1
- **OOM**: Reduce TP/PP, increase DP (enable FSDP sharding)

## Limitations & Safety

- **Resource isolation:** Each run occupies full device set for 5-10 min
- **Cost:** 50 training steps × number of candidates (typically 3-5)
- **Accuracy:** Micro-benchmark may miss long-term issues (memory fragmentation, rare edge cases)
- **Hardware variance:** ±5-10% throughput variation is normal
- **Iteration limit:** Stop after 5 failed candidates (avoid infinite loops)

**Pre-flight checks:**
- Device availability (via `nvidia-smi` / `npu-smi info`)
- Model checkpoint exists and is readable
- Sufficient disk space for logs (~100 MB per run)
- No conflicting jobs on target devices

**Abort conditions:**
- OOM within first 20 steps → immediate rejection, try reduced parallelism
- Hang (no progress for >60s) → timeout, report communication issue
- Loss NaN/Inf → gradient instability, check config validity
- FSDP errors → adjust DP constraints

**Success criteria:**
- Throughput improvement ≥ target (default 1.2x)
- Memory within device limits
- No runtime errors for 50 continuous steps
- Stable throughput (std < 10% of mean)
