# Strategy Validator - Usage Guide

## Overview

The **strategy-validator** skill optimizes parallel training strategies through iterative empirical validation. It automatically generates candidate configurations, runs benchmarks, and identifies the optimal parallelism strategy.

## Quick Start

### Invocation
```
/validate-strategy
```

The skill will guide you through:
1. Baseline analysis (extract current performance)
2. Candidate generation (3-5 optimized strategies)
3. Batch validation (run all candidates)
4. Comparison & selection (identify winner)

## When to Use

- ✅ After initial training to optimize parallelism strategy
- ✅ When throughput is below expectations
- ✅ Before scaling to larger hardware (validate config first)
- ✅ When model/hardware changes (new baseline needed)

## Prerequisites

1. **Baseline training completed**
   - At least 50 steps of training logs
   - Config file accessible

2. **Hardware available**
   - Target cluster ready (e.g., 8× NPUs)
   - No conflicting jobs

3. **Time budget**
   - ~7 minutes per candidate
   - Typical: 3-5 candidates = 20-35 minutes total

4. **Disk space**
   - ~100 MB per validation run

## Workflow Modes

### Mode 1: Iterative Optimization (Recommended)

**Use when**: You want to find the absolute best configuration

```
Phase 1: Baseline Analysis
  ↓ Extract metrics from existing training

Phase 2: Candidate Generation
  ↓ Generate 3-5 candidates based on heuristics

Phase 3: Batch Validation
  ↓ Run all candidates, collect metrics

Phase 4: Comparison & Iteration
  ↓ Identify winner, decide: deploy or iterate
```

**Output**: Optimal config + deployment guide

### Mode 2: Single Strategy Validation (Legacy)

**Use when**: You have a specific config to validate (e.g., from PSA)

```
Phase 1: Config Generation
  ↓ Convert strategy to training config

Phase 2: Benchmark Execution
  ↓ Run 50-step validation

Phase 3: Metrics Extraction
  ↓ Parse logs, extract throughput/memory

Phase 4: PSA Comparison
  ↓ Compare actual vs estimated
```

**Output**: Pass/Fail + metrics delta

## Key Heuristics

The validator uses these rules (learned from experiments):

### Model Size
- **≤10 layers**: TP=1 (communication > benefit)
- **10-30 layers**: TP=2
- **30-60 layers**: TP=4
- **>60 layers**: TP=8+

### Sequence Length
- **< 512**: CP=1 (communication dominates)
- **512-2048**: CP=2
- **2048-4096**: CP=4
- **≥ 4096**: CP=8

### MoE Models
- **Always EP ≥ 2** (framework requirement)
- Avoid DP=1 (triggers FSDP conflicts)

### Bottleneck Patterns
- **Communication-bound** → Reduce TP/CP, increase DP
- **Computation-bound** → May already be optimal
- **Memory-bound** → Increase DP (more sharding)

## Example Session

### Input
```
Baseline: TP=2, CP=2, EP=2, DP=2
Model: Qwen3-30B-A3B (4 layers)
Hardware: 8× Ascend 910B
Throughput: 1081 tokens/s
Goal: 1.2x improvement
```

### Generated Candidates
```
1. TP=2, CP=1, EP=2, DP=2 — Remove CP
2. TP=2, CP=1, EP=4, DP=1 — Increase EP
3. TP=4, CP=1, EP=2, DP=1 — Increase TP
4. TP=1, CP=1, EP=2, DP=4 — Minimal TP
```

### Results
```
| # | Strategy | Throughput | vs Base | Status |
|---|----------|------------|---------|--------|
| 1 | Remove CP | 2305 t/s | 2.13x | ✓ Good |
| 2 | Increase EP | — | — | ✗ FSDP |
| 3 | Increase TP | — | — | ✗ FSDP |
| 4 | Minimal TP | 4274 t/s | 3.95x | ✓ BEST |
```

### Output
```
🏆 Winner: TP=1, CP=1, EP=2, DP=4
  Throughput: 4274 tokens/s (+295%)
  Training time: -75%
  Memory: -9%

✅ Target exceeded (3.95x > 1.2x)
Status: APPROVED FOR DEPLOYMENT
```

## Failure Handling

### Common Issues

**1. All candidates fail with FSDP errors**
```
→ All used DP=1 (framework constraint)
→ Regenerate with DP ≥ 2 constraint
→ Auto-retry once
```

**2. All candidates worse than baseline**
```
→ Baseline already optimal
→ Document findings
→ No deployment change
```

**3. OOM on all candidates**
```
→ Model too large for hardware
→ Suggest: reduce batch size, increase devices, enable gradient checkpointing
```

**4. Hang/timeout**
```
→ Communication deadlock
→ Check HCCL/NCCL config
→ Reduce parallelism dimensions
```

## Output Artifacts

After successful validation:

### 1. Production Config
```yaml
# validation_experiment/production_config.yaml
accelerator:
  tp_size: 1
  cp_size: 1
  ep_size: 2
  pp_size: 1

fsdp_config:
  dp_shard_size: 4
  edp_shard_size: 4
```

### 2. Final Report
```markdown
# validation_experiment/FINAL_REPORT.md

- Winner configuration
- Performance comparison
- Key insights
- Deployment recommendation
```

### 3. Deployment Guide
```markdown
# validation_experiment/DEPLOYMENT_GUIDE.md

- Step-by-step migration
- Expected impact
- Rollback plan
```

### 4. Raw Data
```
validation_experiment/
├── baseline_metrics.json
├── candidate1_*/
│   ├── config.yaml
│   ├── validation_run.log
│   └── metrics.json
├── candidate2_*/
└── comparison_summary.json
```

## Iteration Logic

The validator may iterate (max 2 times) if:
- ✅ Some improvement found (worth exploring further)
- ✅ Clear failure pattern (can fix, e.g., all DP=1 → adjust to DP≥2)
- ❌ Target not met yet

Stops iterating if:
- ✅ Target met (success)
- ❌ Max iterations reached (2)
- ❌ No improvement possible (baseline optimal)

## Best Practices

1. **Run during off-peak hours** (occupies devices for 20-35 min)
2. **Start with baseline** (always establish reference point)
3. **Trust the heuristics** (learned from real experiments)
4. **Validate before scaling** (cheaper to test on 8 devices than 64)
5. **Document findings** (add to project memory for future reference)

## Safety

- ✅ Read-only on model checkpoints
- ✅ Writes only to `validation_experiment/`
- ✅ No production impact (separate validation runs)
- ✅ Auto-abort on critical errors (OOM, hang)
- ✅ Max 2 iteration cycles (prevents infinite loops)

## Cost Estimation

**Per candidate**: 50 steps × training cost
**Typical session**: 3-5 candidates
**Total**: ~150-250 training steps equivalent

**ROI**: If finding 2x improvement, pays for itself in ~200 production steps

## Troubleshooting

### "No baseline found"
```
→ Point to existing training log
→ Or run 50 steps of baseline first
```

### "All candidates failed"
```
→ Check pre-flight (devices available, no OOM)
→ Review error patterns
→ May need manual intervention
```

### "Winner worse than baseline"
```
→ Baseline may already be optimal
→ Try different optimization axis (e.g., batch size, learning rate)
```

## References

- **Real experiment**: `validation_experiment/FINAL_REPORT_CANDIDATE4_WINS.md`
- **Workflows**: `.agent/skills/strategy-validator/workflows/`
- **Heuristics source**: Qwen3-30B-A3B validation (Oct 2026)

---

**Next Steps**: Invoke `/validate-strategy` to start optimization!
