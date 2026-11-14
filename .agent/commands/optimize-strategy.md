# /optimize-strategy — Automated Parallel Strategy Optimization

End-to-end parallel strategy selection and validation (PSA + actual benchmark).
Orchestrates theoretical analysis with empirical validation in a closed loop
to find the optimal parallelism configuration.

**Workflow**: PSA analysis → Top-3 validation → Adaptive refinement (1-3 rounds)

---

## Overview

The `/optimize-strategy` command triggers the **strategy-optimizer** agent, which:

1. **Collects** user input (model, hardware, target)
2. **Analyzes** with PSA → generates Top-3 candidate strategies
3. **Validates** each candidate via actual 50-step benchmarks
4. **Refines** search space based on validation results
5. **Iterates** until target met or max rounds (3) reached
6. **Outputs** best validated configuration + performance data

**Key advantage**: Combines theoretical guidance (PSA) with empirical validation (actual runs).

---

## When to Use

✅ **Use /optimize-strategy when**:
- New model + new hardware (no prior experience)
- Production deployment (need confidence)
- Complex optimization (multiple objectives)
- Time available (1-3 hours)

❌ **Don't use when**:
- Quick validation needed (<30 min) → use `/validate-strategy`
- Strategy already known → deploy directly
- Only need analysis → use PSA directly

---

## Input Required

### Minimal Input

The agent will prompt for:

1. **Model information**:
   - Model name/path
   - Number of layers
   - Hidden size
   - Is MoE? (yes/no)
   - Number of experts (if MoE)

2. **Hardware information**:
   - Device type (Ascend910B, A100, etc.)
   - Device count (e.g., 8, 16, 64)
   - Memory per device (GB)

3. **Optimization target** (optional):
   - Target throughput (e.g., 1.2x vs baseline, or absolute value)
   - Max memory per device (GB)
   - Min MFU (Model FLOPs Utilization)

### Example Invocation

**Simple**:
```
/optimize-strategy
```

Agent will prompt for details interactively.

**With context**:
```
/optimize-strategy

Model: Qwen3-30B-A3B (60 layers, MoE with 64 experts)
Hardware: 64× Ascend 910B (64GB each)
Target: 1.2x throughput improvement vs baseline
```

---

## Workflow Details

### Phase 1: PSA Analysis (Theory)

```
Agent calls parallel-strategy-analyzer:
  Input: model + hardware + constraints
  Process: Enumerate strategies, estimate performance
  Output: Top-3 candidates (sorted by estimated throughput)

Example output:
  1. TP=8, CP=2, EP=2, PP=4, DP=1 → est. 145K t/s
  2. TP=4, CP=4, EP=2, PP=2, DP=2 → est. 132K t/s
  3. TP=8, CP=1, EP=4, PP=2, DP=1 → est. 128K t/s
```

### Phase 2: Validation (Practice)

```
Agent calls strategy-validator for each candidate:
  1. Generate training config (YAML)
  2. Generate launch script
  3. Run 50-step benchmark on target hardware
  4. Extract metrics (throughput, memory, stability)

Example output:
  Candidate 1: OOM at initialization
  Candidate 2: 102K t/s (below target)
  Candidate 3: OOM at initialization
```

### Phase 3: Decision & Refinement

```
Agent analyzes results:
  - Any candidate meets target?
    YES → Deploy winner ✓
    NO  → Analyze failures

  Failure analysis:
    - OOM → Increase FSDP level, enable activation checkpoint
    - FSDP conflict → Force DP ≥ 2
    - Comm bottleneck → Reduce TP/CP
    - High bubble → Switch PP schedule

  Update constraints, go to next iteration
```

### Phase 4: Iteration (if needed)

```
Max iterations: 3

Iteration 1: Permissive constraints
Iteration 2: Refined constraints (based on I1 failures)
Iteration 3: Final attempt (most conservative/aggressive)

If still no target: Deploy best performer
```

---

## Output

### Success Case

```
╔══════════════════════════════════════════════════════════════╗
║ OPTIMIZATION COMPLETE                                        ║
╚══════════════════════════════════════════════════════════════╝

🏆 Optimal Strategy: TP=4, CP=2, EP=4, DP=2

📊 Performance (Validated):
  Throughput:  118,000 tokens/s
  Memory:      59.8 GB
  Step Time:   687 ms

✅ Target:     ≥ 113,000 tokens/s (ACHIEVED)
✅ Iterations: 2

📄 Files Generated:
  - optimizer_output/production_config.yaml
  - optimizer_output/DEPLOYMENT_GUIDE.md
  - optimizer_output/OPTIMIZATION_REPORT.md
```

### Iteration Example

```
ITERATION 1
═══════════════════════════════════════════════════════════

PSA Top-3:
  1. TP=8, EP=2, DP=1 → est. 145K t/s
  2. TP=4, EP=2, DP=2 → est. 132K t/s
  3. TP=8, EP=4, DP=1 → est. 128K t/s

Validation:
  Candidate 1: ✗ OOM (DP=1 insufficient)
  Candidate 2: ✓ 102K t/s (below target)
  Candidate 3: ✗ OOM (DP=1 insufficient)

Decision: ITERATE (best: 102K < 113K target)

Refinement:
  - Force DP ≥ 2 (avoid FSDP conflicts)
  - Reduce TP (communication bottleneck detected)

═══════════════════════════════════════════════════════════

ITERATION 2
═══════════════════════════════════════════════════════════

PSA Top-3 (with constraints: DP≥2, TP≤4):
  1. TP=4, CP=2, EP=4, DP=2 → est. 128K t/s
  2. TP=2, CP=4, EP=4, DP=2 → est. 121K t/s
  3. TP=4, CP=1, EP=8, DP=2 → est. 115K t/s

Validation:
  Candidate 1: ✓ 118K t/s ⭐ (MEETS TARGET)
  Candidate 2: ✓ 110K t/s
  Candidate 3: ✓ 108K t/s

Decision: DEPLOY (Candidate 1)
```

---

## Time Estimation

**Per Iteration**:
- PSA Analysis: ~1 minute
- Validation (3 candidates): ~21 minutes (3 × 7 min)
- Decision + Refinement: ~1 minute
- **Total**: ~23 minutes

**Full Optimization**:
- 1 iteration (best case): ~23 minutes
- 2 iterations (typical): ~46 minutes
- 3 iterations (worst case): ~69 minutes

---

## Configuration Options

### Default Settings

```python
defaults = {
    "max_iterations": 3,
    "validation_steps": 50,
    "warmup_steps": 10,
    "candidates_per_round": 3,
}
```

### Custom Settings (via agent conversation)

```
User: Use only 2 iterations and 30 validation steps

Agent will adjust:
  max_iterations = 2
  validation_steps = 30
```

---

## Integration Points

### With PSA

```
Agent → PSA:
  model_config = {...}
  hardware_config = {...}
  constraints = {...}  # Updated each iteration

PSA → Agent:
  Top-3 strategies with estimates
```

### With Strategy-Validator

```
Agent → Validator:
  candidate_config = {...}
  validation_steps = 50

Validator → Agent:
  metrics = {throughput, memory, step_time, status}
```

### Output Files

```
optimizer_output/
├── production_config.yaml         # Ready-to-deploy config
├── DEPLOYMENT_GUIDE.md            # Migration steps
├── OPTIMIZATION_REPORT.md         # Complete iteration history
├── iteration_1/
│   ├── candidate1_config.yaml
│   ├── candidate1_metrics.json
│   └── ...
└── iteration_2/
    ├── candidate1_config.yaml
    ├── candidate1_metrics.json
    └── ...
```

---

## Decision Logic

### OOM Handling

```python
if validation_result.status == "OOM":
    # Increase FSDP sharding
    constraints.fsdp_level = min(constraints.fsdp_level + 1, 3)
    constraints.activation_checkpoint = True
    constraints.dp_min = max(constraints.dp_min, 2)
```

### FSDP Conflict

```python
if validation_result.error == "FSDP_CONFLICT":
    # Force DP ≥ 2
    constraints.dp_min = 2
```

### Communication Bottleneck

```python
if actual_time > estimated_time * 2:
    # Reduce communication dimensions
    constraints.tp_max = min(constraints.tp_max, 4)
    constraints.cp_max = min(constraints.cp_max, 2)
```

### High Bubble (PP)

```python
if validation_result.bubble_ratio > 0.15:
    # Switch schedule or reduce PP
    constraints.pp_schedule = "interleaved_1F1B"
    constraints.pp_size = max(constraints.pp_size // 2, 1)
```

---

## Comparison

| Feature | /validate-strategy | /optimize-strategy |
|---------|-------------------|-------------------|
| **Input** | Manual candidates | Auto from PSA |
| **Iterations** | 0-2 | 1-3 |
| **Theory** | No | ✅ PSA |
| **Practice** | ✅ Validation | ✅ Validation |
| **Refinement** | Simple rules | Complex analysis |
| **Time** | 20-35 min | 23-69 min |
| **Use Case** | Known domain | Unknown domain |

---

## Troubleshooting

### "PSA not available"

**Solution**: PSA skill must be implemented first, or use `/validate-strategy` standalone.

### "All candidates OOM"

**Cause**: Model too large for hardware.

**Solutions**:
- Increase device count
- Enable CPU offloading
- Reduce batch size

### "No improvement after 3 iterations"

**Cause**: Baseline may already be optimal, or hardware insufficient.

**Action**: Deploy best performer found, report hardware limits.

### "Validation stuck"

**Cause**: Manual execution required on target node.

**Action**: Follow instructions in workflow output, run validation scripts, then resume.

---

## Safety & Limits

- **Max iterations**: 3 (prevent infinite loop)
- **Validation cost**: ~50 steps × 3 candidates × 3 iterations = 450 steps max
- **Time budget**: 1-3 hours (configurable)
- **Fallback**: Always deploy best performer even if < target
- **Audit trail**: Complete logs for each iteration

---

## Example Session

```
User: /optimize-strategy

Agent: I'll help you find the optimal parallel strategy. Please provide:

1. Model information
   - Model name: Qwen3-30B-A3B
   - Number of layers: 60
   - Is MoE?: Yes
   - Number of experts: 64

2. Hardware
   - Device type: Ascend910B
   - Device count: 64
   - Memory per device (GB): 64

3. Target (optional)
   - Target throughput: 1.2x vs baseline

Agent: Starting optimization...

[ITERATION 1]
PSA Analysis... ✓
  Top-3 candidates generated

Validation... (running)
  Candidate 1: ✗ OOM
  Candidate 2: ✓ 102K t/s (below target)
  Candidate 3: ✗ OOM

Refining constraints... ✓
  Force DP ≥ 2, Reduce TP

[ITERATION 2]
PSA Analysis... ✓
  Top-3 candidates generated (with refined constraints)

Validation... (running)
  Candidate 1: ✓ 118K t/s ⭐ (MEETS TARGET!)
  Candidate 2: ✓ 110K t/s
  Candidate 3: ✓ 108K t/s

Optimization complete!

🏆 Optimal: TP=4, CP=2, EP=4, DP=2
📊 Throughput: 118K tokens/s (1.25x)
📄 Files: optimizer_output/production_config.yaml

Ready to deploy!
```

---

## References

- **Strategy-Optimizer Skill**: `.agent/skills/strategy-optimizer/`
- **Strategy-Validator Skill**: `.agent/skills/strategy-validator/`
- **PSA Skill**: `.agent/skills/parallel-strategy-analyzer/`
- **Config Templates**: `.agent/skills/strategy-validator/references/config-templates.md`
- **Benchmark Suite**: `tests/benchmarks/strategy_validation/`

---

**The /optimize-strategy command provides end-to-end automated strategy optimization with validated results!** 🚀
