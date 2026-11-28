# /validate-strategy — Validate Parallel Strategies

Validate parallel strategy recommendations through actual distributed training.

Execute the `strategy-validator` skill to validate PSA recommendations or custom parallel strategies.

## Trigger

User types `/validate-strategy` followed by:
- PSA output (JSON or natural language summary)
- Or reference to PSA analysis already in conversation
- Or manual strategy specification

## Action

Invoke the `strategy-validator` skill with appropriate arguments.

## Examples

```bash
# Validate PSA recommendation from conversation
/validate-strategy Use the dp=4,tp=8,pp=2 strategy recommended above

# Validate with explicit PSA output
/validate-strategy {
  "strategy": {"dp": 4, "tp": 8, "pp": 2, "fsdp_level": "level2"},
  "model": {"name": "Qwen3-30B-A3B", "params": 30e9, "layers": 60},
  "hardware": {"device_type": "Ascend 910B", "device_count": 64},
  "estimates": {"memory_per_device_gb": 58.0, "throughput_tokens_per_sec": 125000}
}

# Validate multiple candidates (Top-3)
/validate-strategy Compare top 3 candidates from PSA analysis

# Quick validation (fast recipe, 20 steps)
/validate-strategy Quick check dp=8,tp=8,pp=1 on 64 NPUs
```

## Workflow

1. Parse input (PSA output or strategy specification)
2. Generate training configuration (Phase 1)
3. Launch distributed benchmark (Phase 2)
4. Extract metrics (Phase 3)
5. Generate comparison report (Phase 4)
6. Output validation result with recommendation

## Prerequisites

- Target hardware available (check with `npu-smi info` or `nvidia-smi`)
- Model checkpoint accessible
- 5-15 minutes per validation run

## Output

Validation report with:
- ✓/✗ assessment (metrics within tolerance)
- Actual vs PSA estimate comparison table
- Baseline improvement metrics
- Production deployment recommendation
