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

Replace angle-bracket placeholders with the current job's values before use;
numeric fields must become numbers. Device counts refer to logical training
devices. Recompute candidate degrees and estimates for that allocation.

```bash
# Validate PSA recommendation from conversation
/validate-strategy Use the dp=4,tp=8,pp=2 strategy recommended above

# Validate with explicit PSA output
/validate-strategy {
  "strategy": {"dp": "<dp>", "tp": "<tp>", "pp": "<pp>", "fsdp_level": "<fsdp_level>"},
  "model": {"name": "<model_name>", "params": "<parameter_count>", "layers": "<num_layers>"},
  "hardware": {"device_type": "<device_type>", "device_count": "<device_count>", "memory_per_device": "<memory_per_device_gb>"},
  "estimates": {"memory_per_device_gb": "<estimated_memory_gb>", "throughput_tokens_per_sec": "<estimated_tokens_per_sec>"}
}

# Validate multiple candidates (Top-3)
/validate-strategy Compare top 3 candidates from PSA analysis

# Quick validation (fast recipe, 20 steps)
/validate-strategy Quick check <candidate_strategy> on <device_count> <device_type> devices, <memory_per_device_gb> GB each
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
