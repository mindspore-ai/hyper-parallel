# Routing Validator

**MoE routing correctness validation tool.**

## Overview

The `routing-validator` skill validates MoE routing correctness through three core checks:
1. **Token Integrity**: Tokens conserved through routing
2. **Routing Correctness**: Expert-major ordering maintained
3. **Gradient Flow**: All activated experts receive gradients

**Pure correctness validation** — no performance metrics (use `moe-profiler` for that).

## Quick Start

```bash
python scripts/validate_routing.py \
  --checkpoint-dir /path/to/checkpoint \
  --log-dir /path/to/logs \
  --output-dir ./validation
```

## Three Core Checks

### 1. Token Integrity ✓

**Validates:** Input token count == Output token count

```python
assert input_tokens.shape[0] == output_tokens.shape[0]
assert sum(tokens_per_expert) == total_tokens
```

**Catches:** Token drops, incorrect All-to-All, buffer overflow

---

### 2. Routing Correctness ✓

**Validates:** Tokens in expert-major order after All-to-All

```python
# Tokens grouped by expert: [Expert 0][Expert 1]...[Expert N]
for expert_id in range(num_experts):
    assert all(expert_assignments[range] == expert_id)
```

**Catches:** Permutation bugs, wrong split sizes

---

### 3. Gradient Flow ✓

**Validates:** All activated experts receive gradients

```python
for expert_id, tokens in enumerate(tokens_per_expert):
    if tokens > 0:
        assert expert.parameters.grad is not None
```

**Catches:** Backward bugs, disconnected graph, dead experts

---

## Output

### JSON Report

```json
{
  "overall_status": "PASS",
  "checks": {
    "token_integrity": {"status": "PASS", "details": {...}},
    "routing_correctness": {"status": "PASS", "details": {...}},
    "gradient_flow": {"status": "PASS", "details": {...}},
    "expert_coverage": {"status": "WARN", "warning": "2 dead experts"}
  }
}
```

### Status Values

- **PASS**: Check passed ✓
- **WARN**: Passed with warnings ⚠️
- **FAIL**: Critical issue ✗
- **SKIP**: Missing required data ⏭️

## Integration

### With moe-optimizer

Used in Phase 4 (Validation) for candidate correctness checks:

```python
# moe-optimizer automatically calls routing-validator
validation = routing_validator.validate(checkpoint, logs)

if validation["overall_status"] != "PASS":
    candidate["correctness_pass"] = False
```

### Standalone Usage

```bash
# After training run
python scripts/validate_routing.py \
  --checkpoint-dir ./checkpoint-1000 \
  --log-dir ./logs \
  --output-dir ./validation_results
```

## Documentation

- [references/correctness-invariants.md](references/correctness-invariants.md) — 8 formal invariants
- [references/gradient-flow-patterns.md](references/gradient-flow-patterns.md) — 7 gradient patterns

## Common Issues

### Issue: All checks SKIP

**Cause:** No checkpoint or logs found

**Solution:** Verify paths exist and contain `.pt` / `.log` files

---

### Issue: Token integrity FAIL

**Cause:** Tokens dropped during routing

**Solution:** Increase `capacity_factor` or check All-to-All implementation

---

### Issue: Gradient flow FAIL

**Cause:** Experts activated but no gradients

**Solution:** Check backward implementation, avoid detached tensors

---

## Limitations

- **Post-hoc only**: Validates after training (not during)
- **Checkpoint required**: For gradient checks
- **No performance metrics**: Only correctness

## Exit Codes

- `0`: All checks passed
- `1`: One or more checks failed
- `2`: Missing required inputs
- `3`: Checkpoint loading error

## License

Apache 2.0 (same as HyperParallel project)
