---
name: routing-validator
description: >
  Validate MoE routing correctness: token integrity (conservation),
  routing correctness (expert-major order), gradient flow (all experts receive gradients).
  Pure correctness validation - no performance metrics.
---

Executable evidence contract: [supported inputs and missing evidence](references/evidence-contract.md).
Missing evidence is null/UNVERIFIED; extraction completion does not imply correctness PASS.


# Routing Validator

**Correctness-only validation tool** for MoE routing mechanisms.

## When to use

- After MoE training run to verify correctness
- During optimization to ensure changes don't break correctness
- Debugging MoE routing issues
- Required by `moe-optimizer` for candidate validation
- **Not for**: Performance analysis (use `moe-profiler` instead)

## Trigger patterns

- "Validate MoE routing correctness"
- "Check MoE gradient flow"
- "MoE correctness validation"
- "validate MoE routing"

## Prerequisites

At least one of:
- **Checkpoint** with MoE module state (for gradient flow check)
- **Training logs** with MoE metrics (for token integrity check)
- Ideally both for complete validation

## Three Core Checks

### Check 1: Token Integrity (Conservation)

**Invariant:** Input token count == Output token count

```python
# Every input token produces exactly one output token
assert input_tokens.shape[0] == output_tokens.shape[0]

# All tokens assigned to some expert
assert sum(tokens_per_expert) == total_tokens

# No tokens duplicated (top_k already handled)
# Each token appears in exactly top_k experts
```

**What it catches:**
- Token drops during routing
- Incorrect All-to-All sizes
- Buffer overflow/underflow

---

### Check 2: Routing Correctness (Expert-Major Order)

**Invariant:** After All-to-All dispatch, tokens are in expert-major order

```python
# Tokens should be grouped by expert
# Expert 0's tokens, then Expert 1's tokens, etc.
for expert_id in range(num_experts):
    start_idx = sum(tokens_per_expert[:expert_id])
    end_idx = start_idx + tokens_per_expert[expert_id]

    # All tokens in [start_idx:end_idx] should belong to expert_id
    assert all(expert_assignments[start_idx:end_idx] == expert_id)
```

**What it catches:**
- Incorrect permutation indices
- Wrong All-to-All split sizes
- Expert assignment bugs

---

### Check 3: Gradient Flow (All Experts)

**Invariant:** All activated experts receive gradients

```python
for expert_id in range(num_experts):
    if expert_was_activated(expert_id):
        expert_params = get_expert_params(expert_id)

        for param in expert_params:
            if param.requires_grad:
                # Must have gradient if expert was used
                assert param.grad is not None
                # Gradient should be non-trivial
                assert param.grad.abs().sum() > 0
```

**What it catches:**
- Gradient backward bugs
- Disconnected computational graph
- Dead experts (no gradient updates)

---

## Command

```bash
python <skill-dir>/scripts/validate_routing.py \
  --checkpoint-dir <checkpoint> \
  --log-dir <training-logs> \
  --output-dir <validation-results>
```

**Arguments:**
- `--checkpoint-dir`: Model checkpoint directory (for gradient checks)
- `--log-dir`: Training logs (for token integrity checks)
- `--output-dir`: Where to write validation report (default: `./routing_validation`)
- `--strict`: Fail on warnings (default: false)

**Minimal example:**
```bash
python scripts/validate_routing.py \
  --checkpoint-dir /path/to/checkpoint \
  --output-dir ./validation_results
```

## Execution Flow

```text
Phase 1: Input Discovery
  - Locate checkpoint files (*.pt)
  - Locate training logs (*.log)
  - Verify at least one source available

Phase 2: Token Integrity Check
  - Parse logs for tokens_per_expert
  - Check sum equals total tokens
  - Verify no missing assignments
  Output: PASS / FAIL + details

Phase 3: Routing Correctness Check
  - Load routing metadata from checkpoint
  - Verify expert-major ordering
  - Check permutation indices consistency
  Output: PASS / FAIL + details

Phase 4: Gradient Flow Check
  - Load model state_dict from checkpoint
  - Find all expert parameters
  - Verify gradient presence and non-zero
  Output: PASS / FAIL + details

Phase 5: Generate Report
  - Aggregate all check results
  - Generate JSON report + markdown summary
  - Exit code: 0 if all pass, 1 if any fail
```

## Output Format

### JSON Report

```json
{
  "validation_id": "routing_val_20240120_143022",
  "timestamp": "2024-01-20T14:30:22",
  "input_sources": {
    "checkpoint_dir": "/path/to/checkpoint",
    "log_dir": "/path/to/logs"
  },
  "checks": {
    "token_integrity": {
      "status": "PASS",
      "details": {
        "total_tokens": 4096,
        "tokens_assigned": 4096,
        "tokens_dropped": 0,
        "conservation_ratio": 1.0
      }
    },
    "routing_correctness": {
      "status": "PASS",
      "details": {
        "layers_checked": 4,
        "expert_major_order": true,
        "permutation_valid": true
      }
    },
    "gradient_flow": {
      "status": "PASS",
      "details": {
        "experts_checked": 128,
        "experts_with_gradients": 128,
        "dead_experts": [],
        "zero_gradients": []
      },
      "warnings": []
    }
  },
  "overall_status": "PASS",
  "summary": "All 3 checks passed"
}
```

### Status Values

- **PASS**: Check passed, no issues
- **WARN**: Check passed with warnings (non-critical)
- **FAIL**: Check failed, critical issue
- **SKIP**: Check skipped (missing required data)

### Markdown Summary

```markdown
# MoE Routing Validation Report

**Status:** ✅ PASS
**Validation ID:** routing_val_20240120_143022
**Timestamp:** 2024-01-20 14:30:22

## Check Results

### ✅ Token Integrity: PASS
- Total tokens: 4096
- Tokens assigned: 4096
- Conservation ratio: 1.0

### ✅ Routing Correctness: PASS
- Layers checked: 4
- Expert-major order: Valid
- Permutation indices: Valid

### ✅ Gradient Flow: PASS
- Experts checked: 128
- Experts with gradients: 128
- Dead experts: None

## Summary
All correctness checks passed. MoE routing is functioning correctly.
```

## Integration with moe-optimizer

Used in Phase 4 (Validation) of moe-optimizer:

```python
# moe-optimizer calls routing-validator
validation_result = invoke_routing_validator(
    checkpoint_dir="./validation_candidate_1/checkpoint",
    log_dir="./validation_candidate_1/logs"
)

if validation_result["overall_status"] != "PASS":
    # Candidate fails correctness check
    candidate_result["correctness"]["pass"] = False
    candidate_result["failure_reasons"].append(
        f"Correctness failed: {validation_result['summary']}"
    )
```

## References (on-demand)

Load when interpreting results or debugging:

- [references/correctness-invariants.md](references/correctness-invariants.md) — Detailed invariants
- [references/gradient-flow-patterns.md](references/gradient-flow-patterns.md) — Expected gradient patterns

## Failure Examples

### Example 1: Token Drop

```json
{
  "token_integrity": {
    "status": "FAIL",
    "details": {
      "total_tokens": 4096,
      "tokens_assigned": 3987,
      "tokens_dropped": 109,
      "conservation_ratio": 0.973
    },
    "error": "109 tokens dropped during routing (2.66%)"
  }
}
```

**Cause:** Capacity overflow not handled correctly

---

### Example 2: Gradient Missing

```json
{
  "gradient_flow": {
    "status": "FAIL",
    "details": {
      "experts_checked": 128,
      "experts_with_gradients": 120,
      "dead_experts": [23, 47, 56, 89, 91, 103, 117, 125],
      "zero_gradients": []
    },
    "error": "8 experts missing gradients"
  }
}
```

**Cause:** Experts never activated OR backward bug

---

### Example 3: Expert Order Wrong

```json
{
  "routing_correctness": {
    "status": "FAIL",
    "details": {
      "layers_checked": 4,
      "expert_major_order": false,
      "permutation_valid": false,
      "error_layer": 2
    },
    "error": "Layer 2 tokens not in expert-major order"
  }
}
```

**Cause:** Incorrect permutation indices

---

## Debugging Guide

### Issue: All checks SKIP

**Cause:** No valid checkpoint or logs found

**Solution:**
```bash
# Check inputs exist
ls -la /path/to/checkpoint/*.pt
ls -la /path/to/logs/*.log

# Verify checkpoint has MoE modules
python -c "import torch; print(torch.load('checkpoint.pt').keys())"
```

---

### Issue: Token integrity FAIL

**Cause:** Capacity overflow or routing bug

**Solution:**
1. Check capacity_factor setting
2. Verify All-to-All split sizes
3. Enable detailed logging (HP_MOE_PROFILING=1)

---

### Issue: Gradient flow FAIL (dead experts)

**Cause:** Experts not activated OR backward bug

**Solution:**
1. Check if dead experts have 0 tokens assigned
   - If yes: routing issue (load imbalance)
   - If no: backward bug

2. Enable gradient checkpointing carefully (can break gradients)

3. Verify expert_bias not blocking gradients

---

## Limitations

- **Checkpoint requirement**: Gradient checks need checkpoint (can't validate from logs alone)
- **Post-hoc only**: Validates after training, not during
- **No performance metrics**: Only correctness (use moe-profiler for performance)
- **Limited to PyTorch**: Assumes PyTorch checkpoint format

## Safety

- **Read-only**: Never modifies checkpoint or logs
- **Idempotent**: Can run multiple times on same inputs
- **Fast**: ~10-30 seconds for typical validation
- **No GPU needed**: Loads checkpoint to CPU

## Exit Codes

- `0`: All checks passed
- `1`: One or more checks failed
- `2`: Missing required inputs
- `3`: Checkpoint loading error

---

**Next Steps:**
1. Load checkpoint with `torch.load()`
2. Run three core checks
3. Generate validation report
4. Return status to caller (moe-optimizer)
