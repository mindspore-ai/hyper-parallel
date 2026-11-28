# MoE Metrics Reference

Detailed definitions of all MoE-specific metrics extracted by moe-profiler.

## Load Distribution Metrics

### 1. Expert Load

**Definition:** Number of tokens routed to each expert per forward pass.

```python
tokens_per_expert[layer_id][expert_id] = count(tokens routed to expert)
```

**Properties:**
- Range: [0, capacity × num_tokens]
- Ideal: Uniform distribution across experts
- Reality: Often skewed due to learned routing preferences

**Statistical measures:**
- `mean_tokens`: Average tokens per expert across steps
- `std_tokens`: Standard deviation (indicates variability)
- `min_tokens` / `max_tokens`: Range
- `utilization_ratio`: actual / (total_tokens / num_experts)

**Example:**
```
Layer 0, Expert 3: mean=412, std=23, min=367, max=458, utilization=1.01
→ Slightly above-average utilization, stable across steps
```

---

### 2. Load Imbalance Factor

**Definition:** Ratio of max expert load to mean expert load.

```python
load_imbalance_factor = max(tokens_per_expert) / mean(tokens_per_expert)
```

**Interpretation:**
- 1.0 = Perfect balance
- 1.0 - 1.2 = Good balance
- 1.2 - 1.5 = Moderate imbalance
- 1.5 - 2.0 = High imbalance
- \> 2.0 = Severe imbalance

**Note:** This metric is purely descriptive. moe-profiler does NOT classify as "good" or "severe" — that's for downstream analysis.

---

### 3. Gini Coefficient

**Definition:** Measure of inequality in token distribution (0 = perfect equality, 1 = maximal inequality).

```python
def gini_coefficient(tokens_per_expert):
    n = len(tokens_per_expert)
    sorted_tokens = sorted(tokens_per_expert)
    cumulative = 0
    for i, tokens in enumerate(sorted_tokens):
        cumulative += (2 * (i + 1) - n - 1) * tokens
    return cumulative / (n * sum(tokens_per_expert))
```

**Typical ranges:**
- 0.0 - 0.1: Nearly uniform
- 0.1 - 0.3: Moderate skew
- 0.3 - 0.5: High skew
- \> 0.5: Extreme skew (some experts rarely used)

---

## Communication Metrics

### 4. All-to-All Timing

**Definition:** Time spent in token dispatch and combine All-to-All collectives.

**Dispatch (forward):**
```python
# Exchange tokens from rank-major to expert-major order
alltoall_dispatch_time = time(dist.all_to_all_single(
    input=tokens,  # shape: [local_tokens, hidden_size]
    output_splits=[tokens_per_rank[i] for i in range(ep_size)],
    input_splits=[tokens_from_rank[i] for i in range(ep_size)],
))
```

**Combine (forward/backward):**
```python
# Exchange expert outputs back to original ranks
alltoall_combine_time = time(dist.all_to_all_single(
    input=expert_outputs,
    output_splits=...,  # reverse of dispatch
    input_splits=...,
))
```

**Components:**
- `duration_ms`: Wall-clock time for collective
- `volume_mb`: Data transferred (tokens × hidden_size × dtype_size)
- `bandwidth_gbps`: Effective bandwidth (volume / duration)

**Expected bandwidth:**
- InfiniBand (200 Gbps): 15-20 GB/s effective
- HCCS (Ascend): 10-15 GB/s effective
- NVLink (A100): 20-25 GB/s effective

---

### 5. All-to-All Overhead Ratio

**Definition:** Fraction of total step time spent in All-to-All communication.

```python
alltoall_ratio = (dispatch_time + combine_time) / total_step_time
```

**Typical ranges:**
- < 0.1 (10%): Communication well-amortized
- 0.1 - 0.3: Moderate overhead
- 0.3 - 0.5: Communication-bound
- \> 0.5: Severe communication bottleneck

**Factors affecting ratio:**
- `ep_size`: Larger EP → more communication
- `hidden_size`: Larger hidden → more bytes per token
- `tokens_per_expert`: Load imbalance increases idle time
- `sequence_length`: Longer sequences amortize comm better

---

## Correctness Metrics

### 6. Routing Integrity

**Definition:** Verification that token count is conserved through MoE forward pass.

**Invariant:**
```python
assert input_tokens.shape[0] == output_tokens.shape[0]
# Every input token produces exactly one output token
```

**Check:**
```python
def check_routing_integrity(input_shape, output_shape, routing_plan):
    # 1. Token count conservation
    assert input_shape[0] == output_shape[0]

    # 2. All tokens assigned to some expert
    assert sum(routing_plan.tokens_per_expert) == input_shape[0]

    # 3. No tokens duplicated (top_k already handled)
    # Each token appears in exactly top_k experts
```

**Status values:**
- `PASS`: All checks passed
- `FAIL_COUNT_MISMATCH`: Input/output token count differs
- `FAIL_ASSIGNMENT_GAP`: Some tokens unassigned
- `FAIL_DUPLICATION`: Token appears in > top_k experts

---

### 7. Gradient Flow

**Definition:** Verification that all expert parameters receive gradients (or correctly remain frozen).

**Check:**
```python
def check_gradient_flow(model, expected_sparse=True):
    for name, param in model.named_parameters():
        if 'expert' in name and param.requires_grad:
            if param.grad is None:
                if expected_sparse and not is_expert_activated(name):
                    # OK: expert not used in this micro-batch
                    continue
                else:
                    # ERROR: expert used but no gradient
                    return FAIL
            else:
                # Check gradient is non-trivial
                if torch.all(param.grad == 0):
                    return WARN_ZERO_GRAD
    return PASS
```

**Status values:**
- `PASS`: All activated experts have non-zero gradients
- `WARN_ZERO_GRAD`: Gradient exists but is all zeros (possible dead expert)
- `FAIL_MISSING_GRAD`: Expert activated but no gradient computed

**Metrics:**
- `has_grad`: Boolean (gradient exists)
- `grad_norm`: L2 norm of gradient
- `grad_zero_ratio`: Fraction of gradient elements that are exactly zero

---

### 8. Expert Coverage

**Definition:** Fraction of training steps where each expert receives at least one token.

```python
expert_coverage[expert_id] = count(steps where tokens_per_expert[expert_id] > 0) / total_steps
```

**Interpretation:**
- 1.0: Expert used in every step (ideal for uniform load)
- 0.5 - 1.0: Expert used frequently
- 0.1 - 0.5: Expert used occasionally (possible specialization)
- < 0.1: Expert rarely used (dead or highly specialized)

**Related metric:**
- `zero_assignment_count`: Number of steps where expert received 0 tokens

---

## Straggler Detection Metrics

### 9. Per-Rank Step Time

**Definition:** Wall-clock time for one training step on each rank.

**Measurement:**
```python
step_time[rank] = time(forward + backward + optimizer_step)
```

**Analysis:**
- `mean_step_time_ms`: Average across steps
- `std_ms`: Variability (high std indicates inconsistent performance)
- `min_ms` / `max_ms`: Range

**Ideal:** All ranks have similar mean step time (< 5% variance)

---

### 10. Straggler Score

**Definition:** Ratio of a rank's mean step time to the global mean.

```python
straggler_score[rank] = mean_step_time[rank] / mean(mean_step_time[all_ranks])
```

**Classification:**
- 1.0 ± 0.05: Normal
- 1.05 - 1.15: Slight slowdown
- 1.15 - 1.30: Moderate straggler
- \> 1.30: Severe straggler

**Common causes:**
- Load imbalance: Rank holds hot experts
- Hardware heterogeneity: Slower device
- Thermal throttling: Device overheated
- Memory pressure: Frequent cache evictions

---

## Composite Metrics

### 11. MoE Efficiency Score

**Definition:** Combined metric capturing load balance and communication overhead.

```python
efficiency_score = (1 / load_imbalance_factor) × (1 - alltoall_ratio)
```

**Range:** [0, 1]
- 1.0: Perfect balance, no communication overhead
- 0.5: Moderate efficiency
- < 0.3: Poor efficiency

**Note:** This is a **derived metric** computed in post-processing, not a ground truth.

---

## Extraction Rules

### Log Parsing Patterns

**Expert load:**
```python
# Pattern: tokens_per_expert=tensor([245, 312, 289, ...])
pattern = r'tokens_per_expert=tensor\(\[([\d,\s]+)\]\)'
```

**All-to-All timing:**
```python
# Pattern: [MoE] Layer 0 dispatch: 12.3ms, 4.2MB
pattern = r'\[MoE\] Layer (\d+) (dispatch|combine): ([\d.]+)ms, ([\d.]+)MB'
```

**Step time:**
```python
# Pattern: [Step 10/50] step_time=174.5ms
pattern = r'\[Step \d+/\d+\] step_time=([\d.]+)ms'
```

### Profiler Trace Extraction

**PyTorch Profiler:**
```python
import torch.profiler
trace = torch.profiler.load(profile_dir)
alltoall_events = [e for e in trace.events() if 'all_to_all' in e.name]
```

**Ascend Profiler:**
```python
# Parse ASCEND_PROFILER_OUTPUT/communication.json
import json
with open('communication.json') as f:
    comm_data = json.load(f)
alltoall_ops = [op for op in comm_data['ops'] if op['type'] == 'AllToAll']
```

---

## Metric Limitations

### 1. Sampling Bias
- Profiler overhead may slow down execution (5-10%)
- First few steps excluded (warmup)
- May miss rare events (e.g., infrequent expert activation)

### 2. Aggregation Granularity
- Per-step metrics averaged over time
- May hide micro-batch variance
- Rank-level aggregation may hide device-level issues

### 3. Causality Limitations
- Metrics show correlation, not causation
- High alltoall_ratio correlated with slow steps, but not necessarily causal
- Expert load imbalance may be symptom, not root cause

**Mitigation:** Use moe-profiler for fact extraction, rely on domain experts or moe-optimizer for causal analysis.

---

## Metric Validation

Each metric should satisfy sanity checks:

```python
# Token conservation
assert sum(tokens_per_expert) == total_tokens

# Positive time
assert all(step_time > 0 for step_time in step_times)

# Bounded ratios
assert 0 <= alltoall_ratio <= 1
assert load_imbalance_factor >= 1.0

# Gradient norms
assert all(grad_norm >= 0 for grad_norm in grad_norms)
```

If any check fails, moe-profiler reports `VALIDATION_ERROR` and aborts.
