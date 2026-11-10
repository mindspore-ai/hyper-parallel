# MoE-Specific Metrics for Strategy Validation

**Extension to standard validation metrics** for Mixture-of-Experts models.

Load this reference when validating MoE model training strategies.

---

## Overview

Standard validation metrics (throughput, memory, MFU) are insufficient for MoE models. Additional MoE-specific metrics are required to diagnose:
- Expert load imbalance
- All-to-All communication overhead
- Routing efficiency
- Straggler issues

---

## Metric Collection Points

### 1. During Training Loop

Insert hooks into MoE modules to capture per-step metrics:

```python
from hyper_parallel.core.expert_parallel import MoE

def setup_moe_hooks(model):
    """Setup hooks to collect MoE metrics during training"""
    moe_metrics = {
        "tokens_per_expert": [],
        "expert_load_std": [],
        "routing_entropy": [],
        "dropped_tokens": [],
    }

    def moe_forward_hook(module, input, output):
        if hasattr(module, 'tokens_per_expert'):
            # Capture token distribution
            tokens = module.tokens_per_expert.cpu().numpy()
            moe_metrics["tokens_per_expert"].append(tokens)

            # Compute load std
            load_std = compute_load_std(tokens)
            moe_metrics["expert_load_std"].append(load_std)

            # Capture routing probs if available
            if hasattr(module, 'routing_probs'):
                entropy = compute_routing_entropy(module.routing_probs)
                moe_metrics["routing_entropy"].append(entropy)

            # Count dropped tokens (if capacity exceeded)
            if hasattr(module, 'dropped_tokens_count'):
                moe_metrics["dropped_tokens"].append(
                    module.dropped_tokens_count.item()
                )

    # Register hooks for all MoE layers
    for name, module in model.named_modules():
        if isinstance(module, MoE):
            module.register_forward_hook(moe_forward_hook)

    return moe_metrics
```

### 2. During All-to-All Communication

Measure communication overhead:

```python
import torch.distributed as dist
import time

def measure_alltoall_overhead(moe_module, step_start_time):
    """Measure All-to-All communication time"""

    # Hook into All-to-All calls
    alltoall_times = []

    original_alltoall = dist.all_to_all_single

    def timed_alltoall(*args, **kwargs):
        start = time.time()
        result = original_alltoall(*args, **kwargs)
        elapsed = time.time() - start
        alltoall_times.append(elapsed * 1000)  # Convert to ms
        return result

    # Monkey-patch temporarily
    dist.all_to_all_single = timed_alltoall

    # ... training step ...

    # Restore original
    dist.all_to_all_single = original_alltoall

    # Compute overhead ratio
    total_step_time = (time.time() - step_start_time) * 1000
    total_alltoall_time = sum(alltoall_times)
    overhead_ratio = total_alltoall_time / total_step_time

    return {
        "alltoall_times": alltoall_times,
        "total_alltoall_time": total_alltoall_time,
        "alltoall_overhead_ratio": overhead_ratio
    }
```

---

## Metric Definitions

### 1. Expert Load Imbalance

**expert_load_std**: Coefficient of variation of expert loads

```python
def compute_load_std(tokens_per_expert):
    """Compute normalized standard deviation of expert loads"""
    mean_load = np.mean(tokens_per_expert)
    std_load = np.std(tokens_per_expert)

    # Coefficient of variation (CV)
    cv = std_load / mean_load if mean_load > 0 else 0.0

    return cv

# Interpretation:
# CV < 0.15: Well balanced
# CV 0.15 - 0.30: Moderate imbalance
# CV > 0.30: Severe imbalance
```

**load_imbalance_factor**: Max load / average load

```python
def compute_load_imbalance_factor(tokens_per_expert):
    """Compute imbalance factor"""
    max_load = np.max(tokens_per_expert)
    avg_load = np.mean(tokens_per_expert)

    factor = max_load / avg_load if avg_load > 0 else 1.0

    return factor

# Interpretation:
# 1.0 - 1.2: Excellent balance
# 1.2 - 1.5: Good balance
# 1.5 - 2.0: Moderate imbalance
# > 2.0: Severe imbalance
```

---

### 2. All-to-All Communication Overhead

**alltoall_overhead_ratio**: Fraction of step time spent in All-to-All

```python
def compute_alltoall_overhead(alltoall_times, total_step_time):
    """Compute All-to-All overhead as fraction of step time"""
    total_alltoall = sum(alltoall_times)
    ratio = total_alltoall / total_step_time

    return ratio

# Interpretation:
# < 0.15: Low overhead (compute-bound)
# 0.15 - 0.30: Moderate overhead
# 0.30 - 0.50: High overhead (communication-bound)
# > 0.50: Critical overhead
```

**alltoall_efficiency**: Effective bandwidth / theoretical bandwidth

```python
def compute_alltoall_efficiency(message_size_bytes, time_ms, theoretical_bw_gbps):
    """Compute All-to-All efficiency"""
    # Actual bandwidth achieved
    actual_bw_gbps = (message_size_bytes / (1024**3)) / (time_ms / 1000)

    # Efficiency
    efficiency = actual_bw_gbps / theoretical_bw_gbps

    return efficiency

# Interpretation:
# > 0.7: Excellent
# 0.5 - 0.7: Good
# 0.3 - 0.5: Fair
# < 0.3: Poor (network congestion or overhead)
```

---

### 3. Routing Efficiency

**routing_entropy**: Entropy of expert utilization distribution

```python
def compute_routing_entropy(expert_probabilities):
    """Compute entropy of routing distribution"""
    # expert_probabilities: [num_experts]
    # Sum of routing probabilities across all tokens

    # Normalize to get distribution
    probs = expert_probabilities / np.sum(expert_probabilities)

    # Compute entropy: -Σ p_i log(p_i)
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(probs))
    normalized_entropy = entropy / max_entropy

    return normalized_entropy

# Interpretation:
# > 0.9: Excellent utilization (uniform)
# 0.7 - 0.9: Good utilization
# 0.5 - 0.7: Moderate skew
# < 0.5: Severe skew (many dead experts)
```

**expert_coverage**: Fraction of experts activated

```python
def compute_expert_coverage(tokens_per_expert):
    """Compute fraction of experts that received tokens"""
    num_active = np.sum(tokens_per_expert > 0)
    num_total = len(tokens_per_expert)

    coverage = num_active / num_total

    return coverage

# Interpretation:
# 1.0: All experts active
# 0.8 - 1.0: Healthy coverage
# 0.5 - 0.8: Some dead experts
# < 0.5: Many dead experts (over-capacity model)
```

---

### 4. Straggler Detection

**straggler_factor**: Max rank time / average rank time

```python
def compute_straggler_factor(rank_times):
    """Compute straggler factor across ranks"""
    max_time = np.max(rank_times)
    avg_time = np.mean(rank_times)

    factor = max_time / avg_time

    return factor

# Interpretation:
# < 1.1: No stragglers
# 1.1 - 1.2: Mild stragglers
# 1.2 - 1.5: Moderate stragglers
# > 1.5: Severe stragglers
```

**rank_time_cv**: Coefficient of variation of rank times

```python
def compute_rank_time_cv(rank_times):
    """Compute CV of rank execution times"""
    std_time = np.std(rank_times)
    mean_time = np.mean(rank_times)

    cv = std_time / mean_time if mean_time > 0 else 0.0

    return cv

# Interpretation:
# < 0.05: Excellent balance
# 0.05 - 0.10: Good balance
# 0.10 - 0.20: Moderate variance
# > 0.20: High variance (stragglers present)
```

---

### 5. Dropped Tokens

**dropped_token_ratio**: Fraction of tokens dropped due to capacity

```python
def compute_dropped_token_ratio(dropped_tokens, total_tokens):
    """Compute fraction of tokens dropped"""
    ratio = dropped_tokens / total_tokens if total_tokens > 0 else 0.0

    return ratio

# Interpretation:
# 0.0: No drops (ideal)
# < 0.01: Negligible drops
# 0.01 - 0.05: Moderate drops
# > 0.05: Excessive drops (increase capacity_factor)
```

---

## Aggregated MoE Metrics

Combine per-step metrics into validation summary:

```python
def aggregate_moe_metrics(moe_metrics_per_step):
    """Aggregate MoE metrics across validation steps"""

    aggregated = {}

    # Load imbalance
    load_stds = [m["expert_load_std"] for m in moe_metrics_per_step]
    aggregated["mean_load_std"] = np.mean(load_stds)
    aggregated["max_load_std"] = np.max(load_stds)

    # Load imbalance factor
    all_tokens = [m["tokens_per_expert"] for m in moe_metrics_per_step]
    imbalance_factors = [
        compute_load_imbalance_factor(tokens) for tokens in all_tokens
    ]
    aggregated["mean_load_imbalance_factor"] = np.mean(imbalance_factors)
    aggregated["max_load_imbalance_factor"] = np.max(imbalance_factors)

    # Communication overhead
    alltoall_ratios = [m["alltoall_overhead_ratio"] for m in moe_metrics_per_step]
    aggregated["mean_alltoall_ratio"] = np.mean(alltoall_ratios)
    aggregated["max_alltoall_ratio"] = np.max(alltoall_ratios)

    # Routing entropy
    if "routing_entropy" in moe_metrics_per_step[0]:
        entropies = [m["routing_entropy"] for m in moe_metrics_per_step]
        aggregated["mean_routing_entropy"] = np.mean(entropies)

    # Expert coverage
    coverages = [
        compute_expert_coverage(m["tokens_per_expert"])
        for m in moe_metrics_per_step
    ]
    aggregated["mean_expert_coverage"] = np.mean(coverages)
    aggregated["min_expert_coverage"] = np.min(coverages)

    # Dropped tokens
    if "dropped_tokens" in moe_metrics_per_step[0]:
        total_dropped = sum(m["dropped_tokens"] for m in moe_metrics_per_step)
        total_tokens = sum(np.sum(m["tokens_per_expert"]) for m in moe_metrics_per_step)
        aggregated["dropped_token_ratio"] = compute_dropped_token_ratio(
            total_dropped, total_tokens
        )

    return aggregated
```

---

## Output Format for Validation Report

Extend standard validation report with MoE section:

```json
{
  "validation_id": "val_20240120_143022",
  "candidate_config": {
    "ep_size": 8,
    "tp_size": 2,
    "dp_size": 4,
    "capacity_factor": 1.5
  },

  "standard_metrics": {
    "throughput": 115000,
    "memory_per_device_gb": 58.3,
    "mfu": 0.42
  },

  "moe_metrics": {
    "mean_load_imbalance_factor": 1.42,
    "max_load_imbalance_factor": 1.68,
    "mean_load_std": 0.24,
    "max_load_std": 0.31,

    "mean_alltoall_ratio": 0.22,
    "max_alltoall_ratio": 0.28,
    "alltoall_efficiency": 0.68,

    "mean_routing_entropy": 0.87,
    "mean_expert_coverage": 0.96,
    "min_expert_coverage": 0.91,

    "dropped_token_ratio": 0.003,

    "straggler_factor": 1.15,
    "rank_time_cv": 0.08
  },

  "interpretation": {
    "load_balance": "Good (factor 1.42)",
    "communication": "Moderate overhead (22%)",
    "routing_efficiency": "Excellent (entropy 0.87)",
    "expert_utilization": "Healthy (96% coverage)",
    "stragglers": "Acceptable (factor 1.15)"
  }
}
```

---

## Integration with strategy-validator

Update validation workflow to collect MoE metrics:

```python
# In validation script
if model_config.get("is_moe", False):
    # Setup MoE hooks
    moe_metrics = setup_moe_hooks(model)

    # Run validation steps
    for step in range(num_validation_steps):
        # ... training step ...

        # Measure communication
        comm_metrics = measure_alltoall_overhead(model, step_start_time)
        moe_metrics.update(comm_metrics)

    # Aggregate metrics
    aggregated_moe = aggregate_moe_metrics(moe_metrics)

    # Add to validation report
    report["moe_metrics"] = aggregated_moe
```

---

## Thresholds for Pass/Fail

Define acceptable ranges for MoE metrics:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Load imbalance factor | < 1.3 | 1.3 - 1.8 | > 1.8 |
| AllToAll ratio | < 0.20 | 0.20 - 0.35 | > 0.35 |
| Routing entropy | > 0.85 | 0.70 - 0.85 | < 0.70 |
| Expert coverage | > 0.95 | 0.85 - 0.95 | < 0.85 |
| Dropped token ratio | < 0.01 | 0.01 - 0.05 | > 0.05 |
| Straggler factor | < 1.15 | 1.15 - 1.30 | > 1.30 |

---

## Example Diagnosis

### Scenario A: High Communication Overhead

```json
{
  "moe_metrics": {
    "mean_alltoall_ratio": 0.38,
    "alltoall_efficiency": 0.52
  },
  "diagnosis": "Communication-bound. Reduce ep_size or improve network."
}
```

### Scenario B: Load Imbalance

```json
{
  "moe_metrics": {
    "mean_load_imbalance_factor": 2.15,
    "mean_load_std": 0.42
  },
  "diagnosis": "Severe load imbalance. Increase capacity_factor or enable expert_bias."
}
```

### Scenario C: Dead Experts

```json
{
  "moe_metrics": {
    "mean_expert_coverage": 0.73,
    "mean_routing_entropy": 0.58
  },
  "diagnosis": "Many dead experts (27%). Model over-capacity or poor routing."
}
```

---

## References

- [moe-profiler/references/moe-metrics.md](../../moe-profiler/references/moe-metrics.md) — Metric definitions
- [parallel-strategy-analyzer/references/moe-strategy-rules.md](../../parallel-strategy-analyzer/references/moe-strategy-rules.md) — Strategy rules
