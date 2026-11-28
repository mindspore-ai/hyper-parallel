# Phase 3: Strategy Generation

**Goal:** Generate Top-3 candidate parallelism configs using parallel-strategy-analyzer with MoE-specific constraints.

## Input from Phase 2

```json
{
  "diagnosis": {
    "primary_problem": "communication_bottleneck",
    "root_cause": "...",
    ...
  },
  "strategy_hints": {
    "max_ep_size": 4,
    "capacity_factor": 1.25,
    ...
  }
}
```

## Step 1: Prepare PSA Input

Translate diagnosis into PSA-compatible constraints:

```python
psa_input = {
  "model": {
    "num_layers": model_config.num_layers,
    "hidden_size": model_config.hidden_size,
    "num_attention_heads": model_config.num_attention_heads,
    "num_experts": model_config.num_experts,
    "expert_hidden_size": model_config.moe_intermediate_size,
    "top_k": model_config.num_experts_per_tok,
    "is_moe": True  # Critical flag
  },
  "hardware": {
    "device_type": hardware.device_type,
    "device_count": hardware.device_count,
    "memory_per_device_gb": hardware.memory_per_device_gb,
    "interconnect": hardware.interconnect  # e.g., "HCCS", "InfiniBand"
  },
  "baseline": {
    "throughput": baseline_metrics.throughput,
    "memory_used_gb": baseline_metrics.memory_per_device_gb
  },
  "constraints": {
    # From Phase 2 diagnosis
    "max_ep_size": strategy_hints.get("max_ep_size", 16),
    "min_tp_size": strategy_hints.get("min_tp_size", 1),
    "enable_fsdp": strategy_hints.get("enable_fsdp", False),
    "enable_activation_checkpoint": False,  # Keep disabled for fair comparison

    # Memory constraints
    "max_memory_per_device_gb": hardware.memory_per_device_gb * 0.95,

    # Objective
    "optimization_target": "throughput"  # or "memory", "balanced"
  },
  "moe_specific": {
    "capacity_factor": strategy_hints.get("capacity_factor", 1.0),
    "expert_bias": strategy_hints.get("expert_bias", False),
    "prefer_tp_over_ep": strategy_hints.get("prefer_tp_over_ep", False)
  }
}
```

## Step 2: Invoke parallel-strategy-analyzer

```yaml
skill: parallel-strategy-analyzer
inputs: ${psa_input}
output: strategy_candidates.json
```

**Expected output:** Analysis with Top-N candidates

## Step 3: Parse PSA Output

```python
with open('strategy_candidates.json') as f:
    psa_output = json.load(f)

candidates = psa_output['recommendations'][:3]  # Top-3
```

**Example candidates:**
```json
[
  {
    "rank": 1,
    "config": {
      "tp_size": 4,
      "ep_size": 4,
      "dp_size": 4,
      "pp_size": 1
    },
    "estimated_performance": {
      "throughput": 112000,
      "memory_per_device_gb": 58.2,
      "mfu": 0.42
    },
    "communication_analysis": {
      "alltoall_ratio_estimated": 0.22,
      "allreduce_ratio": 0.08
    }
  },
  ...
]
```

## Step 4: Apply MoE-Specific Adjustments

PSA may not fully account for MoE nuances. Adjust candidates:

### Adjustment 1: Capacity Factor

```python
for candidate in candidates:
    # Add capacity_factor if load imbalance detected
    if "load_imbalance" in problems:
        candidate["config"]["capacity_factor"] = strategy_hints["capacity_factor"]
    else:
        candidate["config"]["capacity_factor"] = 1.0
```

### Adjustment 2: Expert Bias

```python
for candidate in candidates:
    # Enable expert_bias if load imbalance is moderate/high
    if load_imbalance_factor > 1.5:
        candidate["config"]["expert_bias"] = True
    else:
        candidate["config"]["expert_bias"] = False
```

### Adjustment 3: EP+TP Balance

If PSA suggests large EP with small TP:
```python
for candidate in candidates:
    ep_size = candidate["config"]["ep_size"]
    tp_size = candidate["config"]["tp_size"]

    # If comm bottleneck + large EP, reduce EP and increase TP
    if "communication_bottleneck" in problems and ep_size >= 8 and tp_size < 4:
        # Rebalance: halve EP, double TP (keeping product same)
        candidate["config"]["ep_size"] = ep_size // 2
        candidate["config"]["tp_size"] = tp_size * 2
        candidate["adjusted"] = "ep_tp_rebalance"
```

### Adjustment 4: Validate Feasibility

```python
for candidate in candidates:
    config = candidate["config"]

    # Check: tp * ep * dp * pp == device_count
    total = config["tp_size"] * config["ep_size"] * config["dp_size"] * config.get("pp_size", 1)
    assert total == hardware.device_count, f"Invalid config: {total} != {hardware.device_count}"

    # Check: ep_size <= num_experts (can't have more EP ranks than experts)
    assert config["ep_size"] <= model_config.num_experts, "ep_size exceeds num_experts"

    # Check: num_experts % ep_size == 0 (even distribution)
    if model_config.num_experts % config["ep_size"] != 0:
        # Adjust ep_size to nearest divisor
        config["ep_size"] = find_nearest_divisor(model_config.num_experts, config["ep_size"])
```

## Step 5: Re-estimate Performance with MoE Adjustments

PSA estimates may not account for MoE-specific overheads. Adjust estimates:

```python
def adjust_moe_estimates(candidate, baseline_metrics):
    config = candidate["config"]

    # All-to-All overhead scales with EP size
    ep_size = config["ep_size"]
    baseline_ep_size = baseline_config.get("ep_size", 8)

    # Approximate: AllToAll time ∝ ep_size × log(ep_size)
    alltoall_scaling = (ep_size * np.log2(ep_size)) / (baseline_ep_size * np.log2(baseline_ep_size))

    baseline_alltoall_time = baseline_metrics.step_time_ms * baseline_metrics.alltoall_ratio
    estimated_alltoall_time = baseline_alltoall_time * alltoall_scaling

    # Adjust throughput estimate
    psa_throughput = candidate["estimated_performance"]["throughput"]

    # If capacity_factor > 1.0, some tokens may be dropped → lower throughput
    capacity_penalty = 1.0 - (config.get("capacity_factor", 1.0) - 1.0) * 0.03  # 3% per 0.1 capacity

    adjusted_throughput = psa_throughput * capacity_penalty

    candidate["estimated_performance"]["throughput_adjusted"] = adjusted_throughput
    candidate["estimated_performance"]["alltoall_ratio_adjusted"] = (
        estimated_alltoall_time / (baseline_metrics.step_time_ms * (baseline_metrics.throughput / psa_throughput))
    )

    return candidate
```

## Step 6: Rank Candidates

Re-rank candidates based on adjusted estimates:

```python
def rank_candidates(candidates, user_target):
    for candidate in candidates:
        config = candidate["config"]
        perf = candidate["estimated_performance"]

        # Score = weighted combination of objectives
        throughput_score = perf["throughput_adjusted"] / user_target  # > 1.0 is good
        memory_score = 1.0 - (perf["memory_per_device_gb"] / hardware.memory_per_device_gb)
        comm_score = 1.0 - perf["alltoall_ratio_adjusted"]  # lower is better

        # Weights
        candidate["composite_score"] = (
            0.6 * throughput_score +
            0.2 * memory_score +
            0.2 * comm_score
        )

    # Sort by composite score
    candidates.sort(key=lambda c: c["composite_score"], reverse=True)

    return candidates[:3]  # Keep Top-3
```

## Step 7: Generate DeviceMesh Code

For each candidate, generate PyTorch DeviceMesh setup code:

```python
def generate_device_mesh_code(config):
    tp_size = config["tp_size"]
    ep_size = config["ep_size"]
    dp_size = config["dp_size"]

    code = f'''
import torch
from torch.distributed.device_mesh import init_device_mesh

# Initialize 2D mesh for MoE: [DP, TP, EP]
device_mesh = init_device_mesh(
    "npu",  # or "cuda"
    mesh_shape=({dp_size}, {tp_size}, {ep_size}),
    mesh_dim_names=("dp", "tp", "ep")
)

# Extract sub-meshes
dp_mesh = device_mesh["dp"]
tp_mesh = device_mesh["tp"]
ep_mesh = device_mesh["ep"]

# MoE-specific config
moe_config = {{
    "capacity_factor": {config["capacity_factor"]},
    "expert_bias": {config["expert_bias"]},
}}
'''
    return code
```

## Output

```json
{
  "phase": "strategy_generation",
  "status": "complete",
  "candidates": [
    {
      "rank": 1,
      "config": {
        "tp_size": 2,
        "ep_size": 4,
        "dp_size": 8,
        "capacity_factor": 1.25,
        "expert_bias": true
      },
      "estimated_performance": {
        "throughput": 119000,
        "throughput_adjusted": 115000,
        "memory_per_device_gb": 57.8,
        "alltoall_ratio_adjusted": 0.19
      },
      "composite_score": 0.92,
      "device_mesh_code": "..."
    },
    {
      "rank": 2,
      "config": {
        "tp_size": 4,
        "ep_size": 4,
        "dp_size": 4,
        "capacity_factor": 1.25,
        "expert_bias": true
      },
      "estimated_performance": {
        "throughput": 112000,
        "throughput_adjusted": 108000,
        "memory_per_device_gb": 58.3,
        "alltoall_ratio_adjusted": 0.22
      },
      "composite_score": 0.87
    },
    {
      "rank": 3,
      "config": {
        "tp_size": 4,
        "ep_size": 2,
        "dp_size": 8,
        "capacity_factor": 1.0,
        "expert_bias": false
      },
      "estimated_performance": {
        "throughput": 105000,
        "memory_per_device_gb": 59.1,
        "alltoall_ratio_adjusted": 0.15
      },
      "composite_score": 0.81
    }
  ],
  "generation_method": "PSA + MoE adjustments",
  "constraints_applied": {
    "max_ep_size": 4,
    "capacity_factor": 1.25,
    "expert_bias": true
  }
}
```

---

**Next Phase:** [04-validation.md](04-validation.md)
