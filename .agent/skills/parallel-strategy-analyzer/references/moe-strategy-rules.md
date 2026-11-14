# MoE-Specific Strategy Rules

**Extension to strategy-rules.md** for Mixture-of-Experts (MoE) models.

These rules augment the base parallelism strategy selection with MoE-specific considerations for expert parallelism (EP), capacity management, and All-to-All communication.

---

## Memory Estimation Adjustments

### Base Expert Memory

```python
# Per-rank expert memory (baseline)
expert_params_per_rank = (num_experts / ep_size) * expert_param_size

# Expert activation memory (accounts for load imbalance)
expert_activation_memory = (
    (num_experts / ep_size) *          # Experts per rank
    expert_hidden_size *                # Expert FFN hidden size
    capacity_factor *                   # Overflow buffer
    avg_tokens_per_rank                 # Average token count
)

# Peak can be higher due to load imbalance
peak_expert_activation = expert_activation_memory * (1 + load_imbalance_factor)
```

### All-to-All Communication Buffers

```python
# Bidirectional All-to-All requires 2× buffers
alltoall_buffer_memory = 2 * (
    num_tokens *                        # Tokens in batch
    hidden_size *                       # Model hidden dimension
    dtype_size                          # 2 bytes for fp16/bf16
)

# Per layer, 4× All-to-All (dispatch + combine, forward + backward)
total_alltoall_buffer = 4 * alltoall_buffer_memory
```

### Total MoE Memory

```python
total_moe_memory_per_rank = (
    expert_params_per_rank +            # Expert parameters
    peak_expert_activation +            # Peak activation (with imbalance)
    total_alltoall_buffer +             # Communication buffers
    router_memory                       # Router parameters (usually negligible)
)
```

**Key insight:** Capacity factor and load imbalance directly impact memory. A `capacity_factor=1.5` can increase activation memory by 50%.

---

## Communication Cost

### All-to-All Cost

```python
# Each MoE layer has 4 All-to-All operations per step
# (dispatch forward, combine forward, dispatch backward, combine backward)

single_alltoall_cost = (
    num_tokens *                        # Tokens to exchange
    hidden_size *                       # Hidden dimension
    dtype_size *                        # 2 bytes
    (ep_size - 1) / ep_size *          # Fraction exchanged
    2 / bandwidth                       # Bidirectional exchange
)

# Total per layer
alltoall_cost_per_layer = 4 * single_alltoall_cost

# Scaling with EP size: O(ep_size × log(ep_size))
# Empirical: alltoall_time ∝ ep_size^1.2
effective_alltoall_cost = alltoall_cost_per_layer * (ep_size ** 0.2)
```

**Empirical data:**
```
ep_size=2:  1.2 ms (32 MB message, InfiniBand)
ep_size=4:  3.4 ms
ep_size=8:  8.1 ms
ep_size=16: 18.3 ms
```

### Load Imbalance Penalty

```python
# Imbalance causes stragglers → collective wait time
load_imbalance_factor = max_tokens_per_expert / avg_tokens_per_expert

# Effective step time includes waiting for slowest rank
load_imbalance_penalty = load_imbalance_factor - 1.0
effective_step_time = compute_time * (1 + load_imbalance_penalty)

# Example: 2× imbalance → one rank takes 2× time → 100% slowdown
```

**Typical values:**
- Balanced: 1.0 - 1.2 (0-20% penalty)
- Moderate: 1.2 - 1.5 (20-50% penalty)
- Severe: 1.5 - 2.5 (50-150% penalty)

---

## Strategy Search Rules

### Rule 1: EP Size Selection

**Base heuristic:**
```python
def select_ep_size(num_experts, device_count, interconnect_type):
    # Case 1: Many experts (> 2× devices)
    if num_experts > device_count * 2:
        # Prefer expert sharding
        ep_size = min(device_count, num_experts // 2)

    # Case 2: Few experts (< devices)
    elif num_experts < device_count:
        # Expert replication to reduce communication
        ep_size = max(2, device_count // 4)

    # Case 3: Moderate experts (≈ devices)
    else:
        ep_size = min(8, device_count // 2)

    # Constraint: ep_size must divide num_experts evenly
    ep_size = find_largest_divisor(num_experts, ep_size)

    # Interconnect adjustment
    if interconnect_type == "slow":  # < 50 Gbps
        ep_size = max(2, ep_size // 2)  # Reduce communication

    return ep_size
```

**Example (Qwen3-30B-A3B):**
```
num_experts = 128
device_count = 64
→ ep_size = min(64, 128 // 2) = 64
→ But 64 is large, adjust to 8 for communication
→ Final: ep_size = 8 (128 / 8 = 16 experts per rank)
```

---

### Rule 2: EP + TP Hybrid (2D Mesh)

**When to use:**
```python
def use_ep_tp_hybrid(hidden_size, ep_size, device_count):
    # Condition 1: Large hidden size (benefits from TP)
    if hidden_size < 4096:
        return False, None

    # Condition 2: EP doesn't use all devices
    if ep_size >= device_count:
        return False, None

    # Condition 3: Remaining devices can form TP
    remaining_devices = device_count // ep_size
    if remaining_devices < 2:
        return False, None

    # Use 2D mesh: [ep, tp]
    tp_size = remaining_devices
    return True, tp_size
```

**Device mesh configuration:**
```python
# Option 1: EP + TP (2D)
mesh_shape = (ep_size, tp_size)  # e.g., (8, 8) for 64 devices
mesh_names = ("ep", "tp")

# Option 2: EP + TP + DP (3D)
if has_data_parallelism:
    mesh_shape = (dp_size, ep_size, tp_size)
    mesh_names = ("dp", "ep", "tp")
```

**Example:**
```
64 devices, hidden=4096
→ ep_size=8, tp_size=8, dp_size=1
→ mesh: [8 EP, 8 TP]
→ Each EP group has 8 TP-sharded devices
```

---

### Rule 3: Capacity Factor Selection

**Based on load imbalance:**
```python
def select_capacity_factor(load_imbalance_factor, num_experts, top_k):
    if load_imbalance_factor is None:
        # No profiling data, use safe default
        return 1.25

    # Mild imbalance
    if load_imbalance_factor < 1.3:
        return 1.0  # No overflow needed

    # Moderate imbalance
    elif load_imbalance_factor < 1.6:
        return 1.25  # 25% overflow buffer

    # High imbalance
    elif load_imbalance_factor < 2.0:
        return 1.5   # 50% overflow buffer

    # Severe imbalance
    else:
        return 2.0   # 100% overflow buffer

    # Constraint: capacity_factor × top_k ≤ num_experts / 2
    # (avoid excessive memory)
```

**Memory impact:**
```
capacity_factor=1.0: baseline memory
capacity_factor=1.25: +25% expert activation memory
capacity_factor=1.5: +50% expert activation memory
capacity_factor=2.0: +100% expert activation memory
```

---

### Rule 4: Communication vs Memory Tradeoff

**Decision tree:**
```python
def optimize_ep_for_tradeoff(current_config, constraints):
    memory_pressure = current_config["memory_used"] / constraints["max_memory"]
    comm_overhead = current_config["alltoall_ratio"]

    # High memory pressure → increase EP (reduce expert replication)
    if memory_pressure > 0.85:
        new_ep = min(current_config["ep_size"] * 2, num_experts)
        return {"action": "increase_ep", "new_ep": new_ep}

    # High communication → reduce EP (increase expert replication)
    elif comm_overhead > 0.4:
        new_ep = max(current_config["ep_size"] // 2, 2)
        return {"action": "reduce_ep", "new_ep": new_ep}

    # Balanced
    else:
        return {"action": "keep", "new_ep": current_config["ep_size"]}
```

**Example scenarios:**
```
Scenario A: Memory 55GB/64GB, comm 15%
  → Balanced, keep ep_size=8

Scenario B: Memory 62GB/64GB, comm 12%
  → Memory pressure, increase ep_size to 16

Scenario C: Memory 48GB/64GB, comm 45%
  → Communication bound, reduce ep_size to 4
```

---

## Strategy Generation Algorithm

### MoE-Enhanced Search

```python
def generate_moe_strategies(model_config, hardware_config, constraints):
    candidates = []

    # Base EP sizes to try
    ep_candidates = [2, 4, 8, 16]
    ep_candidates = [e for e in ep_candidates if e <= hardware_config["device_count"]]
    ep_candidates = [e for e in ep_candidates if model_config["num_experts"] % e == 0]

    for ep_size in ep_candidates:
        # For each EP, determine TP
        remaining = hardware_config["device_count"] // ep_size

        if model_config["hidden_size"] >= 4096 and remaining >= 2:
            tp_candidates = [2, 4, 8]
            tp_candidates = [t for t in tp_candidates if t <= remaining]
        else:
            tp_candidates = [1]

        for tp_size in tp_candidates:
            # DP fills the rest
            dp_size = hardware_config["device_count"] // (ep_size * tp_size)

            if dp_size < 1:
                continue

            # Estimate costs
            config = {
                "ep_size": ep_size,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "capacity_factor": select_capacity_factor(
                    constraints.get("load_imbalance_factor"),
                    model_config["num_experts"],
                    model_config["top_k"]
                )
            }

            # Memory check
            memory = estimate_moe_memory(model_config, config)
            if memory > hardware_config["memory_per_device"] * 0.95:
                continue

            # Communication cost
            comm_cost = estimate_alltoall_cost(model_config, config, hardware_config)

            # Compute cost
            compute_cost = estimate_compute_time(model_config, config)

            # Overall score (lower is better)
            score = compute_cost + comm_cost

            candidates.append({
                "config": config,
                "memory": memory,
                "comm_cost": comm_cost,
                "compute_cost": compute_cost,
                "score": score
            })

    # Sort by score
    candidates.sort(key=lambda c: c["score"])

    return candidates[:3]  # Top-3
```

---

## Integration with Base Strategy Rules

### Priority Order

When both MoE and non-MoE dimensions exist:

1. **Memory constraints** (highest priority)
   - Check total memory including MoE components
   - Adjust TP/FSDP first, then EP

2. **Communication constraints**
   - Balance All-to-All (EP) vs All-reduce (TP/DP)
   - Prefer reducing EP if both are high

3. **Compute efficiency**
   - Ensure expert utilization > 70%
   - Avoid excessive capacity_factor (>2.0)

### Example Decision Flow

```
Model: 48 layers, 128 experts, hidden=4096
Hardware: 64× Ascend 910B (64GB each)
Target: Max throughput

Step 1: Check memory without EP
  → 72GB baseline (OOM)

Step 2: Add EP to shard experts
  → Try ep_size=8: 58GB ✓

Step 3: Check if TP helps
  → hidden=4096 ≥ threshold
  → remaining=64/8=8 devices
  → Add tp_size=4: 56GB ✓, better compute

Step 4: Fill with DP
  → dp_size=64/(8×4)=2

Final: ep=8, tp=4, dp=2
```

---

## Validation Checks

### Pre-Strategy Validation

```python
def validate_moe_config(config, model):
    # Check 1: EP divides experts evenly
    assert model["num_experts"] % config["ep_size"] == 0, \
        "ep_size must divide num_experts"

    # Check 2: Capacity reasonable
    assert 1.0 <= config["capacity_factor"] <= 3.0, \
        "capacity_factor out of range"

    # Check 3: EP not too large (need at least 2 experts per rank)
    assert config["ep_size"] <= model["num_experts"] // 2, \
        "ep_size too large, need ≥2 experts per rank"

    # Check 4: Device count matches
    total_devices = config["ep_size"] * config["tp_size"] * config["dp_size"]
    assert total_devices == hardware["device_count"], \
        "Device count mismatch"
```

---

## Case Studies

### Case 1: Qwen3-30B-A3B

```yaml
model:
  num_experts: 128
  expert_hidden: 768
  hidden_size: 2048
  top_k: 8

hardware:
  devices: 64
  memory: 64GB

baseline_profiling:
  load_imbalance: 1.68
  alltoall_ratio: 0.32

psa_output:
  recommended:
    ep_size: 4         # Reduced from 8 to lower communication
    tp_size: 2         # Small TP for hidden=2048
    dp_size: 8
    capacity_factor: 1.5  # High due to load_imbalance=1.68

  memory: 61GB (95% utilized)
  alltoall_ratio_est: 0.22 (improved)
  throughput_est: 115K tok/s
```

---

## References

See also:
- [strategy-rules.md](strategy-rules.md) — Base parallelism rules
- [memory-estimation.md](memory-estimation.md) — Memory formulas
- [../../moe-profiler/references/moe-metrics.md](../../moe-profiler/references/moe-metrics.md) — MoE metrics definitions
