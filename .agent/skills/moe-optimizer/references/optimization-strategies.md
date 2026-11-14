# MoE Optimization Strategies

**Problem → Solution mapping table** for MoE training optimization.

This document is **descriptive only** — provides known strategies for known problems, without making decisions.

---

## Problem Type 1: Load Imbalance

### Pattern: Heavy-Tail Distribution

**Characteristics:**
- Load imbalance factor: 1.5 - 2.5
- 20-30% experts receive 2-3× average tokens
- Gini coefficient: 0.25 - 0.45

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Increase `capacity_factor`** | Allow more tokens per expert, reduce drops | 5-15% throughput | +10-20% memory |
| **Enable `expert_bias`** | Dynamic bias adjustment to balance load | 10-20% load improvement | Slight accuracy impact |
| **Increase `top_k`** | More experts per token → spread load | 15-25% load improvement | +30-50% compute |
| **Load-balance loss** | Auxiliary loss to encourage uniform routing | 10-15% load improvement | Hyperparameter tuning needed |

**Recommended combination:**
```yaml
# Conservative (first try)
capacity_factor: 1.25
expert_bias: true

# Aggressive (if conservative insufficient)
capacity_factor: 1.5
expert_bias: true
load_balance_loss_weight: 0.01
```

---

### Pattern: Bimodal Distribution

**Characteristics:**
- Two clusters: "active" (high load) vs "dormant" (rarely used)
- Some experts < 10% utilization
- Gini coefficient: > 0.5

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Reduce `num_experts`** | Fewer experts → higher utilization | Model change required | Requires retraining |
| **Expert dropout** | Randomly drop experts during training | Better generalization | May hurt specialized experts |
| **Retrain with load-balance loss** | From scratch with aux loss | Fundamental fix | Time-consuming |

**Recommended:**
- If possible, retrain with fewer experts
- Otherwise, increase capacity and bias (temporary fix)

---

## Problem Type 2: Communication Bottleneck

### Pattern: Large EP Size

**Characteristics:**
- AllToAll ratio: 0.3 - 0.5
- `ep_size` ≥ 8
- Hidden size < 4096

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Reduce `ep_size`** | Fewer ranks in All-to-All | 20-40% comm reduction | Need more memory (expert replication) |
| **Increase `tp_size`** | More computation per All-to-All | 10-25% effective improvement | More TP communication |
| **Hybrid EP+TP** | 2D expert parallelism | Balanced comm/compute | More complex setup |
| **Use static routing cache** | Reduce All-to-All frequency | 5-10% improvement | Less dynamic routing |

**Recommended progression:**
```yaml
# Step 1: Halve EP, double TP
ep_size: 8 → 4
tp_size: 2 → 4

# Step 2: If still high, further reduce EP
ep_size: 4 → 2
tp_size: 4 → 8

# Step 3: Enable caching (advanced)
static_routing: true
routing_cache_steps: 10
```

---

### Pattern: Slow Interconnect

**Characteristics:**
- Effective bandwidth < 50% theoretical
- AllToAll latency > 2ms for 32MB
- Network topology issues

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Optimize placement** | Co-locate EP ranks on same switch | 10-30% latency reduction | Placement constraints |
| **Reduce EP size** | Fewer hops per All-to-All | 15-25% improvement | More memory |
| **Async All-to-All** | Overlap comm with compute | 5-15% improvement | More complex impl |

**Recommended:**
- First try reducing `ep_size`
- If hardware allows, optimize rank placement
- Async overlap is advanced (requires code changes)

---

## Problem Type 3: Stragglers

### Pattern: Load-Induced Stragglers

**Characteristics:**
- Straggler factor: 1.2 - 1.5
- Correlation between expert load and rank slowdown
- Hot experts on specific ranks

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Address load imbalance** | Same as Problem Type 1 | 15-30% straggler reduction | See load imbalance trade-offs |
| **Dynamic expert mapping** | Reassign experts to balance ranks | 10-20% improvement | Requires dynamic scheduling |
| **Increase capacity** | Allow stragglers to buffer more tokens | 5-10% improvement | More memory |

**Recommended:**
- Treat as load imbalance problem
- Use capacity + bias solution

---

### Pattern: Hardware Stragglers

**Characteristics:**
- Straggler factor: 1.3 - 2.0
- No correlation with expert assignment
- Consistent slowdown on specific ranks

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Exclude straggler ranks** | Use fewer devices | Immediate improvement | Less parallelism |
| **Hardware diagnostics** | Identify and fix degraded devices | Full fix | Requires maintenance window |
| **Redundant computation** | Duplicate work on fast ranks | 10-20% improvement | Wasted compute |

**Recommended:**
- First: diagnose hardware (check device health)
- If unfixable: exclude straggler ranks from training
- Last resort: tolerate and optimize around it

---

## Problem Type 4: Memory Pressure

### Pattern: Near-OOM

**Characteristics:**
- Memory usage > 90% capacity
- Frequent cache evictions
- Occasional OOMs

**Known Solutions:**

| Solution | Mechanism | Expected Improvement | Trade-offs |
|----------|-----------|---------------------|------------|
| **Increase TP** | Shard model more | Directly reduces memory | More TP communication |
| **Enable FSDP** | Shard optimizer states | 30-50% memory reduction | Gradient All-reduce overhead |
| **Activation checkpointing** | Trade compute for memory | 40-60% memory reduction | 20-30% slower |
| **Reduce capacity factor** | Fewer active experts | 10-20% memory reduction | May increase drops |
| **Reduce batch size** | Less activation memory | Direct reduction | Lower throughput |

**Recommended progression:**
```yaml
# Step 1: Increase sharding (preferred)
tp_size: 2 → 4
fsdp: false → true

# Step 2: If still OOM, activation checkpoint
activation_checkpoint: true
checkpoint_every_n_layers: 4

# Step 3: Last resort, reduce batch size
global_batch_size: 2048 → 1024
```

---

## Combined Problems (Common Scenarios)

### Scenario A: Load Imbalance + Communication Bottleneck

**Strategy:**
1. Address load imbalance first (capacity + bias)
2. Then reduce EP size for communication
3. Monitor if load improvements persist with lower EP

```yaml
# Phase 1: Load balancing
capacity_factor: 1.25
expert_bias: true

# Phase 2: Reduce communication (next iteration)
ep_size: 8 → 4
```

**Caution:** Reducing EP may worsen load balance (fewer experts per rank)

---

### Scenario B: Communication Bottleneck + Memory Pressure

**Strategy:**
1. Cannot easily reduce EP (would increase memory)
2. Must increase TP/FSDP for memory
3. Accept communication overhead or optimize placement

```yaml
# Balance memory and communication
tp_size: 4  # Increase for memory
ep_size: 4  # Keep moderate for communication
fsdp: true  # Enable for optimizer memory
```

---

### Scenario C: All Three (Load + Comm + Memory)

**Strategy:**
1. Enable FSDP (most impactful for memory)
2. Moderate EP size (balance comm and memory)
3. Aggressive load balancing (capacity + bias)
4. May need activation checkpoint

```yaml
# Balanced approach
tp_size: 4
ep_size: 4
fsdp: true
capacity_factor: 1.5
expert_bias: true
activation_checkpoint: true  # If still OOM
```

---

## Parameter Ranges

### Capacity Factor

| Range | Effect | Use Case |
|-------|--------|----------|
| 1.0 | No overflow buffer | Balanced load, plenty memory |
| 1.1 - 1.25 | Small buffer | Moderate imbalance |
| 1.25 - 1.5 | Medium buffer | High imbalance |
| 1.5 - 2.0 | Large buffer | Severe imbalance or unstable routing |
| > 2.0 | Very large buffer | Last resort (high memory cost) |

### EP Size

| Range | Communication | Memory | Use Case |
|-------|--------------|---------|----------|
| 2 | Very low | High (expert replication) | Small models, slow interconnect |
| 4 | Low | Medium | Balanced default |
| 8 | Medium | Low | Large models, fast interconnect |
| 16+ | High | Very low | Huge models, memory-constrained |

**Rule of thumb:** `ep_size` should divide `num_experts` evenly

### TP Size

| Range | Communication | Memory | Use Case |
|-------|--------------|---------|----------|
| 1 | None | High | Small models |
| 2 | Low | Medium | Default for medium models |
| 4 | Medium | Low | Large models or memory pressure |
| 8+ | High | Very low | Huge models, aggressive sharding |

---

## Verification Checklist

After applying any strategy, verify:

- [ ] Throughput improved (or memory reduced if that was goal)
- [ ] No OOM or errors
- [ ] Routing integrity still PASS
- [ ] Gradient flow still PASS
- [ ] Load imbalance did not worsen
- [ ] Communication overhead acceptable

---

## When to Stop Optimizing

**Stop conditions:**
- Target throughput achieved ✓
- Improvement < 5% for 2 consecutive iterations
- Trade-offs no longer acceptable (e.g., memory too high)
- Correctness at risk

**Deploy best-effort if:**
- 3 iterations completed without reaching target
- Further optimization requires code changes
- Current config is stable and "good enough"

---

**See also:** [case-studies.md](case-studies.md) for real-world examples.
