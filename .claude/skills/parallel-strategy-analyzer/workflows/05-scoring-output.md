# Phase 5-7: Post-Sharding Memory, Scoring & Output

## Phase 5: Post-Sharding Per-Device Memory

For each candidate strategy from Phase 3, compute actual per-device memory.

### 5.1 Model States After Sharding

```
# Split shared (non-expert) and expert params to avoid double-counting
shared_params = total_params - expert_params  # expert_params = 0 for dense models

params_per_device  = (shared_params / tp / pp + expert_params / tp / pp / ep) × 2B
grad_per_device    = (shared_params / tp / pp + expert_params / tp / pp / ep) × 2B
opt_per_device     = (shared_params / tp / pp + expert_params / tp / pp / ep) × 12B

# FSDP further shards across dp
level1: opt_per_device   /= dp
level2: opt_per_device   /= dp;  grad_per_device /= dp
level3: all three        /= dp
```

### 5.2 Activations After Sharding

Apply sharding to the itemized formula from Phase 2:

```
# TP: shards n_h and d_ff
attn_scores terms: n_h → n_h/tp  (but QKV projection of full h remains if TP splits output)
MLP terms: d_ff → d_ff/tp
Other h terms: unchanged (residuals, norms are replicated)

Simplified: act_per_layer / tp  (approximate, slightly underestimates)

# CP: shards sequence
s → s/cp  (for most terms)
attn_scores: s² → (s/cp)²  (quadratic reduction — biggest win for long sequences)

# PP: shards layers
layers_per_device = L / pp

act_total_sharded = act_per_layer_sharded × layers_per_device
```

**With activation checkpoint:**

```
act_total = (L/pp) × (s/cp) × b × h × 4B / tp  (approximate)
```

### 5.3 Total Per Device

```
total = model_states_per_device + act_total_sharded + 1.5GB

fits = total <= M × 0.9
memory_utilization = total / M
```

### 5.4 Compare with Baseline

```
┌───────────────────┬──────────┬──────────┬───────────┐
│ Component         │ Baseline │ Sharded  │ Reduction │
├───────────────────┼──────────┼──────────┼───────────┤
│ Parameters        │ X GB     │ Y GB     │ Nx        │
│ Gradients         │ X GB     │ Y GB     │ Nx        │
│ Optimizer         │ X GB     │ Y GB     │ Nx        │
│ Activations       │ X GB     │ Y GB     │ Nx        │
├───────────────────┼──────────┼──────────┼───────────┤
│ Total             │ X GB     │ Y GB     │ Nx        │
│ Fits in M GB?     │ No       │ Yes/No   │           │
│ Utilization       │ —        │ Y%       │           │
│ Remaining bottleneck│ —      │ [component]│          │
└───────────────────┴──────────┴──────────┴───────────┘
```

**If still doesn't fit:**

- Activations dominate → add activation checkpoint
- Model states dominate → upgrade FSDP level (1→2→3)
- Still tight → optimizer offload → parameter offload → activation swap

---

## Phase 6: Scoring & Ranking

```python
def score(config, mem_per_device, comm_analysis, flops_per_step, M, peak_tflops):
    dp, tp, pp, cp, ep = config

    # --- OOM filter ---
    if mem_per_device > M * 0.9:
        return -float('inf')

    score = 0

    # --- Compute efficiency (highest weight) ---
    compute_time_ms = flops_per_step / (peak_tflops * 1e12) * 1000
    total_exposed_ms = comm_analysis['total_exposed_ms']
    bubble_overhead_ms = comm_analysis['bubble_ratio'] * compute_time_ms

    step_time_ms = compute_time_ms + total_exposed_ms + bubble_overhead_ms
    mfu = flops_per_step / (peak_tflops * 1e12 * step_time_ms / 1000)
    score += mfu * 100  # MFU is the primary metric

    # --- Memory fitness: prefer 70-80% utilization ---
    mem_util = mem_per_device / M
    score += (1 - abs(mem_util - 0.75)) * 20

    # --- Simplicity bonus ---
    active_dims = sum(1 for x in [dp>1, tp>1, pp>1, cp>1, ep>1] if x)
    score -= active_dims * 2

    return score, mfu, step_time_ms
```

---

## Phase 7: Generate Output Report

### 1. Global Baseline

From Phase 2: total memory, FLOPs, bottleneck analysis.

### 2. Strategy Recommendation

```
Recommended: DP=4, TP=8, PP=2, CP=1, FSDP=level2
  Memory/device:  58 GB / 80 GB (72%)
  Bubble:         5.3% (1F1B, recommend interleaved for 2.7%)
  Comm overhead:  12% of compute time
  MFU estimate:   ~45%
  Throughput:     ~125K tokens/s
```

### 3. DeviceMesh Code

```python
from hyper_parallel import init_device_mesh
mesh = init_device_mesh("npu", (4, 2, 8), mesh_dim_names=("dp", "pp", "tp"))
```

### 4. Memory Breakdown (Baseline vs Sharded)

Table comparing Phase 2 baseline with Phase 5 per-device numbers.

### 5. Communication Breakdown

Summary table from Phase 4.

### 6. Memory Optimizations

```
if activations > model_states_per_device:
    → "Enable activation checkpoint (saves ~60% activation memory)"
if model_states tight:
    → "Upgrade FSDP: level1→2→3 as needed"
if still tight:
    → "Optimizer offload to CPU"
    → "Activation swap for long sequences"
```

### 7. FSDP Level Recommendation

```python
def recommend_fsdp_level(params_pd, grad_pd, opt_pd, act_pd, M):
    """Find lowest FSDP level that fits in memory."""
    total_no_fsdp = params_pd + grad_pd + opt_pd + act_pd + 1.5e9
    if total_no_fsdp <= M * 0.9:
        return None  # no FSDP needed

    # Level 1: shard optimizer
    total_l1 = params_pd + grad_pd + opt_pd/dp + act_pd + 1.5e9
    if total_l1 <= M * 0.9:
        return "level1"

    # Level 2: shard optimizer + gradients
    total_l2 = params_pd + grad_pd/dp + opt_pd/dp + act_pd + 1.5e9
    if total_l2 <= M * 0.9:
        return "level2"

    # Level 3: shard all
    return "level3"
```

### 8. Top 3 Alternatives

```
| # | Config              | Mem/Dev  | Bubble | Comm  | MFU  | Notes           |
|---|---------------------|----------|--------|-------|------|-----------------|
| 1 | dp=4,tp=8,pp=2     | 58/80 GB | 5%     | 12%   | ~45% | Recommended     |
| 2 | dp=8,tp=8,pp=1     | 72/80 GB | 0%     | 8%    | ~48% | No PP, tight mem|
| 3 | dp=2,tp=8,pp=4     | 40/80 GB | 15%    | 18%   | ~38% | More headroom   |
```

Include brief trade-off commentary for each alternative.
