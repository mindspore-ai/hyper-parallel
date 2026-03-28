# Parallel Strategy Selection Rules

## Principle

Strategy selection is **memory-driven**. The total memory cost is determined by model parameters and sequence length. Estimate memory first, then select the minimal set of strategies that fits within device memory while maximizing compute efficiency.

## Decision Tree

```
Start → Estimate memory (model states + activations)
 │
 ├─ Model fits on single device?
 │   └─ Yes → DP only (simplest, best efficiency)
 │
 ├─ Short sequence (≤ 32K): memory dominated by model states
 │   │
 │   ├─ Step 1: FSDP (shard optimizer/grads/params across DP ranks)
 │   │   └─ Fits? → Done (FSDP + DP)
 │   │
 │   ├─ Step 2: + Activation checkpoint (recomputation)
 │   │   └─ Fits? → Done (FSDP + DP + recompute)
 │   │
 │   ├─ Step 3: + Activation swap (A3 / 950DT with high bandwidth)
 │   │   └─ Fits? → Done (FSDP + DP + recompute + swap)
 │   │
 │   ├─ Step 4: + TP (intra-node, shard params & activations)
 │   │   └─ Fits? → Done (FSDP + TP + DP)
 │   │
 │   └─ Step 5: + PP (split layers across nodes)
 │       └─ FSDP + TP + PP + DP (+ recompute + swap as needed)
 │
 ├─ Long sequence: activations dominate (s² attention scores)
 │   │
 │   │  Trigger: act_s2_ratio > 50% of per-layer activation memory
 │   │    act_s2 = 5·s²·b·n_h;  act_linear = 14·s·b·h + 6·s·b·d_ff
 │   │    ratio  = act_s2 / (act_s2 + act_linear)
 │   │    Rough threshold: s > (14·h + 6·d_ff) / (5·n_h)
 │   │    Practical minimum: s ≥ 8192 (below this, act ckpt alone is enough)
 │   │
 │   ├─ Classify attention type → determines CP priority:
 │   │   │
 │   │   ├─ MLA (DeepSeek V2/V3): KV compressed to kv_lora_rank=512
 │   │   │   CP comm volume uses kv_lora_rank instead of n_kv × d_head
 │   │   │   → CP nearly free → strongly prefer CP before TP
 │   │   │   → cp_max: 16 (comm so cheap, large CP viable)
 │   │   │
 │   │   ├─ GQA (n_kv << n_h, e.g. n_kv=8, n_h=64):
 │   │   │   CP comm reduced by n_kv/n_h (e.g. 8×)
 │   │   │   → CP cheap → prefer CP before TP
 │   │   │   → cp_max: 8
 │   │   │
 │   │   └─ MHA (n_kv = n_h):
 │   │       CP comm is full (no KV compression)
 │   │       → CP expensive → prefer TP first, then CP
 │   │       → cp_max: 4 (avoid excessive comm overhead)
 │   │
 │   ├─ MLA/GQA escalation: FSDP → recompute → CP → swap → TP → PP
 │   ├─ MHA escalation:     FSDP → recompute → TP → CP → swap → PP
 │   └─ Activation checkpoint strongly recommended (eliminates stored s² terms)
 │
 ├─ MoE model?
 │   └─ Yes → Add EP (Expert Parallelism)
 │
 └─ Summary: FSDP first → recompute → CP (if long seq) → swap → TP → PP → EP
```

## Core Rules

### Rule 1: TP Prefers Intra-Node

TP requires AllReduce at every transformer layer (forward + backward), generating the most frequent communication. Always keep TP within a single node where NVLink/HCCS provides high bandwidth.

```
tp_size <= devices_per_node
```

**Typical values**: tp=1, 2, 4, 8 (matching intra-node device count)

### Rule 2: DP/FSDP for Scaling Across Nodes

DP/FSDP communication (gradient AllReduce/ReduceScatter) happens once per training step, making it tolerant of lower inter-node bandwidth.

```
dp_size = total_devices / (tp_size × pp_size × cp_size × ep_size)
```

### Rule 3: PP for Very Large Models

PP splits model layers across devices. Use when model doesn't fit even with TP:

```
pp_size = ceil(model_layers × param_per_layer / max_param_per_device)
```

**Trade-off**: PP introduces pipeline bubbles. Minimize with:
- Interleaved 1F1B schedule
- Virtual Pipeline Parallelism (VPP)
- More micro-batches (reduce bubble ratio)

```
bubble_ratio ≈ (pp_size - 1) / num_micro_batches
```

### Rule 4: CP for Long Sequences (Attention-Type Aware)

CP shards the sequence dimension, giving **quadratic** activation reduction (`s² → (s/cp)²`).

**Dynamic trigger** (not a fixed threshold):
```
act_s2 = 5 × s² × b × n_h                    # quadratic attention terms
act_linear = 14 × s × b × h + 6 × s × b × d_ff  # linear terms
if act_s2 / (act_s2 + act_linear) > 0.5:
    long_sequence = True  # s² terms dominate, CP is valuable
# Practical minimum: s ≥ 8192 (below this, act ckpt alone handles it)
```

**CP priority depends on attention type** (lower comm → higher priority):

| Attention | KV volume factor | CP comm cost | Priority vs TP | cp_max |
|-----------|-----------------|--------------|----------------|--------|
| MLA (DeepSeek) | `kv_lora_rank` (512) | ~32× smaller than MHA | **CP first** | 16 |
| GQA (`n_kv << n_h`) | `n_kv × d_head` | `n_kv/n_h` × MHA | **CP first** | 8 |
| MHA (`n_kv = n_h`) | `n_h × d_head` | Full | **TP first, then CP** | 4 |

> **Cross-node CP is unavoidable on 8-die nodes** when `tp > 1` (TP occupies intra-node slots). GQA/MLA models tolerate this well because KV volume is small relative to inter-node bandwidth. MHA models pay a steep penalty — prefer TP=8 (filling the node) and only add CP=2-4 cross-node if activation memory still doesn't fit.

```
# CP placement in escalation order:
MLA/GQA: FSDP → recompute → CP → swap → TP → PP
MHA:     FSDP → recompute → TP → CP → swap → PP
```

### Rule 5: EP for MoE Models

Expert Parallelism distributes experts across devices:

```
ep_size <= num_experts
ep_size should divide num_experts evenly
```

**Constraint**: `ep_size × tp_size <= devices_per_node` preferred (All-to-All is bandwidth-sensitive)

### Rule 6: FSDP as First-Line Memory Strategy

For short-sequence training, FSDP is the **first strategy to apply** — it shards model states across DP ranks with minimal communication overhead (once per step). Only add TP/PP when FSDP + recomputation is insufficient.

```
if model does not fit on single device:
    enable FSDP first
    level = "level1" if optimizer_states alone cause OOM
    level = "level2" if gradients also significant
    level = "level3" if parameters also need sharding
```

### Rule 7: Memory Optimization Escalation

Apply memory optimizations in this order (each adds overhead, so stop as soon as memory fits):

```
1. FSDP (shard model states)         — low overhead, once-per-step comm
2. Activation checkpoint (recompute) — ~33% recompute cost, large mem savings
3. Activation swap (A3/950DT)        — needs high PCIe/die bandwidth, prefetch scheduling
4. TP (tensor parallel)              — per-layer AllReduce, must be intra-node
5. PP (pipeline parallel)            — pipeline bubble overhead
6. Optimizer/param offload (CPU)     — PCIe bound, last resort
```

---

## Constraint Validation Rules

### Divisibility Constraints

```
num_attention_heads % tp_size == 0       # Heads must divide evenly for TP
num_layers % pp_size == 0                # Layers must divide evenly for PP
global_batch_size % (dp_size × micro_batch_size) == 0  # Batch must divide for DP
num_experts % ep_size == 0               # Experts must divide for EP (MoE)
seq_len % (cp_size × 2) == 0            # Sequence must divide for CP
```

### Memory Constraint

```
estimated_memory_per_device <= device_memory × 0.9  # 10% safety margin
```

### Communication Constraint

```
# TP must be intra-node
tp_size <= devices_per_node

# CP bandwidth reality:
#   On 8-die nodes: tp already occupies most/all intra-node slots,
#   so cp > 1 almost always requires cross-node communication.
#   Only 16-die nodes (A3/950DT) can fit both tp and cp intra-node.
cp_intra = devices_per_node / tp_size        # max CP that stays intra-node
# If cp > cp_intra → effective_bw = bw_inter (not bw_intra)
# For GQA/MLA this is acceptable (low KV volume); for MHA it's costly.
```

---

## Common Strategy Templates (Short Sequence ≤ 32K)

### Small Model (< 10B), 8-16 Devices

```
FSDP level1 + DP=N              # FSDP alone is enough
# or DP=N without FSDP           # if model fits on single device
```

### Medium Model (10B-30B), 16-64 Devices

```
FSDP level2 + DP=N              # shard optimizer + grads
FSDP level2 + DP=N + recompute  # if activations tight
```

### Large Model (30B-100B), 64-256 Devices

```
FSDP level2 + DP=N + recompute                    # try FSDP-only first
FSDP level2 + TP=8 + DP=N/8 + recompute           # add TP if still OOM
FSDP level2 + TP=8 + PP=2 + DP=N/16 + recompute   # add PP for very large
# A3/950DT: can add activation swap before resorting to TP
```

### Very Large Model (100B+), 256+ Devices

```
FSDP level2 + TP=8 + PP=4 + DP=N/32 + recompute
FSDP level3 + TP=8 + PP=8 + DP=N/64 + recompute   # level3 if needed
# A3/950DT: FSDP level2 + TP=16 + PP=4 + swap + recompute
```

## Common Strategy Templates (Long Sequence)

### GQA/MLA Models (CP-first — low CP comm)

```
# GQA (LLaMA-2/3, Qwen3): CP before TP
FSDP level2 + CP=4 + DP=remaining + recompute          # try CP-only first
FSDP level2 + CP=4 + TP=4 + DP=remaining + recompute   # add TP if still OOM

# MLA (DeepSeek V2/V3): CP nearly free
FSDP level2 + CP=8~16 + DP=remaining + recompute       # large CP viable
FSDP level2 + CP=8 + TP=4 + DP=remaining + recompute   # add TP for model states
```

### MHA Models (TP-first — high CP comm)

```
# MHA (n_kv = n_h): TP before CP
FSDP level2 + TP=8 + DP=remaining + recompute          # TP reduces both params & acts
FSDP level2 + TP=8 + CP=2~4 + DP=remaining + recompute # add CP if still OOM
```

### MoE Model

```
FSDP level2 + EP=num_experts/k + TP=8 + DP=remaining
# EP × TP should fit in a node if possible
```

---

> Hardware specs and hardware-specific strategy guidelines: see `references/known-hardware.md`.
