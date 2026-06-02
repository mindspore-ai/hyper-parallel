# Memory Estimation Reference

## Memory Components

### 1. Model States

#### Per-Parameter Cost (BF16 + Adam)

| Component | Bytes/param | Notes |
|-----------|-------------|-------|
| Parameter | 2B | BF16 |
| Gradient | 2B | BF16 |
| Optimizer: FP32 master weight | 4B | Adam |
| Optimizer: momentum | 4B | Adam |
| Optimizer: variance | 4B | Adam |
| **Total** | **16B** | |

```
model_states = total_params × 16B
```

#### With Parallel Strategies

```
params_per_device  = total_params / tp / pp × 2B
grad_per_device    = total_params / tp / pp × 2B
opt_per_device     = total_params / tp / pp × 12B

With FSDP (shards across dp dimension):
  level1 (ZeRO-1): opt_per_device   /= dp
  level2 (ZeRO-2): opt_per_device   /= dp, grad_per_device /= dp
  level3 (ZeRO-3): all three        /= dp
```

### 2. Activation Memory

#### Per Transformer Layer (Itemized, BF16)

Notation: `s`=seq_len, `b`=micro_batch, `h`=hidden, `n_h`=heads, `n_kv`=KV heads, `d_ff`=FFN dim, `d_head`=h/n_h

**Attention block:**

| Tensor | Shape | Size (bytes) |
|--------|-------|-------------|
| QKV projection output | `s × b × 3h` | `6·s·b·h` |
| Attention scores | `s × s × b × n_h` | `2·s²·b·n_h` |
| Softmax output | `s × s × b × n_h` | `2·s²·b·n_h` |
| Dropout mask | `s × s × b × n_h` | `s²·b·n_h` |
| Attention output projection | `s × b × h` | `2·s·b·h` |

**MLP block (SwiGLU):**

| Tensor | Shape | Size (bytes) |
|--------|-------|-------------|
| Gate + Up projection | `s × b × d_ff × 2` | `4·s·b·d_ff` |
| SiLU activation | `s × b × d_ff` | `2·s·b·d_ff` |
| Down projection input | `s × b × d_ff` | `2·s·b·d_ff` (reuses gate·up) |
| Down projection output | `s × b × h` | `2·s·b·h` |

**LayerNorm + Residuals:**

| Tensor | Shape | Size (bytes) |
|--------|-------|-------------|
| 2× LayerNorm input | `s × b × h × 2` | `4·s·b·h` |

> Residual-stream values are stored as layernorm inputs; the residual add backward passes gradients through without extra stored tensors.

**Total per layer:**
```
act_per_layer = 14·s·b·h + 5·s²·b·n_h + 6·s·b·d_ff   (bytes)
```

**Standard MLP (d_ff = 4h, no SwiGLU):**
```
act_per_layer = 14·s·b·h + 5·s²·b·n_h + 4·s·b·(4h) = 30·s·b·h + 5·s²·b·n_h
```

#### With Parallel Strategies

```
# TP shards heads and FFN dim:
attention terms: n_h → n_h/tp, so attn_score terms / tp
MLP terms: d_ff → d_ff/tp

# CP shards sequence:
all terms with s → s/cp (but attn_scores: s² → (s/cp)² × cp ring steps...)
simplified: act / cp  (approximate for ring attention)

# PP shards layers:
layers_per_device = L / pp
```

#### With Activation Checkpoint

| Mode | Saved per layer | Total |
|------|----------------|-------|
| Full recomputation | Only layer I/O | `L/pp × s × b × h × 4B` |
| Selective (recompute attn) | ~50% reduction | `act_total × 0.5` |
| No checkpoint | Full activations | `act_per_layer × L/pp` |

### 3. Other Memory

| Component | Typical Size | Notes |
|-----------|-------------|-------|
| Communication buffers | 0.5-1.5 GB | AllReduce, FSDP AllGather buffers |
| Framework overhead | 0.5-1.0 GB | CUDA context, memory allocator |
| **Total overhead** | **~1.5 GB** | Conservative estimate |

## FSDP Memory Impact

| Level | Sharded | Savings | Communication |
|-------|---------|---------|---------------|
| Level 1 (ZeRO-1) | Optimizer states | `opt_mem × (1 - 1/dp)` | Baseline |
| Level 2 (ZeRO-2) | + Gradients | + `grad_mem × (1 - 1/dp)` | + gradient ReduceScatter |
| Level 3 (ZeRO-3) | + Parameters | + `param_mem × (1 - 1/dp)` | + param AllGather per layer |

## Memory Optimization Techniques

| Technique | Memory Savings | Cost |
|-----------|---------------|------|
| Activation checkpoint (full) | ~60-70% activation | ~33% recompute overhead |
| Activation checkpoint (selective) | ~30-50% activation | ~15% recompute overhead |
| FSDP level3 | Up to `(1-1/dp)` of all model states | AllGather per layer in forward |
| Optimizer offload (CPU) | ~optimizer_size freed | CPU-GPU transfer, PCIe bound |
| Parameter offload (CPU) | ~param_size freed | CPU-GPU transfer, PCIe bound |
| Activation swap (CPU) | ~activation_size freed | Prefetch scheduling, PCIe bound |

**Priority order** (short sequence): FSDP → activation checkpoint → activation swap (A3/950DT) → TP → PP → optimizer/param offload
