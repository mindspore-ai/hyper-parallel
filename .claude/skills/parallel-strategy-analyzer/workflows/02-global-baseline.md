# Phase 2: Global Baseline (No Parallelism)

Estimate total memory as if the entire model runs on one device. This determines the minimum parallelism needed.

## 2.1 Model States

```
BF16 mixed-precision training with Adam:
  param:     total_params × 2B
  gradient:  total_params × 2B
  optimizer: total_params × 12B  (FP32 master 4B + momentum 4B + variance 4B)
  ─────────────────────────────────
  total:     total_params × 16B
```

**Example**: 70B model → 70×10⁹ × 16B = **1120 GB** model states

## 2.2 Activation Memory (Per Layer, Itemized)

Notation: `s`=seq_len, `b`=micro_batch, `h`=hidden, `n_h`=heads, `n_kv`=KV heads, `d_ff`=FFN dim

### Attention Block

| Tensor | Shape | Bytes | Formula |
|--------|-------|-------|---------|
| Q projection | `[s, b, h]` | `2·s·b·h` | BF16 |
| K projection | `[s, b, n_kv·d_head]` | `2·s·b·h·(n_kv/n_h)` | GQA: smaller if n_kv < n_h |
| V projection | `[s, b, n_kv·d_head]` | `2·s·b·h·(n_kv/n_h)` | same as K |
| Attention scores | `[b, n_h, s, s]` | `2·s²·b·n_h` | **Quadratic in s** |
| Softmax output | `[b, n_h, s, s]` | `2·s²·b·n_h` | same shape |
| Dropout mask | `[b, n_h, s, s]` | `s²·b·n_h` | 1 byte per element |
| Attention output | `[s, b, h]` | `2·s·b·h` | before output proj |

```
attn_total = 2·s·b·h·(1 + 2·n_kv/n_h) + 5·s²·b·n_h + 2·s·b·h
```

For MHA (n_kv = n_h): `attn_total = 6·s·b·h + 5·s²·b·n_h`
For GQA (n_kv << n_h): QKV is smaller, but attn_scores still uses n_h

### MLP Block

**SwiGLU** (LLaMA-style, 3 weight matrices: gate, up, down):

| Tensor | Shape | Bytes |
|--------|-------|-------|
| Gate projection output | `[s, b, d_ff]` | `2·s·b·d_ff` |
| Up projection output | `[s, b, d_ff]` | `2·s·b·d_ff` |
| SiLU(gate) × up | `[s, b, d_ff]` | `2·s·b·d_ff` |
| Down projection output | `[s, b, h]` | `2·s·b·h` |

```
mlp_swiglu = 6·s·b·d_ff + 2·s·b·h
```

**Standard MLP** (2 weight matrices: up, down with d_ff = 4h):

```
mlp_standard = 4·s·b·(4h) + 2·s·b·h = 16·s·b·h + 2·s·b·h
```

### LayerNorm

| Tensor | Bytes |
|--------|-------|
| 2× RMSNorm/LayerNorm input | `4·s·b·h` |

> Residual-stream values are stored as layernorm inputs; the residual add backward passes gradients through without extra stored tensors.

```
norm_total = 4·s·b·h
```

### Total Per Layer

```
act_per_layer = attn_total + mlp_total + norm_residual

# SwiGLU model (LLaMA-style):
act_per_layer = (14·s·b·h + 5·s²·b·n_h + 6·s·b·d_ff)  bytes
  where the 14h term = 6h(QKV) + 2h(attn_out) + 2h(MLP_down) + 4h(layernorms)

# Standard MLP model (d_ff = 4h):
act_per_layer = (14·s·b·h + 5·s²·b·n_h + 4·s·b·d_ff)  bytes
  = (30·s·b·h + 5·s²·b·n_h)  when d_ff = 4h
```

### Total Activations

```
act_total = act_per_layer × L

# With full activation checkpoint:
act_total_recomp = L × s × b × h × 4B  (only layer boundary I/O)

# With selective checkpoint (recompute attn, keep MLP):
act_total_selective ≈ act_total × 0.5
```

## 2.3 Compute FLOPs

```
# Per layer, forward pass:
qkv_proj:    6·s·b·h²                   (2×s×b×h × 3h)
attn_score:  2·s²·b·h                   (2×s×b×n_h×d_head × s)
attn_ctx:    2·s²·b·h                   (2×s×b×n_h×s × d_head)
out_proj:    2·s·b·h²
mlp (SwiGLU): 6·s·b·h·d_ff              (gate + up + down, each 2×s×b×h×d_ff)
mlp (std):    4·s·b·h·(4h) = 16·s·b·h²

flops_per_layer_fwd = 8·s·b·h² + 4·s²·b·h + 6·s·b·h·d_ff  (SwiGLU)

# Backward ≈ 2× forward
flops_per_layer = 3 × flops_per_layer_fwd

flops_per_step = flops_per_layer × L × (B / (dp × b))
```

**Simplified**: `flops_per_step ≈ 6 × total_params × s × B`

## 2.4 Baseline Summary

```
┌─────────────────────────────────────────────────────┐
│ Global Baseline (single device, no sharding)        │
├──────────────────────┬──────────────────────────────┤
│ Parameters           │ X GB (total_params × 2B)     │
│ Gradients            │ X GB (total_params × 2B)     │
│ Optimizer states     │ X GB (total_params × 12B)    │
│ Model states total   │ X GB                         │
├──────────────────────┼──────────────────────────────┤
│ Activations          │ X GB (or Y GB w/ act ckpt)   │
│ Buffers              │ ~1.5 GB                      │
├──────────────────────┼──────────────────────────────┤
│ TOTAL                │ X GB                         │
│ Device memory        │ M GB                         │
│ Gap                  │ X / M = need ≥ N-way sharding│
├──────────────────────┼──────────────────────────────┤
│ Bottleneck           │ [model_states | activations]  │
│ FLOPs per step       │ X TFLOPS                     │
└──────────────────────┴──────────────────────────────┘
```

**Decision from baseline:**

- `total <= M × 0.9` → DP only
- `model_states <= M × 0.9, activations cause OOM` → act ckpt / CP / smaller batch
- `model_states > M × 0.9` → must shard: FSDP first → then TP → then PP (escalate as needed)
