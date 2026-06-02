# Phase 4: Communication & Bubble Cost Analysis

For each candidate, estimate per-step communication volume and exposed time.

## 4.1 TP Communication

TP has **4 AllReduce per layer** (2 in forward, 2 in backward — one for attention, one for MLP):

```python
ar_size = s * b * h * 2  # bytes, one AllReduce (BF16)

# Ring AllReduce: each rank sends (tp-1)/tp × ar_size
tp_volume_per_ar = 2 * (tp - 1) / tp * ar_size  # ring: 2× for reduce + scatter

tp_fwd_per_layer = tp_volume_per_ar * 2   # attn + MLP
tp_bwd_per_layer = tp_fwd_per_layer       # same in backward

layers_on_device = L // pp
tp_total_volume = (tp_fwd_per_layer + tp_bwd_per_layer) * layers_on_device

tp_num_ops = layers_on_device * 4
tp_latency_overhead = tp_num_ops * 8e-6   # ~8μs per intra-node AllReduce

tp_time = tp_total_volume / bw_intra + tp_latency_overhead
```

**TP is partially overlappable** with compute (backward AllReduce can overlap with next layer's gradient compute), but forward AllReduce is on critical path.

**Effective exposed time ≈ 60-80% of raw tp_time.**

## 4.2 CP Communication (Ring Attention)

Ring Attention: each device holds `s/cp` tokens, passes KV blocks in a ring for `cp-1` steps.

### KV Volume by Attention Type

The KV block size per ring step depends critically on the attention mechanism:

```python
d_head = h // n_h

# KV dimension per head — the key differentiator:
if attn_type == "MLA":
    # DeepSeek V2/V3: KV compressed to low-rank latent vector
    kv_dim = kv_lora_rank  # typically 512, vs n_kv × d_head = 16384 for equiv MHA
elif attn_type == "GQA":
    # LLaMA-2/3, Qwen3: fewer KV heads
    kv_dim = n_kv * d_head  # e.g. 8 × 128 = 1024 (vs 64 × 128 = 8192 for MHA)
else:  # MHA
    kv_dim = n_h * d_head   # = h, full size

kv_per_step = (s // cp) * b * kv_dim * 2 * 2  # K+V, BF16
```

**Relative CP communication cost:**

| Attention | Example (h=8192) | kv_dim | Relative to MHA |
| --------- | ----------------- | ------ | --------------- |
| MHA (n_kv=64) | LLaMA-65B | 8192 | 1× (baseline) |
| GQA (n_kv=8) | LLaMA-3-70B | 1024 | **0.125×** |
| MLA (rank=512) | DeepSeek-V3 | 512 | **0.0625×** |

### Ring Overlap with Compute

```python
# Each step's send/recv overlaps with that step's attention compute
# Exposed time = max(0, comm_per_step - compute_per_step) × (cp - 1)

# effective_bw: on 8-die nodes tp usually fills the node, so cp is cross-node
# cp * tp <= n_dev → bw_intra; cp * tp > n_dev → bw_inter
effective_bw = bw_intra if cp * tp <= n_dev else bw_inter

cp_comm_per_step = kv_per_step / effective_bw
cp_compute_per_step = 2 * (s//cp) * (s//cp) * b * n_h * d_head / (peak_tflops * 1e12)

# GQA/MLA: compute >> comm → ring perfectly hidden
# MHA: compute ≈ comm for large s → may have exposed time
cp_exposed_per_layer = max(0, cp_comm_per_step - cp_compute_per_step) * (cp - 1)
cp_total_exposed = cp_exposed_per_layer * layers_on_device * 2  # fwd + bwd

# Total volume (for reporting, even if hidden)
cp_total_volume = kv_per_step * (cp - 1) * layers_on_device * 2
```

**Summary**: For GQA/MLA models, CP communication is so cheap that it's nearly always hidden by compute — even when crossing nodes (8-die setups where `tp=8` fills the node). This makes CP the preferred first step for long-sequence training with these architectures. For MHA models, cross-node CP is costly (`bw_inter` ≈ 50-100 GB/s vs `bw_intra` ≈ 400-900 GB/s); prefer TP first, add CP only when activation memory still doesn't fit.

## 4.3 EP Communication (All-to-All)

```python
tokens = s * b
dispatched = tokens * top_k
ep_per_moe_layer = dispatched * h * 2 * (ep - 1) / ep  # BF16, both directions
ep_total_volume = ep_per_moe_layer * 2 * num_moe_layers  # fwd + bwd
ep_time = ep_total_volume / effective_bw

# All-to-All is hard to overlap — full penalty
```

## 4.4 DP/FSDP Communication

```python
params_on_device = total_params / tp / pp

# Pure DP: AllReduce gradients, overlapped with backward
dp_volume = 2 * (dp-1)/dp * params_on_device * 2  # ring AllReduce, BF16

# FSDP: AllGather (fwd) + ReduceScatter (bwd)
fsdp_ag = (dp-1)/dp * params_on_device * 2  # AllGather params
fsdp_rs = (dp-1)/dp * params_on_device * 2  # ReduceScatter grads
fsdp_volume = fsdp_ag + fsdp_rs

# Both are overlappable:
#   DP AllReduce: overlaps with backward compute
#   FSDP AllGather: prefetch next layer overlaps with current layer compute
#   FSDP ReduceScatter: overlaps with backward of next layer
dp_raw_time = max(dp_volume, fsdp_volume) / bw_inter
dp_exposed_time = dp_raw_time * 0.2  # ~80% overlap typical
```

## 4.5 PP Bubble & P2P Communication

```python
num_micro_batches = B // (dp * b)

# Standard 1F1B
bubble_1f1b = (pp - 1) / (num_micro_batches + pp - 1)

# Interleaved 1F1B (v virtual stages per device)
v = 2  # typical
bubble_interleaved = (pp - 1) / (num_micro_batches * v + pp - 1)

# P2P activation transfer
pp_act_size = s * b * h * 2  # BF16
pp_total_volume = pp_act_size * num_micro_batches * 2 * (pp - 1)  # fwd+bwd
pp_p2p_time = pp_act_size / bw_inter  # per stage boundary, pipelined
```

**Bubble assessment:**

| Bubble ratio | Assessment | Action |
|-------------|------------|--------|
| ≤ 5% | Excellent | — |
| 5-10% | Good | Production-ready |
| 10-20% | Acceptable | Use interleaved 1F1B |
| 20-30% | Poor | Increase micro-batches or reduce PP |
| > 30% | Reject | Use FSDP instead of PP |

## 4.6 Output: Communication Summary Table

```
| Dim   | Collective      | Volume  | Exposed time | Overlap       |
|-------|-----------------|---------|-------------|---------------|
| TP    | AllReduce ×4/L  | X.X GB  | X.X ms      | ~30% w/ comp  |
| CP    | Ring KV ×(cp-1) | X.X GB  | X.X ms      | Ring w/ attn  |
| EP    | All-to-All      | X.X GB  | X.X ms      | None          |
| DP    | AG+RS (FSDP)    | X.X GB  | X.X ms      | ~80% w/ bwd   |
| PP    | P2P             | X.X GB  | X.X ms      | 1F1B pipeline |
| PP bubble | —           | —       | ratio: X.X% | —             |
| TOTAL exposed         | | | X.X ms      |               |
```
