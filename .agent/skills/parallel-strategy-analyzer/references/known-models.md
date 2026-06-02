# Known Model Architectures

Reference table for auto-filling model parameters in Phase 1.

## Dense Models

| Model | Params | L | h | n_h | n_kv | d_ff | V |
|-------|--------|---|---|-----|------|------|---|
| LLaMA-7B | 6.7B | 32 | 4096 | 32 | 32 | 11008 | 32000 |
| LLaMA-13B | 13B | 40 | 5120 | 40 | 40 | 13824 | 32000 |
| LLaMA-33B | 33B | 60 | 6656 | 52 | 52 | 17920 | 32000 |
| LLaMA-65B | 65B | 80 | 8192 | 64 | 64 | 22016 | 32000 |
| LLaMA-2-7B | 6.7B | 32 | 4096 | 32 | 32 | 11008 | 32000 |
| LLaMA-2-13B | 13B | 40 | 5120 | 40 | 40 | 13824 | 32000 |
| LLaMA-2-70B | 70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 32000 |
| LLaMA-3-8B | 8B | 32 | 4096 | 32 | 8 (GQA) | 14336 | 128256 |
| LLaMA-3-70B | 70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 128256 |
| GPT-3 175B | 175B | 96 | 12288 | 96 | 96 | 49152 | 50257 |
| Qwen-72B | 72B | 80 | 8192 | 64 | 64 | 24576 | 152064 |
| Qwen3-8B | 8B | 36 | 4096 | 32 | 8 (GQA) | 12288 | 151936 |
| Qwen3-32B | 32B | 64 | 5120 | 64 | 8 (GQA) | 25600 | 151936 |
| Qwen3.5-27B | 27B | 64 | 5120 | 24 | 4 (GQA) | 17408 | 248320 |

> **Qwen3.5**: head_dim=256, mixed attention (3 linear + 1 full, repeating). Multimodal (vision+text).

## MoE Models

| Model | Total Params | Active Params | L | h | n_h | n_kv | d_ff (shared) | V | Experts | top_k | MoE layers | moe_d_ff |
|-------|-------------|---------------|---|---|-----|------|---------------|---|---------|-------|------------|----------|
| Mixtral 8x7B | 46.7B | — | 32 | 4096 | 32 | 8 | 14336 | 32000 | 8 | 2 | 32 (all) | 14336 |
| Qwen3-30B-A3B | 30B | 3B | 48 | 2048 | 32 | 4 (GQA) | 6144 | 151936 | 128 | 8 | all | 768 |
| Qwen3-235B-A22B | 235B | 22B | 94 | 4096 | 64 | 4 (GQA) | 12288 | 151936 | 128 | 8 | all | 1536 |
| Qwen3.5-122B-A10B | 122B | 10B | 48 | 3072 | 32 | 2 (GQA) | — | 248320 | 256 | 8 | all | 1024 |
| Qwen3.5-397B-A17B | 397B | 17B | 60 | 4096 | 32 | 2 (GQA) | — | 248320 | 512 | 10 | all | 1024 |
| DeepSeek-V2 | 236B | 21B | 60 | 5120 | 128 | MLA | 12288 | 102400 | 160 | 6 | 59 | — |
| DeepSeek-V3 | 671B | 37B | 61 | 7168 | 128 | MLA | 18432 | 129280 | 256 | 8 | 58 | 2048 |
| DeepSeek-V3.2 | 685B | — | 61 | 7168 | 128 | MLA | 18432 | 129280 | 256 | 8 | 58 | 2048 |

> DeepSeek-V3/V3.2: 1 shared expert (d_ff=18432), first 3 layers dense.

## Special Attention Mechanisms

### MLA (Multi-head Latent Attention) — DeepSeek V2/V3/V3.2

Compresses KV cache into a low-rank latent vector instead of storing per-head K/V:

| Parameter | V2 | V3/V3.2 |
|-----------|-----|---------|
| kv_lora_rank | 512 | 512 |
| q_lora_rank | 1536 | 1536 |
| qk_rope_head_dim | 64 | 64 |
| v_head_dim | 128 | 128 |

**Impact on parallel strategy**: MLA compresses KV cache ~32× (512-dim vs 128×128=16384 for standard MHA). For CP ring attention, use `kv_lora_rank` instead of `n_kv × d_head` when estimating KV transfer volume — makes CP nearly free for MLA models.

### GQA (Grouped Query Attention) — LLaMA-2/3, Qwen3/3.5

`n_kv < n_h`: multiple query heads share one KV head. Reduces CP communication by `n_kv/n_h` ratio.

### Mixed Linear + Full Attention — Qwen3.5

Alternates 3 linear attention layers + 1 full attention layer (repeating). Linear attention layers have lower compute cost and no quadratic attention score memory. Only full attention layers contribute `s²` activation terms.

## Parameter Inference (Unknown Models)

```python
# From total_params, estimate architecture
# Standard transformer: params ≈ L × (12h² + 13h) + V×h
# Simplified: params ≈ 12 × L × h²

if only total_params and L known:
    h ≈ sqrt(total_params / (12 × L))
    n_h = h // 128                     # typical head_dim = 128
    n_kv = n_h                         # default MHA, not GQA
    d_ff = int(h × 8 / 3)             # SwiGLU default
    V = 32000                          # default

if only total_params known:
    # Use scaling law heuristics
    L ≈ total_params^0.2 × 4          # rough
    then infer h from above
```
