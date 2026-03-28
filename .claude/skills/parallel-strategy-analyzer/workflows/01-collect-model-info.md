# Phase 1: Collect Model & Hardware Info

## Goal

Gather all model architecture and hardware parameters needed for analysis. Auto-fill known models, infer missing values, validate consistency.

## Process

### 1.1 Parse User Input

Extract from user's request:
- Model name or total parameter count
- Device count and type
- Constraints: sequence length, batch size, memory budget, existing config

### 1.2 Auto-Fill Known Models

See `references/known-models.md` for full model architecture tables (LLaMA, GPT-3, Mixtral, DeepSeek-V2/V3/V3.2, Qwen/Qwen3/Qwen3.5) and parameter inference formulas for unknown models.

### 1.3 Collect Hardware Info

See `references/known-hardware.md` for full hardware specs (Ascend A2/A3/950DT, A100, H100/H200) and hardware-specific strategy guidelines. All values are per die — multi-die chips share bandwidth between dies.

### 1.3 Validate & Request Missing Info

**Required** (cannot proceed without):
- Total parameter count or model name
- Device count
- Device type or (memory + devices_per_node)

**Defaults if not provided:**
- `seq_len = 4096`
- `global_batch_size = 1024`
- `micro_batch_size = 1`
- `precision = "bf16"`
- `optimizer = "adam"`

## Output

Complete parameter set:

```
Model:    LLaMA-2-70B
  params: 70B, L=80, h=8192, n_h=64, n_kv=8 (GQA), d_ff=28672, V=32000
  seq_len=4096, global_batch=1024, micro_batch=1

Hardware: 64× Ascend A2 (1 die/chip)
  dies/node=8, total_dies=64, nodes=8
  mem/die=64GB, bw_intra=400GB/s, bw_inter=50GB/s, peak/die=256 TFLOPS
```
