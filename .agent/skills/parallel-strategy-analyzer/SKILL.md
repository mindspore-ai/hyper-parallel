---
name: parallel-strategy-analyzer
description: Analyze model architecture and hardware constraints to recommend optimal parallel strategy combinations (DP/FSDP/TP/PP/EP/CP) with memory, communication, compute, and pipeline bubble estimation.
---

# Parallel Strategy Analyzer

> Analyze model architecture, hardware topology, and resource constraints to recommend the optimal combination of parallel strategies for distributed training.

## When to Use

- Starting a new distributed training project and need to decide on parallel strategies
- Scaling an existing model to more devices and need to adjust strategies
- Encountering OOM errors and need memory/strategy optimization
- Evaluating trade-offs between different parallel strategy combinations
- Estimating communication overhead or pipeline bubble for a given config

## How to Use

```bash
# Basic usage
/parallel-strategy-analyzer Analyze strategy for a 70B LLaMA model on 64 Ascend A2 NPUs

# With specific constraints
/parallel-strategy-analyzer 13B model, 8× A100 80GB, sequence length 8192, batch size 512

# MoE model
/parallel-strategy-analyzer Mixtral 8x7B on 32 H100, seq_len 4096

# Optimize existing setup
/parallel-strategy-analyzer Currently using DP=8 for 7B model on 16 GPUs, OOM at seq_len=32768

# Compare specific configs
/parallel-strategy-analyzer Compare DP=4,TP=8,PP=2 vs DP=8,TP=8,PP=1 for LLaMA-70B on 64 A100s
```

## Input Requirements

### Required

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Model size** | Parameter count or model name | 70B, LLaMA-2-70B |
| **Device count** | Total devices | 64 |
| **Device type** | Hardware | Ascend A2/A3/950DT, A100 80GB, H100 |

### Optional (improves accuracy)

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Sequence length** | Training seq len | 4096, 32768, 128K |
| **Batch size** | Global batch size | 1024 |
| **Hidden/Layers/Heads** | Architecture details | h=8192, L=80, n_h=64 |
| **KV heads** | For GQA models | n_kv=8 |
| **FFN dim** | Intermediate size | d_ff=28672 |
| **MoE config** | Expert count, top-k | 8 experts, top-2 |
| **Framework** | PyTorch or MindSpore | PyTorch |

Well-known models (LLaMA, GPT, Mixtral) are auto-filled from built-in tables.

## Analysis Flow

```
Phase 1: Collect Info        → Parse model & hardware params, auto-fill known models
Phase 2: Global Baseline     → Single-device memory + FLOPs (no parallelism)
Phase 3: Strategy Search     → Enumerate valid (dp,tp,pp,cp,ep) combinations
Phase 4: Cost Analysis       → Per-strategy: TP/CP/EP/DP comm + PP bubble
Phase 5: Post-Shard Memory   → Per-device memory after sharding
Phase 6: Compute Efficiency  → MFU estimate, tokens/s throughput
Phase 7: Scoring & Ranking   → Weighted score: MFU + memory + comm + bubble
Phase 8: Output Report       → Recommendation + alternatives + code
```

Key improvement over naive approaches: **Phase 2 first determines the global baseline** (total memory, bottleneck type), then Phase 4-5 evaluate the **cost of sharding** (communication, bubble, actual per-device memory). This avoids recommending strategies that "fit in memory" but have prohibitive communication overhead.

## Output Format

### 1. Global Baseline
Total memory without parallelism, FLOPs per step, bottleneck analysis (model states vs activations).

### 2. Strategy Summary
```
Recommended: DP=4, TP=8, PP=2, FSDP=level2
Memory/device: 58GB / 80GB (72%)
Bubble: 5% (1F1B)
Communication overhead: 12%
MFU estimate: ~45%
```

### 3. DeviceMesh Code
Concrete `init_device_mesh()` + `fully_shard()` / `shard_module()` calls.

### 4. Post-Sharding Memory Breakdown
Per-device: params, grads, optimizer, activations, total.

### 5. Communication Breakdown
Per-dimension: volume, ops, exposed time, overlap potential.

### 6. Pipeline Bubble
Bubble ratio, schedule recommendation (1F1B vs interleaved).

### 7. Memory Optimizations
Activation checkpoint, FSDP level, offload recommendations.

### 8. Top 3 Alternatives
Comparison table with memory, bubble, comm overhead, and MFU.

## Detailed Reference

See workflow and reference files for formulas:
- `references/known-models.md` — model architecture tables (LLaMA, DeepSeek, Qwen3/3.5, etc.)
- `references/known-hardware.md` — hardware specs per die (Ascend A2/A3/950DT, A100, H100/H200)
- `references/memory-estimation.md` — per-tensor memory formulas, FSDP levels
- `references/strategy-rules.md` — decision tree, parallel dimension rules, config templates
- `workflows/01-collect-model-info.md` — Phase 1: hardware parameter tables
- `workflows/02-global-baseline.md` — Phase 2: global memory baseline + FLOPs
- `workflows/03-strategy-search.md` — Phase 3: strategy enumeration + pruning
- `workflows/04-cost-analysis.md` — Phase 4: TP/CP/EP/DP communication + PP bubble
- `workflows/05-scoring-output.md` — Phase 5-7: post-sharding memory + scoring + output

The `parallel-strategy-analyzer` agent has the complete formulas inline.

## Limitations

- Memory estimates are approximate (~10-20% error vs actual)
- Communication time estimates assume ideal bandwidth (no congestion)
- Overlap estimation (comp-comm) is heuristic, not from profiling
- MoE routing load imbalance is not modeled
- For production tuning, use `auto_parallel/fast-tuner` with profiling data
