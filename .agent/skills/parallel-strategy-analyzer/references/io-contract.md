# PSA input / output contract

Load with `SKILL.md` when collecting inputs or formatting the final report.

## How to invoke (examples)

```bash
/parallel-strategy-analyzer Analyze strategy for a 70B LLaMA model on 64 Ascend A2 NPUs
/parallel-strategy-analyzer 13B model, 8× A100 80GB, sequence length 8192, batch size 512
/parallel-strategy-analyzer Mixtral 8x7B on 32 H100, seq_len 4096
/parallel-strategy-analyzer Currently using DP=8 for 7B on 16 GPUs, OOM at seq_len=32768
/parallel-strategy-analyzer Compare DP=4,TP=8,PP=2 vs DP=8,TP=8,PP=1 for LLaMA-70B on 64 A100s
```

## Required inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| Model size | Parameter count or name | 70B, LLaMA-2-70B |
| Device count | Total devices | 64 |
| Device type | Hardware | Ascend A2/A3/950DT, A100 80GB, H100 |

## Optional inputs (improves accuracy)

| Parameter | Description | Example |
|-----------|-------------|---------|
| Sequence length | Training seq len | 4096, 32768, 128K |
| Batch size | Global batch | 1024 |
| Hidden / Layers / Heads | Architecture | h=8192, L=80, n_h=64 |
| KV heads | GQA | n_kv=8 |
| FFN dim | Intermediate | d_ff=28672 |
| MoE config | Experts, top-k | 8 experts, top-2 |
| Framework | Backend | PyTorch / MindSpore |

Known models (LLaMA, GPT, Mixtral, …) auto-fill from `known-models.md`.

## Output sections (required)

1. **Global baseline** — memory without parallelism, FLOPs/step, bottleneck
2. **Strategy summary** — e.g. `DP=4, TP=8, PP=2, FSDP=level2` + mem/bubble/comm/MFU
3. **DeviceMesh code** — `init_device_mesh` + `fully_shard`
4. **Post-shard memory** — params, grads, optimizer, activations per device
5. **Communication breakdown** — per-dimension volume / exposed time
6. **Pipeline bubble** — ratio + 1F1B vs interleaved
7. **Memory optimizations** — checkpoint / FSDP level / offload
8. **Top 3 alternatives** — comparison table

## Limitations

- Memory ~10–20% vs actual; comm assumes ideal bandwidth
- Comp–comm overlap is heuristic; MoE load imbalance not modeled
- Production tuning: prefer `auto_parallel/fast-tuner` + profiling
