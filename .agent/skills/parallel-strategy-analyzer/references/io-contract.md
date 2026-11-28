# PSA input / output contract

Load with `SKILL.md` when collecting inputs or formatting the final report.

## How to invoke (examples)

Replace hardware placeholders with the current job's confirmed allocation.
Counts refer to logical training devices, not physical multi-die cards.

```bash
/parallel-strategy-analyzer Analyze <model> on <device_count> <device_type> devices, <memory_per_device_gb> GB each
/parallel-strategy-analyzer <model>, <device_count> <device_type>, <memory_per_device_gb> GB each, seq_len=<seq_len>, batch_size=<batch_size>
/parallel-strategy-analyzer <model>, <current_strategy>, <device_count> <device_type>, <memory_per_device_gb> GB each, OOM at seq_len=<seq_len>
/parallel-strategy-analyzer Compare <candidate_1> vs <candidate_2> for <model> on <device_count> <device_type>, <memory_per_device_gb> GB each
```

## Required inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| Model size | Parameter count or name | 70B, LLaMA-2-70B |
| Device count | Total logical devices allocated to the job | `<device_count>` |
| Device type | User-specified hardware and deployment configuration | `<device_type>` |

## Optional inputs (improves accuracy)

| Parameter | Description | Example |
|-----------|-------------|---------|
| Sequence length | Training seq len | 4096, 32768, 128K |
| Batch size | Global batch | 1024 |
| Hidden / Layers / Heads | Architecture | h=8192, L=80, n_h=64 |
| KV heads | GQA | n_kv=8 |
| FFN dim | Intermediate | d_ff=28672 |
| MoE config | Experts, top-k | 8 experts, top-2 |
| Framework | Backend | PyTorch |

Known models (LLaMA, GPT, Mixtral, …) auto-fill from `known-models.md`.

For hardware outside the reference configurations, collect job-visible devices
per node, device mapping, per-device memory and effective communication bandwidth
as described in [the hardware collection procedure](known-hardware.md).
Record the source and units; request missing inputs before calculations that
depend on them. A device name alone does not determine the deployment topology.

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
