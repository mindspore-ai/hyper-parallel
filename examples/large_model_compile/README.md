# Large Qwen3-MoE compile demo

`large_qwen3_moe.py` builds a Qwen3-MoE model from `Qwen3MoeConfig`; it does
not load a checkpoint and therefore does not add large binary assets to the
repository. The configured network has:

- hidden size 8192, 6 decoder layers;
- 64 routed experts, 3840 expert intermediate size, top-8 routing;
- 64 attention heads / 8 KV heads and vocabulary size 65536.

The 8-NPU topology uses TP=2, EP=2, dense FSDP shard=2, and expert FSDP
shard=4. Full activation checkpointing and CP4 keep the backward workspace
within a 64-GiB card while retaining the large projection widths and expert
count. The benchmark uses the EP factory's per-expert fallback
(`use_grouped_gemm: false`) so the backward workspace remains bounded on the
test NPU runtime. The configured global sequence length is 4096; use
`SEQ_LENGTH=16384` for the 16K run (CP4 processes 4096 tokens per card).

Compilation is enabled in `train.yaml` with the Trainer's decoder-layer
contract and `backend: aot_eager`:

```yaml
compile:
  enabled: true
  backend: aot_eager
  fullgraph: false
  dynamic: false
```

`fullgraph: false` is required because FSDP keeps its eager hooks outside the
compiled graph. The local indexed mock dataset is only a deterministic input
source; no network or dataset download is performed.

## Run

```bash
cd /path/to/hyper-parallel
source /home/wyd/env.sh
export HYPER_PARALLEL_PLATFORM=torch
bash examples/large_model_compile/run.sh
```

The script launches 8 NPU processes with HCCL and performs one complete
forward/backward/optimizer step by default. Use `run_compare.sh` for a
repeatable compile on/off comparison, for example:

```bash
SEQ_LENGTH=4096 bash examples/large_model_compile/run_compare.sh
SEQ_LENGTH=16384 bash examples/large_model_compile/run_compare.sh
```

The script reports the first step separately from the later steady
steps because the first compiled step includes AOT eager graph creation. The
indexed mock batch currently reports zero `tokens/s` in the Trainer callback;
use the reported step time for comparison, or calculate throughput from the
configured sequence length.

With the configured AdamW state, increasing `STEPS` may exceed the 64-GiB
device memory after the first update; use a smaller layer count or an optimizer
with lower state memory when collecting steady-state multi-step measurements.
