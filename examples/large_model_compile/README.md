# Large Qwen3-MoE compile demo

`large_qwen3_moe.py` builds a Qwen3-MoE model from `Qwen3MoeConfig`; it does
not load a checkpoint and therefore does not add large binary assets to the
repository. The configured network has:

- hidden size 8192, 72 decoder layers;
- 64 routed experts, 3840 expert intermediate size, top-8 routing;
- 64 attention heads / 8 KV heads and vocabulary size 65536.

The model contains about 446.8B parameters. In bf16 this is about 832GiB
globally. The 16-NPU topology uses TP=2 and FSDP shard=8, so the model
parameters alone occupy approximately 52GiB per card, satisfying the
large-memory compile benchmark target. Activations and workspaces require
additional headroom; a card with less than 64GiB should not be used.

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

The script launches 16 NPU processes with HCCL and performs one training step.
The AdamW state is lazy, but the first optimizer update can require substantial
extra memory. For a compile-only capacity check, interrupt after all decoder
layers report successful compilation, or adapt the local Trainer entry point to
skip the optimizer update.
