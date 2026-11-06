# Simple Model Training Example

FSDP + activation-checkpoint (recompute) training with HyperParallel Graph
Mode, driven by a `TrainerConfig` and the production `TextTrainer` entry.
There is no example-local entry script: the run uses the shared
`scripts/train_lm.py`, exactly like `examples/training_demo`.

## Quick Start

```bash
# Single card: override the FSDP shard degree (plain graph mode)
torchrun --nproc_per_node=1 scripts/train_lm.py \
    hyper_parallel/compile/examples/simple_model/config.yaml \
    --fsdp_config.dp_shard_size=1

# 8 cards
bash hyper_parallel/compile/examples/simple_model/run_8cards.sh
```

## Files

- `config.yaml` - The `TrainerConfig`: model / optimizer / data / graph-mode
  FSDP / activation checkpoint
- `model.py` - Toy `SimpleCausalLM` (HF checkpointing marker + repeated
  `layers` blocks) + the `model._target_` factory
- `data.py` - Constant synthetic batch / collator / get-batch targets (no corpus needed).
  Every sample is identical so the model overfits one batch and the loss falls
  monotonically — the signal that the graph-mode run is wired correctly.
- `run_8cards.sh` - 8-card launcher (runs `scripts/train_lm.py`)

## How it maps to the config

```yaml
compile:            # joint fwd+bwd graph capture (FSDP + AC passes)
  enabled: true
  use_joint_graph: true
accelerator:        # pure data parallel here
  tp_size: 1
fsdp_config:        # dp_shard_size > 1 enables FSDPPass parameter sharding
  dp_shard_size: 8
activation_checkpoint:   # projected onto PassConfig.ac_mode
  mode: full             # off / full / selective
  selection:
    source: default
model:
  _target_: ...simple_model.model.build_model
  num_layers: 4          # repeated recompute blocks
  hidden_dim: 512
dataset:
  _target_: ...simple_model.data.build_dataset
dataloader:
  collate_fn: {_target_: ...simple_model.data.build_collate_fn}
  get_batch:  {_target_: ...simple_model.data.build_get_batch}
```

The framework owns all parallel logic. `model.py` only materializes a
Trainer-compatible module (a `config.to_dict()` and a `forward(...)` returning
an object with `.loss`); `FSDPPass` shards its parameters in place during
`compile`, and `ActivationCheckpointPass` rebuilds block activations during
backward. `TextTrainer` owns the optimizer, micro-batching and the train loop.

## Activation checkpoint (recompute)

`ActivationCheckpointPass` runs on the plain joint graph **before** `FSDPPass`,
so recompute duplicates are ordinary forward ops when the FSDP collectives are
inserted. Forward ops inside the discovered blocks are tagged; every tagged op
the backward consumes is duplicated in front of its earliest backward consumer
and the original activation is freed at its last forward use.

| `mode` | Behaviour |
|--------|-----------|
| `off` | Save every activation; the pass is not added to the pipeline |
| `full` | Recompute every op inside the discovered blocks |
| `selective` | Save matmul / attention / communication ops, recompute the rest |

Always saved regardless of mode: RNG ops, in-place mutations, their forward
consumers, the mutated tensor's forward readers, and values read by forward code
across a block boundary. Ops outside the blocks (embedding, head, loss chain) are
never tagged.

`PassConfig.ac_save_forward_all_gather` (default `true`) saves forward
(TP/SP/CP) all-gathers instead of replaying them during backward recompute;
FSDP parameter all-gathers are inserted after this pass and are unaffected.
Set it to `false` through a programmatic `PassConfig`; it is not yet exposed
on `CompileConfig`.

Override the mode without editing the file:

```bash
torchrun --nproc_per_node=1 scripts/train_lm.py \
    hyper_parallel/compile/examples/simple_model/config.yaml \
    --fsdp_config.dp_shard_size=1 \
    --activation_checkpoint.mode=selective
```

The compile log reports the rewrite:

```text
hyper_parallel.compile.passes.activation_checkpoint: Activation checkpointing
(mode=full) recomputed 80 node(s)
```

To recompute only a subset of blocks, set the safe-region source with exactly
one selector: `selection.source: model_adapter_safe_regions` plus
`layer_count: 2` (or `layer_indices: [1, 3]`).

## Contract for a custom (toy) model

- a `torch.nn.Module` with a `config` exposing `to_dict()`
- `forward(**model_inputs, use_cache=False)` returning an object with `.loss`
- activation-checkpoint discovery contract: a `gradient_checkpointing`
  attribute on the model (or an ancestor module) and a repeated child container
  (`layers`); without it the graph AC pass rejects the model, exactly like the
  eager path
- the factory must leave the live module unwrapped: graph mode recomputes on
  the traced joint graph, so a custom `build_model` accepts and ignores the
  `activation_checkpoint` build kwargs
- `dataset`, `dataloader.collate_fn` and `dataloader.get_batch` are generic
  `Target`s, so the synthetic data plugs in without the indexed/online text
  stack

## Limitations

- FSDP + activation checkpoint only (TP/EP/PP not wired through this example)
- Parameters must have dim 0 divisible by `dp_shard_size`
- RNG ops inside a block stay saved, so a block whose dropout must be replayed
  is out of scope for the static AC path
- `activation_checkpoint.swap_inputs` / `allow_cache_entry_mutation` are
  ignored in graph mode (no eager checkpoint cache)
