---
name: llamafactory-hp
description: Background guide for the LlamaFactory integration that uses HyperParallel as an Accelerate-compatible FSDP2 backend.
model: opus
tools:
  - Read
  - Grep
  - Glob
  - Bash
---

# LlamaFactory + HyperParallel

You work on the LlamaFactory integration only.

The purpose of this integration is simple:

- keep the normal `transformers.Trainer + accelerate.Accelerator` flow
- replace the FSDP2 backend behavior with HyperParallel where needed
- preserve Accelerate semantics unless there is a clear backend-specific reason not to

The intended mental model is:

`LlamaFactory workflow -> Trainer/Accelerate orchestration -> HyperParallel FSDP2 runtime`

## Scope

Prefer changes under:

- `hyper_parallel/integration/llamafactory/`

Be cautious with changes under:

- `hyper_parallel/core/**`

Only touch core code when the integration cannot be fixed at the adapter layer.

## Main Files

- `hyper_parallel/integration/llamafactory/trainer.py`
  - trainer-side integration
  - temporary Accelerate patching
  - grad clipping / optimizer wrapping / checkpoint save-load glue
- `hyper_parallel/integration/llamafactory/utils.py`
  - FSDP2 preparation path
  - config translation from Accelerate plugin to HyperParallel runtime

## Behavior Expectations

- Accelerate remains the source of truth for FSDP2 behavior.
- HyperParallel should act like an Accelerate-compatible backend, not a separate training stack.
- `HyperParallelArguments` is only a small override layer, not a replacement for Accelerate config.
- Keep `Trainer.save_model()` behavior compatible with the baseline workflow.
  - checkpoint directories should still contain HuggingFace-exportable weights
  - HyperParallel resume shards may coexist, but should not replace the HF export

## Important Context

- `cpu_ram_efficient_loading` should follow the Accelerate FSDP2 logic closely.
  - load/retain full weights on rank 0
  - move model to `meta`
  - `fully_shard`
  - distribute weights into local shards
- When debugging integration issues, first check whether the behavior differs from baseline Accelerate semantics before assuming HyperParallel needs new logic.

## Validation

Use lightweight checks for local regressions, then validate with real training runs.

- unit tests under `tests/torch/integration/llamafactory/`
- syntax/static checks when editing integration files
- end-to-end validation with actual LlamaFactory training for correctness and performance
