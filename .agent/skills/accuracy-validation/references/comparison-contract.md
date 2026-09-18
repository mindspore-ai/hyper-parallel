# Accuracy Comparison Contract

## Manifest

Write `run_manifest.json` before either run. At minimum record:

```json
{
  "schema_version": 1,
  "reference": {"framework": "veomni", "revision": "<commit>"},
  "candidate": {"framework": "hyper_parallel", "revision": "<commit>"},
  "model": {"id": "<model>", "checkpoint": "<immutable-id>"},
  "data": {"revision": "<revision>", "sample_ids_file": "sample_ids.jsonl"},
  "training": {
    "seed": 1234,
    "dtype": "bf16",
    "global_batch": 8,
    "micro_batch": 1,
    "gradient_accumulation": 4,
    "loss_formula": "global_loss_sum/global_valid_token_count"
  },
  "parallel": {"dp": 2, "fsdp": 1, "tp": 1, "cp": 1, "ep": 1, "pp": 1},
  "tolerance_tier": "standard"
}
```

Use immutable checkpoint and dataset identifiers when available. Record generated config files and environment packages by
path or digest. Do not put machine-local secrets in the manifest.

## VeOmni Baseline Mapping

VeOmni is the external Trainer baseline. Map semantics, not similarly named fields:

| Contract | Freeze for both runs |
| --- | --- |
| Data | Sample order, packing, truncation, shifted labels, ignore index, masks, and valid-token count |
| Batch | Global/micro batch and the true number of data replicas; exclude TP/CP/EP from sample replication |
| Model | Same pretrained tensors, dtype conversion point, tied weights, dropout, and train/eval mode |
| Loss | Same component coefficients and token/sample normalization; save numerator and denominator separately |
| Optimizer | Parameter groups, state initialization, epsilon, betas, weight decay, clipping, and zero-grad policy |
| Scheduler | Warmup and step timing, including whether the scheduler advances on overflow/skipped update |
| Runtime | Seed streams, deterministic settings, autocast/TF32, activation checkpointing, and fused kernels |

When the two frameworks implement a lifecycle in different orders, compare state at named mathematical boundaries instead
of forcing identical call stacks.

## Trainer Component Matrix

Start from one proven reference configuration and toggle one component per row:

| Component | Primary evidence |
| --- | --- |
| Batch/data adapter | Exact sample IDs, token tensors or hashes, masks, positions, and valid-token counts |
| Model builder/replacements | Parameter names, shapes, dtypes, tied storage, and initial tensor fingerprints |
| Loss | Per-component sums, valid-token denominator, total objective, and label alignment |
| Optimizer/parameter groups | Group membership, hyperparameters, gradient values, and parameter delta |
| Scheduler | Learning rate before/after each successful optimizer update |
| Activation checkpoint/swap | Output, gradient, RNG, and saved-tensor lifecycle parity |
| Checkpoint/resume | Model/optimizer/scheduler/RNG/data position and resumed next-batch identity |
| Logging/callbacks | Step labels and metrics; never use logging parity as a proxy for math parity |

## Parallel Matrix

Validate each changed axis alone before combinations:

```text
reference -> FSDP/HSDP -> TP -> CP -> EP
                         \ selected pairwise interactions /
                          -> flagship hybrid topology
```

For each topology record mesh names/sizes, group membership, logical sample owner, result writer, parameter placements,
gradient reduction groups, and the loss/token reduction group. A topology with a configured axis of size one is not
evidence for that axis.

## High-Performance Module And Parallel Composition

Use the four cells below with identical weights and batches:

| Cell | Module | Parallelism |
| --- | --- | --- |
| A | Reference | Off or minimal |
| B | Optimized | Same as A |
| C | Reference | Target topology |
| D | Optimized | Target topology |

Compare B-A for module semantics, C-A for parallel semantics, and D-C plus D-B for interaction effects. Performance is
measured only after all required accuracy comparisons pass.

## Artifact Layout

```text
<output>/
├── run_manifest.json
├── baseline/loss.jsonl
├── candidate/loss.jsonl
├── boundaries/<name>.json
├── parameters/<step>.json
├── compare_loss.json
└── report.md
```

Large tensors may be stored in binary files, but the JSON evidence must retain shapes, dtypes, owners, stable digests, and
summary errors needed to locate the first mismatch.

## Executable Example

`examples/accuracy/veomni_vs_hyper/run_manifest.demo.json` is a self-contained protocol demonstration. It produces
synthetic loss records and proves only that manifest freezing, command execution, artifact collection, and tolerance
checking work. It is not model-accuracy evidence.

For a real comparison, copy `run_manifest.veomni.template.json`, replace every `REPLACE_ME` value, and preserve the command
contract:

- baseline and candidate commands are JSON argument arrays and execute without a shell;
- `{python}`, `{repo_root}`, `{output_dir}`, `{baseline_dir}`, `{candidate_dir}`, `{baseline_loss}`, and
  `{candidate_loss}` are the only runner substitutions;
- each side writes exactly one optimizer-step record to its declared loss JSONL path;
- the runner freezes the expanded-input manifest before launching either side and refuses to overwrite a different one.
