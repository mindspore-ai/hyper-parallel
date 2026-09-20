---
name: model-development-validation
description: Hard evidence and comparison rules for Trainer, model adapters, parallel strategies, benchmarks, and high-performance modules.
paths:
  - hyper_parallel/trainer/**
  - hyper_parallel/models/**
  - hyper_parallel/distributed/**
  - hyper_parallel/components/**
  - examples/accuracy/**
  - examples/training_demo/**
  - examples/evaluation/**
---

# Model Development And Validation

Use these rules when changing Trainer behavior, a model-family adapter, TP/CP/EP/FSDP integration, benchmark execution,
or a high-performance module.

## Evidence Levels

Use the strongest applicable level and name it in the report:

| Level | Evidence | What it proves |
| --- | --- | --- |
| Structure | Imports, config, registry, plan, and static assertions | The feature is wired and discoverable. |
| Contract | Deterministic CPU/source checks for ownership, shapes, masks, counts, and ordering | The declared interface is internally consistent. |
| Smoke | A minimal real forward/backward/update on the target runtime | The execution path runs. |
| Parity | Baseline and candidate execute the same frozen inputs and compare observables | The change preserves the declared math within tolerance. |
| Benchmark | A pinned benchmark compares complete sample-level results | Model capability is preserved. |
| Impact | Repeated measurements of throughput, latency, memory, or MFU | The claimed performance effect exists. |

Structure and contract evidence never substitute for smoke or parity. A skipped, unavailable, or single-rank run is not
evidence for a multi-rank claim.

## Frozen Comparison Contract

Before running a comparison, freeze a manifest containing:

- source revisions for HyperParallel, VeOmni or another baseline, model/checkpoint, tokenizer/processor, and dataset;
- exact sample IDs or stable batch fingerprints;
- seed and RNG policy, dtype/autocast/TF32 policy, optimizer, scheduler, clipping, and gradient accumulation;
- global/micro batch, sequence/packing rules, valid-token definition, and loss reduction formula;
- device type, world size, mesh axes, rank groups, kernels, compiler flags, and environment versions;
- warmup, measurement window, repetitions, output paths, and the predeclared tolerance tier.

Do not widen tolerances, change failed samples, or alter the workload after seeing candidate results. A changed contract is
a new experiment and must produce a new manifest.

## Correctness Invariants

- Compare the same model state before the first step. Record missing/unexpected keys and parameter fingerprints.
- The same logical sample is replicated within TP/CP/EP model-cooperation groups and partitioned only across true data
  replicas. Model-parallel ranks do not multiply global batch size or metric counts.
- Aggregate causal-LM loss as `global_loss_sum / global_valid_token_count`. Never average rank-local or micro-batch means
  when token counts differ.
- Save loss sums, token counts, component losses, gradient norms, optimizer-update evidence, and final parameters at stable
  boundaries. A scalar total loss alone is insufficient to localize a mismatch.
- Each prediction and benchmark sample has exactly one result owner. Non-owner ranks contribute zero to metric counts and
  do not write duplicate records.
- Compare one changed factor at a time before testing combinations. For a high-performance module plus parallelism, run the
  full 2x2 matrix: reference/non-parallel, optimized/non-parallel, reference/parallel, optimized/parallel.
- A performance claim is gated by correctness. Failed or unresolved parity cannot be offset by higher throughput.
- Resume validation compares uninterrupted and save/resume runs from the same checkpoint boundary, including optimizer,
  scheduler, RNG, dataloader position, and parallel metadata.

## Canonical Workflows

- Baseline, component, parallel, and composition parity:
  `.agent/skills/accuracy-validation/SKILL.md`.
- Quantitative tolerances and zero-centered loss-delta rules:
  `.agent/skills/accuracy-validation/references/tolerance-policy.md`.
- OpenCompass and VLMEvalKit execution:
  `.agent/skills/benchmark-evaluation/SKILL.md`.
- Model-family TP/CP/EP adaptation:
  `.agent/skills/parallel-adaptation/SKILL.md`.
- High-performance module lifecycle and default-on policy:
  `.agent/skills/performance-module-dev/SKILL.md`.

## Gate Ownership

- `model-dev-verifier` executes the model-specific evidence matrix before merge and returns PASS, FAIL, or BLOCKED for
  accuracy, benchmark, parallel-adaptation, and performance gates.
- `code-verifier` remains responsible for general style, lint, unit-test, and cross-platform verification.
- `gate-doctor` diagnoses and, when fixes are authorized, repairs failures reported by the GitCode PR gate.
- Remote CI is the source of truth for merge status. An agent may reproduce or diagnose its checks, but must not claim a
  missing, skipped, or inaccessible CI result is green.

## Reporting

Every validation report states PASS, FAIL, or BLOCKED per evidence level. Include the manifest path, commands, actual
hardware/topology, compared steps/samples, tolerance tier, numerical diagnostics, performance statistics, and remaining
unvalidated combinations. Never generalize one backend, dtype, model family, or topology to unsupported combinations.
