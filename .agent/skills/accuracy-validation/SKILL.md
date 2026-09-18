---
name: accuracy-validation
description: >
  Design, run, or review HyperParallel numerical-accuracy validation against
  VeOmni or an internal reference, including Trainer component A/B tests,
  FSDP/TP/CP/EP parity, loss-delta tolerances, resume checks, and optimized
  module plus parallel composition. Use for 精度、loss 对齐、baseline、parity.
---

# Accuracy Validation

Validate the smallest changed boundary first, then expand to a short training trajectory and capability benchmark.
Load `.agent/rules/model-development-validation.md` and the path-scoped distributed/testing rules before editing tests.

## Modes

Choose the smallest mode that covers the request:

| Mode | Reference | Candidate |
| --- | --- | --- |
| `veomni-baseline` | Pinned VeOmni run | HyperParallel Trainer with mapped semantics |
| `trainer-component` | HyperParallel reference path | One Trainer component toggled or replaced |
| `parallel-component` | Single-device or lower-dimensional topology | FSDP/HSDP, TP, CP, EP, then selected combinations |
| `performance-composition` | Reference module and/or non-parallel path | High-performance module, parallelism, and their interaction |
| `resume-parity` | Uninterrupted run | Save/resume at a fixed optimizer-step boundary |

## Required Inputs

- model/checkpoint and tokenizer or processor;
- exact data revision and sample IDs;
- baseline implementation and revision;
- config for precision, optimizer, scheduler, batch, sequence, packing, and mesh;
- target hardware and available rank count;
- changed component and expected mathematical equivalence;
- output directory for manifests and evidence.

Ask only when the reference or the intended precision tier cannot be derived.

## Workflow

1. Read [comparison-contract.md](references/comparison-contract.md) and freeze `run_manifest.json` before execution. For a
   first run, start from the executable example under `examples/accuracy/veomni_vs_hyper/`.
2. Read [tolerance-policy.md](references/tolerance-policy.md) and record one tolerance tier before seeing results.
3. Run structure and contract checks. Confirm parameter, batch, mask, token-count, and topology identity.
4. Run forward-only parity, then backward/gradient parity, then one optimizer update. Stop at the first mismatching
   boundary and retain its evidence.
5. Run at least 20 comparable optimizer steps for trajectory validation when the task changes training behavior. Save one
   loss record per optimizer step, not per rank or unnormalized micro-batch.
6. Compare the two loss files with:

   ```bash
   python3 .agent/skills/accuracy-validation/scripts/compare_loss.py \
     --baseline <baseline.jsonl> --candidate <candidate.jsonl> \
     --tier standard
   ```

   The example runner performs the same comparison after executing the baseline and candidate commands:

   ```bash
   python3 .agent/skills/accuracy-validation/scripts/run_accuracy_comparison.py \
     --manifest examples/accuracy/veomni_vs_hyper/run_manifest.demo.json \
     --output /tmp/hp-accuracy-demo
   ```

7. Run the applicable matrix. Do not skip the non-parallel optimized-module cell when testing an optimized module plus
   parallelism; otherwise the interaction cannot be isolated.
8. For user-facing capability claims, invoke `benchmark-evaluation` after numerical parity.
9. Write one report using [report-template.md](references/report-template.md). Mark unavailable multi-card evidence BLOCKED,
   not PASS.

## Validation Matrix

- Trainer components: batch adapter, model build/replacement, loss, optimizer, scheduler, checkpoint/resume, activation
  checkpoint/swap, callbacks, and metric aggregation. Toggle one component at a time.
- Parallel axes: non-parallel, FSDP/HSDP, TP, CP, EP; then pairwise interactions touched by the change and one flagship
  supported hybrid topology.
- Observables: batch fingerprint, valid-token count, forward outputs, total/component loss sums, gradients before and
  after synchronization, clip norm, optimizer delta, updated parameters, and checkpoint-restored state.
- MoE: add router logits/probabilities, top-k expert IDs, tokens per expert, dropped/padded tokens, aux loss, expert-owner
  mapping, and dense/expert gradient groups.

## Output

- State mode, reference, candidate, manifest, tolerance tier, and actual topology.
- Report each evidence level separately with commands and PASS/FAIL/BLOCKED.
- Include the first mismatching boundary, not only the final loss.
- List validated combinations and explicitly list untested backends, dtypes, models, and topologies.

## Read On Demand

- [comparison-contract.md](references/comparison-contract.md): frozen manifest, VeOmni mapping, component and topology matrix.
- [tolerance-policy.md](references/tolerance-policy.md): tensor, loss, gradient, and zero-axis trajectory thresholds.
- [report-template.md](references/report-template.md): canonical accuracy report.
