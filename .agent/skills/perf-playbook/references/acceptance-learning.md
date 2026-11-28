# Evidence-based learning at training acceptance

**Evidence provenance:** Engineering procedure derived from training RSI trials.
An experiment's result is specific to its workload, execution path, and numerical
qualification. A short performance screen is not an accepted reusable result.

## Make the retrospective part of acceptance

This is required for every RSI task. Use the
[automatic consolidation tool and review contract](../../autoresearch/references/experience-consolidation.md).
Default to consolidation at acceptance; continuous consolidation is optional.
Both modes deliver a separate RSI improvement report. Search existing experience,
reuse or improve it before adding rules, and exclude deployment-specific facts.

Before closing a training optimization task, reconcile the complete candidate
ledger, including crashes, invalid measurements, numerical failures, noise-band
results, and deferred hypotheses. Preserve original evidence and thresholds.
For each lesson record:

- the triggering symptom, model semantics, workload, topology, and runtime;
- the causal hypothesis, changed source/configuration, and engagement evidence;
- the correctness protocol and its result, separately from measured performance;
- end-to-end benefit, memory and Host costs, measurement coverage, and uncertainty;
- applicability limits, counterexamples, and conditions that justify a retry;
- the Skill, Agent planning rule, or Harness check that should consume it.

Use explicit states: **verified**, **screened but unqualified**, **rejected**,
**inconclusive**, or **untried**. Expert advice starts as a hypothesis. Promote it
to a measured case only after evidence supports the claimed mechanism and scope.
Do not label a new threshold or exception as an existing Harness capability.

## Review interactions rather than accumulating switches

| Observation or change | Required review before reuse |
|---|---|
| A fused module changes the critical path | Re-profile the new parent before ranking the next candidates; old hotspot percentages may be obsolete |
| A native API appears unavailable | Check the complete supported API family, including paged, layout, dtype, and backward variants; verify the actual selected path |
| A cache flush was added during failure recovery | Determine what lifetime or synchronization condition made it necessary; distinguish inactive-cache release from active-data safety; check longer-run Host growth before removing it |
| A workload change reduces retained activations | Reassess checkpoint and offload scope; preserve the objective and local backward semantics; measure both memory and full-step time |
| A dense shard domain changes | Account for replica communication and optimizer ownership as well as AllGather; compare logical initial state, not only seed values or local-shard hashes |
| A locally faster collective gives little MFU improvement | Check whether communication was already hidden and whether the change increased an exposed dependency elsewhere |
| A candidate failed previously | Recheck the recorded parent, prefetch, residency, and runtime before claiming it was tested in the current combination |
| Selective retention is feasible | Compare full and selective checkpointing at matched work. Rank candidates by saved unprofiled step time per incremental GiB, inspect live tensor lifetimes, and remeasure the combined peak and allocator retries |
| HFU exceeds MFU | Attribute extra execution to recomputation, optimizer arithmetic, and other covered work; the percentage-point gap is not directly recoverable MFU |
| More devices are proposed | Recompute the peak denominator and topology costs; compare a separate resource series instead of implying that additional capacity guarantees higher utilization |

When replay repeats an expensive kernel in an otherwise memory-bound block,
identify the replayed invocation in the device timeline. Compare retaining just
its outputs with retaining the complete block. Preserve replay of surrounding
operators and the framework's saved-tensor, RNG and autograd guarantees.
Confirm that the targeted invocation is reused in replay, then check the
whole-model trajectory and measure a separate unprofiled training window.
Record the incremental peak allocation and reservation after optimizer state
initialization. Expand retention in measured increments because the second
increment may yield less speedup or increase allocation pressure.

## Turn lessons into executable safeguards

The Skill describes the diagnosis and applicability. The Agent records a specific
hypothesis, expected critical-path benefit, and falsifier. The Harness or its
workload adapter validates structured evidence before promotion. Candidate checks
should be scoped to the demonstrated failure mechanism, for example:

- reject a cross-layout numerical comparison without logical initialization
  identity and matching data/routing; rank-local external state must use its
  declared placement identity rather than being treated as a global replica;
- reject a diagnostic or profiled run as a clean performance measurement;
- require all allocated ranks, optimizer updates, objective phases, and intended
  implementation paths to complete;
- retain the originally failed gate when a separate matched-state diagnostic is
  added; report the new evidence's actual scope;
- report unsupported memory or FLOPs counters as unavailable, never as zero;
- require a separate final measurement and record allocator retries, sustained
  memory growth, and resource-release results when relevant to the intervention.

These are integration requirements, not a claim that every item is already
enforced by the generic Harness. Add a meaningful regression for each implemented
check: a representative valid receipt must pass and the original failure mode
must be rejected. Keep workload-specific initialization adapters and counter
normalization separate from model-independent candidate decisions.

## Acceptance package

Deliver the frozen candidate, comparison series, numerical qualification,
independent performance measurements, remaining limitations, and the reconciled
learning records together. Link every measured claim to its raw artifact and
source identity. Keep unverified suggestions visible in the next-experiment
queue; do not silently convert them into defaults.

See [training RSI](training-rsi.md), [failure repair](failure-repair.md),
[hierarchical diagnosis](hierarchical-performance-diagnosis.md), and
[memory analysis](../../memory-analysis/SKILL.md).
