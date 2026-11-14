# Training RSI protocol

**Source layer** · Field lessons from training bring-up and post-run
ablations. The procedure applies across models. Measure the size of each
effect on the current workload.

## Performance-driven optimization

When the task prioritizes end-to-end training performance, use the
[MFU Riser performance policy](../../mfu-riser/references/contracts.md#performance-policy)
with declared screening and acceptance gates.

- Audit model FLOPs using the active model architecture and actual sequence
  lengths. Use a simplified PaLM estimate when it agrees with the reference
  calculation within the declared tolerance. Include the accounting convention,
  MFU, HFU evidence coverage, effective tokens/s/device, and optimization costs
  in the report. See [utilization conventions](mfu-targets.md).
- For a low baseline, first verify the measurement and size the target gap.
  Use the [MFU Riser planner](../../mfu-riser/references/planner.md) to rank exposed
  bottlenecks and consider equivalent module replacements, operator fusion,
  communication scheduling, and parallel layout changes. A threshold such as
  10% is a task-specific trigger for this investigation.
- Preserve model depth, width, expert count, logical Top-K, required token
  work, and training semantics. Tune execution settings such as parallel layout,
  activation lifetime, prefetch, and optimizer implementation within that scope.
  Place tunable execution settings in the candidate source/configuration and
  place fixed task constraints in the workload.
- If the task permits sequence length, batch, or accumulation changes, measure
  a [linked capacity series](../../mfu-riser/references/contracts.md#linked-series)
  for each selected workload. Check supported lengths, peak memory, and gradient
  normalization. Carry forward the task's capacity requirements. Treat a memory
  reserve used for baseline planning as a constraint only if the task requires it.

When a material correctness or performance defect appears, follow the
[failure repair procedure](failure-repair.md). Preserve the original failing
workload, localize and repair its cause, and replay it without diagnostic probes.
A reduced workload or disabled path is a diagnostic, not evidence that the defect
is fixed. Verify logical initialization identity before cross-layout numerical
comparisons and audit every accumulated microbatch, not just optimizer updates.

## Freeze the trial

- Freeze the first passing independent adaptation as B2. Keep the accepted
  optimization A on its lineage and record the parent and candidate commits.
  When an independent reference B1 is available, declare the B2 improvement
  target and the B1-level performance target separately. Fix the budget and
  report each result and the remaining gap when the run ends.
- Fix the workload, hardware, routing, timed instrumentation, metric formula,
  per-device peak, warmup, and measurement window before comparison. Check
  process and device occupancy immediately before launch and confirm release
  afterward. Invalidate the affected measurement when there is contention.
- Keep a blind run's reference implementation, resolved recipes, and prior
  solutions out of its readable inputs. Score the frozen endpoint before
  unblinding. Record any later ablation as diagnosis.

## Separate correctness from performance

Run two windows on the same committed candidate. Use a correctness window
with the necessary input hashes, scalar checks, and tensor probes. Use a
performance window with matched instrumentation across candidates and
baselines. Prefer lightweight markers that verify the intended path was
engaged. Measure validation overhead separately when it matters.

Freeze the effective native execution mode as well as the requested configuration:
extension and operator-package digests, reduction determinism, and plan-selection
overrides can change the arithmetic path without changing the model commit. If
correctness and performance use different modes, declare and validate both; see
[native dispatch repeatability](failure-repair.md#verify-repeatability-through-native-dispatch).

Choose the gate before running a candidate:

| Change | Correctness evidence |
|---|---|
| Scheduling or equivalent implementation | Matched inputs and the declared numerical equivalence gate |
| Fused or different-precision kernels | Local parity and a full-model check against predeclared tolerances |
| Training policy such as optimizer, routing, or data semantics | A separately approved training-quality protocol with matched evaluation inputs and budget |

Use fixed expert routing for an authorized performance workload when route learning
or load balancing is outside the task. Declare the routing mode and preserve it
within each comparison series. Use native routing to validate native-routing
training quality.

## Structured training receipts

After scaffolding a fresh autoresearch run, add the following contract to
`run.json` and set separate `gate_cmd` and `benchmark_cmd` argv lists.

```json
{
  "training_protocol": {
    "workload_id": "sha256:replace-with-workload-digest",
    "gate_steps": 30,
    "benchmark_steps": 30
  }
}
```

Derive the series ID from the frozen data, shape, hardware and topology, and
the declared routing and measurement settings for both phases. Each wrapper
checks its resolved phase settings against that declaration before launch and
reports the series ID. The gate checks baseline self-consistency when recording
the baseline and candidate correctness on subsequent iterations.
Each step count includes warmup and counts completed optimizer updates.
Use a short screen separately. A candidate enters the retained trajectory after
the full gate and benchmark. Independently remeasure the frozen final candidate
for acceptance and record those measurements separately from candidate selection.

The wrapper emits exactly one final line with `RSI_TRAINING_RECEIPT`, a space
and a JSON object, including on failure when it can report one.

```text
RSI_TRAINING_RECEIPT {"workload_id":"sha256:replace-with-workload-digest","phase":"gate","status":"numerical_mismatch","completed_steps":17,"engaged":true,"first_error_step":18}
```

- `phase` is `gate` or `benchmark`. Set `engaged` from evidence that the
  intended path ran. A successful invocation reports `passed`, the exact
  planned number of completed steps, positive engagement and exit code zero.
  The benchmark also emits its normal metric line.
- Use `oom`, `numerical_mismatch`, `incomplete`, `cleanup_failed`,
  `contention`, `futile` or `error` for a known terminal failure.
  Include `first_error_step` and `allocation_retries` when observed.
- A futility watcher checks progress before signalling the process group.
  Once the window is complete, let the wrapper finish collecting evidence and
  releasing resources. If cleanup fails, record that separately from an
  incomplete window. A `futile` receipt at or beyond the final step is rejected.

The loop records job duration, exit code, window completion and failure class
in `candidates.jsonl` for the baseline and subsequent candidates. It requires
a passing baseline before iteration, freezes the contract for the series and
applies the same receipt checks to noise reruns. Legacy run directories can
continue using the existing marker protocol.

## Carry evidence across iterations

- Record which path ran, the first failing phase or step, the assertion or
  error, and the gate and performance windows. Retain evidence for recurring
  slow steps, resource contention, and skipped or cancelled trials. For an operator that passes
  local parity but fails while reading offloaded state, isolate the observer
  from the training path before reassessing the operator.
- Preserve deferred candidates with their dependency scopes in the
  [autoresearch ledger](../../autoresearch/SKILL.md#candidate-receipts-and-retries).
  Prefetch can be within noise on a slow configuration and become measurable
  after a compute or optimizer change. Re-measure it against the new parent
  when the declared scope is kept.
- When two changes may interact, measure each from the same parent and then
  measure their combination. Report single-change deltas separately from
  cumulative alignment deltas. Keep correctness and engagement gates on
  every candidate.
