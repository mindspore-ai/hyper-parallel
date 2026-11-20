# Reusing the training acceptance protocol

Reuse the validation layers established by the Qwen3 MFU confirmation runs. Preserve
the method when adapting another model, rather than hard-coding model-specific constants.

1. Every node launcher must exit successfully and every rank must finish the planned
   steps. Verify the loaded checkpoint or reproducibly initialized parameters on all ranks.
   Across layouts, verify logical parameter identity and equal replicated shards;
   the same random seed is insufficient. See the
   [initialization and regression procedure](../../perf-playbook/references/failure-repair.md).
2. Freeze source, configuration, data, and the actual environment. Before importing
   training modules, each worker computes the declared source-manifest digest and records
   its editable-install/import path. Independently compute and compare the actual digest;
   echoing an expected remote digest is not verification.
3. Match input tokens/hashes, actual batches and sequence lengths, masks, and the
   declared routing mode. Keep global tokens and useful model FLOPs unchanged and audit MFU independently
   per step. Exclude padding, redundant fused work, and recomputation from MFU. Device peak
   throughput comes from an external specification or an explicit task parameter, not
   a guessed script default.
4. Check finite loss and gradients and compare input, loss, and gradient-norm trajectories.
   Check bitwise equivalence directly where applicable. When fusion changes rounding,
   declare tolerances in advance, validate module forward outputs and all input/parameter
   gradients, then check whole-model scalar trajectories. **Matching loss and gradient
   norms does not prove elementwise gradient equality.** Use [precision](../../precision/SKILL.md)
   or [hf-model-precision-validation](../../hf-model-precision-validation/SKILL.md).
   For MoE dispatch/combine, also use [routing-validator](../../routing-validator/SKILL.md).
5. Prefer hardware counters for HFU. Account for the executed recomputation, optimizer,
   and other arithmetic on the declared devices. Record step coverage and use the
   matching unprofiled times. Label estimates and state their derivation.
6. Positively verify the candidate path and declared shape coverage. Test the fallback
   when the candidate introduces or changes one.
7. Disable profiling during the formal benchmark. Fix the post-warmup window and use
   the maximum complete-step duration across ranks. Declare timing boundaries for the
   optimizer, input pipeline, synchronization, and logging. Use a separate short profile
   to explain the mechanism.

The earlier `analyze.py` validated rank-steps, weight loading, input hashes, FLOPs, Muon
invocation shapes, and recomputed layer counts. An adapted version can serve as this
controller's `validate` command while retaining raw JSONL validation. Do not copy a PASS
from an old `results.json` into a new trial. Configure model-specific loading markers,
FLOP accounting, and numerical checks instead of reusing Qwen parameter or layer counts.

Reuse the relevant checks on existing per-rank JSONL records with
`scripts/audit_rank_steps.py`.

```bash
python .agent/skills/mfu-riser/scripts/audit_rank_steps.py \
  --run-dir /path/to/current/raw --reference-dir /path/to/reference/raw \
  --expected-work /path/to/independent-work.json \
  --world-size 32 --steps 50 --peak 376 --window 11 50 \
  --out /path/to/trial/rank-audit.json
```

`independent-work.json` contains `derivation`, `source`, and
`samples: [{step, model_tokens, model_flops}, ...]` for all steps. Derive it independently
from the model architecture, weight shapes, and actual sequence lengths, never by
inverting recorded MFU. Exported samples can populate the final measurement.
`unverified` lists remaining checks such as weight loading, module gradients, engagement,
and environment. Scalar comparisons default to bitwise equality; declare any tolerance
before freezing the run configuration. Without a reference, `parity=null`.

For accumulated training, pass `--microbatches-per-update` with the count derived
from the frozen schedule. Each update's `input_hashes` list must contain one ordered
entry per microbatch, covering tokens, masks and positions; its meter `batches` must
have the same count. Both candidate and reference are checked even when their
truncated hashes match. Without the option, microbatch coverage remains explicitly
unverified. Count optimizer updates and microbatches separately in other adapters too.

## Screening and acceptance

For a performance-first task, set the validation level before running experiments.
Screening can use fewer post-warmup steps and a focused implementation gate. Test
the actual shapes, forward outputs, gradients, update semantics, and relevant
distributed reductions. Choose the scope according to the changed path.
Confirm workload identity, input validity, all-rank completion, and engagement in
every trial.

For acceptance, use the declared full numerical gate and the frozen screened
candidate. Cover equivalent parameter updates and model-level behavior with
predeclared tolerances. Capture fixed-routing and learned-routing evidence in
their declared workloads. Use learned routing to accept a learned training policy.

## Five-stage adapter

- `precheck`: Check resource availability, version/shape/backward support, numerical
  parity, fallback behavior, and candidate applicability.
- `benchmark`: Reuse the distributed launcher, wait for all remote tasks to exit, and
  record actual runtime and step evidence. Write artifacts to `RISER_TRIAL_DIR` with
  the unique trial token.
- `profile`: Run a separate short capture with the same source and configuration, then
  call `python .agent/skills/mfu-riser/scripts/collect_state.py --prof-dir ... --out ...`.
- `validate`: Check the raw evidence above and emit the contract's `measurement.json`.
  Failed validation must return a nonzero exit code, not just a warning.
- `cleanup`: Run after success or failure. Inspect and clean up only **this trial's**
  remote workers. Do not use broad `pkill` commands or terminate another user's training.
  Return zero when already stopped; return nonzero if cleanup cannot be confirmed.

The controller freezes argv and `validator_files`; it does not replace domain checks.
A JSON `passed=true` is not a mathematical proof. Include validators and their dependencies
in `validator_files`, and source and configuration files in `source_paths`. Preserve
independent check inputs, raw evidence, and results in the trial directory.

## Retention and stopping

Record `keep` only when MFU, computed from total model FLOPs divided by total time,
improves over the current best by more than the predeclared relative noise threshold.
A retained trial is a selected candidate. A screening result at the target leads to
a [linked acceptance series](contracts.md#linked-series). Target acceptance requires
an independent run with identical source and configuration that agrees within the
noise band. Both acceptance runs must meet the target. The controller records
confirmation variance outside the band as `unstable`.

For gains near the noise threshold, calibrate A/A variation and repeat within the
remaining budget. The v1 fixed noise band is not a statistical-significance test, and
50 adjacent steps are not 50 independent experiments.

Each `init` fixes the budget. Candidate, confirmation, and crashed trials share
`max_trials`, excluding the baseline. `max_seconds` is wall-clock time from initialization,
including planning. A timeout terminates the local process group and invokes job-specific
cleanup, with up to 30 additional seconds for resource release. Failed cleanup leaves a
pending trial and blocks further trials until `recover` succeeds. On stopping, retain
the best result without automatically publishing or overwriting code.
