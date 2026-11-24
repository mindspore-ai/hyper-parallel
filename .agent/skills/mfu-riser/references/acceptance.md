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
   echoing an expected remote digest is not verification. Include imported
   adapter modules and their recipe files, and verify the resolved factory
   before launch when the job adds adapter-specific overrides.
3. Match input tokens/hashes, actual batches and sequence lengths, masks, and the
   declared routing mode. Keep global tokens and useful model FLOPs unchanged and audit MFU independently
   per step. Exclude padding, redundant fused work, and recomputation from MFU. Device peak
   throughput comes from an external specification or an explicit task parameter, not
   a guessed script default.
   For ordinary causal attention, count `sum(L * (L + 1) / 2)` valid query--key pairs
   over the independent segments in each physical row. For sliding-window, sparse,
   or compressed attention, count the visible keys per query on each active layer.
   Count active MoE expert and router work, and audit recurrent blocks separately
   from softmax attention. When the objective supervises only part of a sequence,
   count the vocabulary projection work for the supervised positions. Use this same
   objective work for a reference that projects every row and an equivalent candidate
   that projects only the supervised subset.
   Check packing against separate rows at fixed tokens and model semantics. Before
   timing either path, test a masked cross-segment dependency against the reference
   at the actual attention call site. A boolean keep mask requires conversion
   when passed to an implementation that expects additive attention bias.
   Freeze this component breakdown before comparing candidates. A Trainer's
   full-square estimate can be retained for historical replay. Use the independent
   useful-work audit for the target when the two metrics differ. Recount the baseline
   and every candidate over their original windows when adopting a new convention.
   Across sequence or batch choices, report effective tokens/s/device and peak memory
   alongside the declared model FLOPs. Use a same-workload reference to attribute an
   implementation speedup.
   For multimodal packing, verify text sequence isolation independently of
   spatial rotary positions. Reconstruct image/video encoder and projector
   work from runtime patch grids and attention windows on every source rank.
   Compare against separate-sample outputs and gradients before timing.
   When eliminating a square mask, validate the runtime geometry and the
   backend's implicit causal contract before timing. A/B compares mask
   semantics at the real call site. Include a matched A/A trajectory when
   numerical variation can affect the gate. Keep the original verdict if a
   later replay adds evidence at a different scope.
   If a candidate introduces derived projection or layout metadata, compare
   all logical tensors at the last common boundary, including media values.
   In trajectory mode, hash the public batch and loss inputs before a
   model-integrated loss prepares its execution arguments. Verify the actual
   forward fields, operator shapes, and rank coverage independently. For
   adapters that derive metadata earlier, compare the logical tensors at a
   shared boundary before either adapter derives the fields.
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
   Record device occupancy at admission and verify isolation during the run with a
   lightweight out-of-band check. A process outside the trial appearing on a measured
   device invalidates that run's resource conditions. Retain its logs and repeat on
   isolated resources. Cancel only workers whose ownership by the trial is verified.

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

For projection packing, check the optimizer on the original logical parameter
boundaries. A fused Gate/Up projection can preserve forward outputs and gradients
while changing the matrix used by Muon. Compare an update starting from the same
weights, gradients, and momentum. When the reference orthogonalizes the two
projections independently, preserve both their Newton--Schulz inputs and RMS
scales. If the runtime provides a logical-matrix view hook, verify that it is
invoked for the packed weights. Check the resulting updates after unpacking,
then run the original model-level gate.

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
