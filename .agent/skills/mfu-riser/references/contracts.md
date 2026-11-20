# Configuration, proposal, and evidence contracts v1

The Agent generates paths and commands for the current experiment. Replace
`/absolute/...` placeholders below with real files. The controller executes argv lists,
not shell strings. Put SSH, environment setup, or torchrun orchestration in a reviewed
script and invoke that script through argv. `validator_files` contains acceptance
implementations and dependencies; these are not candidate tuning targets.

## run.json

```json
{
  "schema": "mfu-riser/v1",
  "workload": {
    "model": "model architecture and config hash",
    "weights_id": "checkpoint manifest hash",
    "dataset_id": "dataset hash plus seed and ordering",
    "optimizer": "full optimizer configuration and hash",
    "precision": "bf16",
    "routing": "native",
    "global_batch_size": 32,
    "world_size": 32,
    "nodes": ["node-a", "node-b", "node-c", "node-d"],
    "peak_tflops_per_device": 376,
    "flops_convention": "PaLM full SxS; pinned counter version",
    "timing_scope": "synchronized complete train step, cross-rank MAX; exact log boundary declared",
    "train_steps": 50,
    "window": [11, 50]
  },
  "target_mfu_percent": 20,
  "max_trials": 6,
  "max_seconds": 7200,
  "command_timeout_seconds": 1200,
  "noise_fraction": 0.02,
  "validator_files": ["/absolute/adapter.py", "/absolute/check_numerics.py"],
  "commands": {
    "precheck": ["/absolute/python", "/absolute/adapter.py", "precheck"],
    "benchmark": ["/absolute/python", "/absolute/adapter.py", "benchmark"],
    "profile": ["/absolute/python", "/absolute/adapter.py", "profile"],
    "validate": ["/absolute/python", "/absolute/adapter.py", "validate"],
    "cleanup": ["/absolute/python", "/absolute/adapter.py", "cleanup"]
  }
}
```

376, 20, and 2% are illustrative values. They do not establish the hardware peak,
achievable target, or actual noise. Calibrate noise with same-node A/A measurements
or existing comparable records. Additional workload fields are allowed; every field
is compared.

## Performance policy

Add `performance_policy` to the run configuration when using the S4 performance workflow.
Add `workload.model_config_sha256` with the digest of the pinned model definition and
include every tunable field in the initial workload. The following example permits
sequence length and global batch changes.

```json
{
  "validation_level": "screening",
  "low_mfu_threshold_percent": 10,
  "tunable_workload_fields": ["sequence_length", "global_batch_size"],
  "flops_accounting": {"method": "architecture_aware"}
}
```

- `screening` uses a declared post-warmup window and an implementation correctness
  check. `acceptance` requires at least 20 steady-state steps and the full numerical
  check. A screening run can retain a candidate for further testing.
- Omit the threshold when the task has no low-MFU policy. Below the configured threshold,
  `status` returns `plan_from_critical_path` and the target-to-best throughput multiplier
  for the current workload. The planner sizes candidate gains from measured exposed time.
- List only workload fields the task leaves open. Supported fields are `sequence_length`,
  `micro_batch_size`, `global_batch_size`, and `gradient_accumulation_steps`.
- Use `{"method": "approximation", "max_relative_error": 0.02}` with a declared
  relative error limit and per-step reference calculations. The controller checks
  the approximation against the architecture-aware breakdown.

## Linked series

Initialize a capacity or acceptance series from the current validated parent result.

```bash
python .agent/skills/mfu-riser/scripts/mfu_riser.py init \
  --run-dir /path/to/new-series --config /path/to/child.json \
  --parent-run /path/to/parent-series
```

A capacity series changes only fields declared tunable in the parent. It keeps the
model definition, initial state, data identity, routing, hardware, and accounting
convention. Measure a new baseline and compare later trials against its own best.

An acceptance series changes `validation_level` from `screening` to `acceptance`
and can extend `train_steps` and `window`. Its first trial replays the screened source
manifest and inherits its engagement checks. Run an independent confirmation after
acceptance. Both transitions record the parent run, trial, result digest, and source
digest. They begin with an empty baseline and best record.

## plan.json

A baseline uses `kind=baseline, parent=null, actions=[]`, with `workspace` and
`source_paths`. The following candidate is hypothetical and does not assert that
RMSNorm is a hotspot in the current model.

```json
{
  "kind": "candidate",
  "parent": "000-existing-best-id",
  "workspace": "/absolute/isolated-candidate",
  "source_paths": ["hyper_parallel", "train.yaml", "launch.sh", "fusion_adapter.py"],
  "operator_scope": "compute_fusion",
  "actions": ["rms-norm-fusion"],
  "observation": "Operator, rank, step, and shape from state, with the source call site",
  "mechanism": "Reduce intermediate tensors and kernel launches in the normalization chain",
  "expected_effect": "Range justified by the critical path or a synchronized probe; probe first if unknown",
  "falsifier": "Fusion engages but neither the targeted interval nor complete-step time decreases",
  "compatibility": "Verified and pending dtype, layout, epsilon, backward, and version constraints",
  "evidence": ["/absolute/best-trial/profile/state.json"],
  "knowledge": ["/absolute/perf-playbook/references/ascend-fusion/index.md"]
}
```

With `performance_policy`, `operator_scope` can be `compute_fusion`,
`equivalent_implementation`, `parallel_layout`, `communication`,
`activation_memory`, or `optimizer`. The equivalent implementation scope includes
kernel and module replacements. Every scope preserves the frozen workload and
optimizer semantics. Describe the measured bottleneck, affected path, and expected
complete-step effect in the existing proposal fields.

`source_paths` includes all files or directories affecting measured code, configuration,
and launchers, relative to `workspace`. Exclude output directories; otherwise writing
training logs would trigger source-change detection. Freeze more than the edited file.
Use a worktree, independent checkout, or copied directory for isolation. Explicit feature
switches in a shared directory are supported, but after failure the Agent must resume
from the configuration referenced by `best`. The controller does not revert user files.

A confirmation uses `kind=confirm`, with `parent` pointing to the current best and identical
`workspace`, `source_paths`, and `actions`. The controller requires an identical source
manifest digest; confirmation cannot include new changes.

## Runtime environment

All five stages receive `RISER_TRIAL_DIR` (a unique absolute directory),
`RISER_TRIAL_TOKEN`, `RISER_CONTEXT` (`context.json`), `RISER_SOURCE_DIGEST`, and
`RISER_WORKSPACE`. Context records the configuration, plan, source manifest, and reference
trial. The adapter must not present these expected values as worker measurements.
Workers compute and record source hashes and actual workloads independently. Result files
must originate from this trial. Validation reads raw evidence and exits nonzero on failure.

## measurement.json

The validator writes `RISER_TRIAL_DIR/measurement.json`.

```text
schema: "mfu-riser-measurement/v1"
trial_token: current trial identity; raw worker records must carry the same token
workload: actual configuration, exactly matching run.json workload
profiler_during_benchmark: false
runtime: [{rank: 0, source_digest: independently measured worker hash, steps_completed: 50}, ...all ranks]
samples: [{step, time_s, model_tokens, model_flops, mfu_percent}, ...complete fixed window]
checks: {
  real_weights: {passed: true, evidence: "audit/weights.json", sha256: "..."},
  all_ranks_complete: same structure,
  finite_loss_grad: same structure,
  input_equivalence: same structure,
  numerical_equivalence: same structure,
  independent_flops: same structure,
  timing_boundary: same structure,
  fusion_active: same structure, required when actions are present,
  fallback_correct: same structure, required when actions are present
}
profile_state: "profile/state.json"
```

All evidence paths must be inside the current trial, and each `sha256` must match its
raw check report. Baseline input and numerical checks validate the reference
implementation against the declared data. Candidate checks compare with the frozen
reference path. Nonempty or finite outputs alone cannot establish
`numerical_equivalence=true`.

`samples` contains **one global record per step**. Do not sum a repeated global value
from every rank. `time_s` is the maximum complete-step duration across ranks.
`model_tokens` is the actual global non-padding token count. `model_flops` is independently
validated global useful model work. The controller checks each step's MFU and requires
candidate tokens and model FLOPs to match the reference step by step. Local kernel
execution FLOPs are not model FLOPs. Use actual lengths for variable-length steps;
do not assume every step has length 4096.

```text
Per-step MFU = model_flops / (time_s × world_size × peak_tflops_per_device × 10^12) × 100
Window MFU = sum(model_flops) / (sum(time_s) × world_size × peak × 10^12) × 100
Window throughput = sum(model_tokens) / sum(time_s)
Throughput per device = window throughput / world_size
```

For a performance-policy trial, set `validation_level` to the frozen level. Include
a `model_semantics` check for the model definition, input/mask semantics, layout,
logical Top-K, gradient reduction, and parameter update. Use
`implementation_correctness` in screening and `numerical_equivalence` in acceptance.
Fusion changes use `fusion_active` and `fallback_correct`. Other candidate scopes
use `candidate_active` with proof from the corresponding runtime path.

The `independent_flops` check points to a JSON audit from this trial. Its metadata
contains `trial_token`, `source_digest`, `model_config_sha256`, and `convention`
matching `workload.flops_convention`. Its `steps` list matches the measurement
window, with `step`, `model_tokens`, and `components` on each row. Components
map operation classes such as routed expert GEMMs and sparse attention to
independently derived positive useful FLOPs. The controller sums the components
for `architecture_aware`. For `approximation`, it compares each row's
`approximate_model_flops` with that sum and enforces the declared error limit.
Supply the architecture derivation and shape evidence to the frozen validator.

Optional `executed_work` has `method` (`hardware_counters` or `estimate`),
`coverage`, `evidence`, `sha256`, and `samples` containing `step` and
`executed_flops`. Each step must belong to the unprofiled window and appear
once. The validator checks that the counts cover the declared device set and
configuration. The controller uses the matching unprofiled step times for HFU
and records the sampled steps, coverage, and method.

Optional `allocation` has `device_seconds`, `evidence`, and `sha256`. Derive
device seconds from actual job allocations. Trial results also record wall time
for each controller stage, including unsuccessful stages. Stage wall time
includes any wait in the launcher.

## Profiling state and existing skills

Run `scripts/collect_state.py --prof-dir CAPTURE --out NEW_DIRECTORY`, optionally with
`--prior` and `--memory` JSON outputs from existing skills. The collector invokes the
profiling `breakdown.py` directly, preserves its checks and provenance, and adds actual
shape summaries from `kernel_details.csv`. It does not alter the profiling formulas,
infer numbers from HTML, or invent unknown shapes and dtypes.

A `valid=false` result prevents trial promotion. Declare captured ranks and phases for
single-device, single-node, and multi-node runs. The collector validates supplied reports;
it does not prove global-rank coverage. The adapter must cover representative and slow
ranks for the topology, especially PP and EP, and record that coverage in
`timing_boundary` evidence.

Controller-bound captures also require
`--capture-metadata PROFILE_WORKER_METADATA.json`. The measured profile worker writes
its actual trial token and source digest to this file, and the collector preserves them
in `measurement_binding`. Older traces without this binding remain usable for offline
analysis but cannot promote a new trial. The frozen validation adapter checks input and
workload consistency between profiling and benchmarking.

When timing exists but shapes are missing, state retains `partial_or_unknown`.
The planner must collect `record_shapes` or an independent shape probe before selecting
a fused implementation.
