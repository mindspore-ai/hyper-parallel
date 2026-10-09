---
name: hf-model-precision-validation
description: >
  Validate a HyperParallel HF-native or custom-model integration with native
  and optimized-module parity, paired replacement performance experiments,
  deterministic full-state continuation, parallel-strategy self-consistency,
  recomputation-depth coverage, and
  Production/Validate comparison. Use for numerical acceptance and diagnosis.
---

# HF and Custom Model Precision Validation

Prove that the same model state and global batch produce aligned loss and FP32
global gradient norm across supported execution modes.

## Load

1. Read `AGENTS.md` and `.agent/rules/testing.md` before changing test harnesses.
2. Read [references/preflight-baseline.md](references/preflight-baseline.md) for
   resource acquisition, Offline data, and the full/cropped baseline branch.
3. Read [references/module-parity-and-recompute.md](references/module-parity-and-recompute.md)
   when optimized modules, custom architecture, shared cross-layer state, or
   activation checkpointing are in scope.
4. Read
   [references/optimized-replacement-experiments.md](references/optimized-replacement-experiments.md)
   when a replacement, fused kernel, CP, or EP implementation is claimed to
   improve performance.
5. Read [references/matrix-acceptance.md](references/matrix-acceptance.md) before
   launching distributed cases or deciding PASS/FAIL.
6. Read [references/diagnosis-report.md](references/diagnosis-report.md) only when
   diagnosing divergence or producing the final report.

## Required Precision Policy

Reject or explicitly correct a validation configuration that does not resolve to:

```yaml
model_init_dtype: float32

fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    cast_forward_inputs: false

optimizer:
  fp32_main_params: true
```

Confirm at runtime that the Trainer selected the main-parameter optimizer route,
main parameters are FP32, FSDP gradients reduce into FP32 main gradients, and
forward inputs are not automatically cast. Confirm `output_dtype` resolves to
`None` and the model's scalar loss remains FP32 by leaving the configuration key
absent.

For DTensor parameters, require `main_param`, `main_grad`, and optimizer moments
to retain one logical global shape, compatible mesh topology, and identical
placements on every rank, including ranks with an empty uneven FSDP shard.

## Operating Contract

- Formal cross-run validation uses deterministic, replayable inputs. Offline
  Indexed Dataset is the default. Online is acceptable only if deterministic
  sharding/replay and identical per-step input hashes are proven for every case;
  otherwise it is functional smoke only.
- Training launchers never download weights or datasets implicitly.
- If only a Hugging Face repo ID or URL is available, use the pinned, source-only
  Git checkout produced with ``GIT_LFS_SKIP_SMUDGE=1`` as the reference repository.
  LFS pointer checkout is allowed for source inspection; fetching weight payloads
  still requires a user-selected directory and explicit permission.
- Ask the user for the weight cache/download directory and download permission
  before fetching any weight payload.
- Missing or incomplete assets produce a clear `BLOCKED` result.
- Compare both loss and norm. Loss-only agreement is not a PASS.
- Keep learned MoE routing for formal cases. Forced routing is diagnostic only.
- Compare optimized/custom modules with the authoritative native implementation
  before using end-to-end loss to judge them.
- Treat numerical acceptance and performance acceptance as separate verdicts. An
  optimized replacement must pass authoritative module parity before its
  performance is considered. A precision PASS with no stable, same-topology
  reference/candidate experiment is `PERFORMANCE_INCONCLUSIVE`, not proof of a
  high-performance implementation.
- Do not set `region_dispatch: false` merely to run a model module's native
  forward as a Validate black box. If a native module cannot participate in
  DTensor propagation, report that module and the affected full-model Validate
  scope as unsupported; Production and native parity may still be validated.
  `region_dispatch: false` remains valid for an explicit communication/custom
  compute injection (`local_compute_fn` or `inner_wrapper`), but that local
  region cannot count as internal operator-dispatch coverage.
- Audit ownership before accepting the integration: model-family semantics belong
  to the model adapter. A new generic framework capability must be optional and
  model-independent; a framework-side family/model-type/class/FQN special case is
  a structural failure even when numerical tests pass.

## Workflow

1. Complete module parity first. Reuse its output directory for validation, or
   provide the framework-generated schema-v1 `integration_handoff.yaml` with
   `status: PASS`. Resolve and parse every referenced structure, module-parity,
   and checkpoint-coverage artifact before accepting a split-run handoff; state
   advancement without its evidence is not a valid gate. Keep the validation
   manifest limited to the model adapter, Trainer recipe,
   topology/recomputation/resume matrix, step count, and fixed tolerances.
   Architecture, data, precision, tokenizer, and model-owned assets belong to the
   Trainer recipe. Runtime environment, revisions, and hashes are collected as
   evidence; users do not duplicate them in validation YAML.
   Inspect the integration diff as part of this gate. If generic framework code
   recognizes the integrated family, return `FAIL` and require an adapter provider
   or a model-independent extension contract before running the expensive matrix.
   Before enabling any optimized replacement, run the production architecture's
   direct forward/backward and capture its parameter FQNs and initialization
   probes. Treat failure here as a model-integration failure; do not let a passing
   replacement hide an incomplete reference model.
2. Load `analysis/optimized_capability_inventory.json` and
   `analysis/optimization_plan.yaml` from the integration handoff. For every
   claimed optimized replacement, CP path, and EP path, run the declared
   reference/candidate module parity and target-accelerator microbenchmark. Then
   run a same-topology, same-state, same-input Trainer A/B using separate reference
   and optimized recipes or plan overrides. Only the implementation selection may
   differ. Record results in `analysis/optimization_results.json`. Keep CP/EP
   scaling experiments separate from implementation A/B: changing degree or
   global workload invalidates a replacement speedup ratio.
3. Generate the model-declared minimum matrix before spending devices:

   ```bash
   python -m hyper_parallel.tools.model_integration validate \
     --manifest <validation.yaml> --generate-only
   ```

   Inspect `cases/resolved_cases.json`. It must include EP1 when EP is exercised,
   requested TP/CP/EP combinations, adapter-safe `layer_count` and/or
   `layer_indices` recomputation selections,
   Production/Validate when requested, and explicit resume cases.
   Before accepting the generated cases, write a coverage ledger from the final
   model and adapter capabilities. Give every supported parallel axis, meaningful
   hybrid topology, recomputation depth, resume mode, execution mode, and
   multimodal subtree exactly one disposition: a formal case or an evidence-backed
   exclusion. Count baseline, strategy generalization, recomputation, resume, and
   Production/Validate separately. Resume and recomputation cases never increase
   the strategy-generalization count. Stop before device execution when the
   resolved counts or topology list do not match the ledger.
   Then run the pre-execution feasibility gate from `preflight-baseline.md`: check
   crop semantic/cardinality invariants and every topology divisor; resolve the
   launcher-default world size, mesh, and global-batch divisor; reconcile exact
   final-tree plan matches; assert activation wrapper FQNs; and require one
   high-risk runtime-evidence smoke. Multimodal scope additionally requires real
   media plus vision/projector gradients at the default world size.
4. Declare `launcher.module` and `launcher.config` for the standard Trainer path.
   The tool derives ordinary and prepare/restore commands from the matrix and
   configures every case with `model_integration.mode: runtime`. The matrix launcher
   owns the isolated evidence directory and passes it through
   `HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR`; output location and probe
   selection are framework policy, not additional Trainer YAML knobs. Runtime
   environment is inherited from the invoking shell rather than duplicated in
   the validation manifest. Give every case phase an isolated evidence directory
   and rendezvous/HCCL port range. A retry starts only after all child ranks from
   the preceding attempt have exited.
5. Execute the matrix through the same command without `--generate-only`. The
   framework records resolved config/environment, canonical global input identity,
   final parameter ownership, FP32 main-gradient/update probes, optimizer and DCP
   layout, shared-state contracts, activation-checkpoint wrappers, step time,
   throughput, and peak allocated/reserved memory. It compares JSON evidence rather
   than parsing Trainer logs.

   ```bash
   python -m hyper_parallel.tools.model_integration validate \
     --manifest <validation.yaml>
   ```

   For cropped scratch models whose sharded meta initialization can consume a
   topology-dependent RNG stream, require one common full-state DCP at the declared
   initial or warm-start step and restore the formal baseline plus every candidate
   from it. The standard launcher currently creates this checkpoint after step 1,
   so evidence and documentation must call it a step-1 warm start rather than
   step 0. A shared seed alone is not state identity. Before the full matrix, smoke one nonuniform ownership
   case such as an intermediate EP degree: diagnostic collectives must execute
   in the same sequence on every rank even when optimizer state is lazily absent.
   The standard shared warm-start deliberately restores model/optimizer/scheduler/
   RNG but requests replay from the configured data start; it does not restore the
   initialize step's cursor. Starting from the same configuration is not proof of
   identical input when rank sharding or packing depends on the DP world size, so
   require equal global input hashes before any numerical comparison. Preflight
   true K-to-K+N resume separately: a stateful Online loader may reject a restored
   cursor when the DP world size changes. Use topology-neutral fixed Offline replay
   or a loader with a proven global-cursor reshard contract; otherwise mark that
   resume case `BLOCKED`.
   Never drop dataloader state and still claim K-to-K+N continuation.

6. For a failure, diagnose the first causal failed gate: preflight/state identity,
   input identity, scalar curve, parameter probe, semantic/shared-state probe, or
   checkpoint continuation. Treat later transport/bind/timeout failures as
   cascading until process and port cleanup is proven. LR=0 or forced routing may
   isolate a cause but remains diagnostic-only and never replaces the formal case.
7. Generate the final auditable report:

   ```bash
   python -m hyper_parallel.tools.model_integration report --output-dir <evidence-dir>
   ```

   The command writes detailed `summary.md` (English) and `summary.zh-CN.md`
   (Chinese). Both must retain the machine-readable status on line one and report
   category counts, every topology and threshold, compared steps, maximum loss and
   norm deltas, input/LR identity, parameter probes, checkpoint layout, performance,
   memory, exact-callable operator support, actual multimodal-subtree placements,
   the resolved/open issue ledger, and exclusions. Accept only a first-line `PASS`.
   Before accepting either report, require evidence dependency closure and
   distinguish measured numeric, exact categorical, non-applicable, and missing
   evidence. An absent numeric field on a structural optimizer-state record is not
   a non-finite tensor; a required missing artifact is `BLOCKED`, never `n/a`.
   Missing parity, missing
   matrix artifacts, absent
   strict parameter probes, unsupported operators without an independent eager
   oracle, and unavailable assets/devices remain `BLOCKED` or `FAIL` as reported.
   Reference the optimization inventory, plan, and results from a bilingual
   optimization appendix. Preserve the CLI-generated reports when the current
   renderer does not ingest those artifacts; write adjacent
   `optimization-summary.md` and `optimization-summary.zh-CN.md` instead. State
   precision and performance verdicts independently for every optimized target.
   The built-in matrix's descriptive throughput is not a substitute for the paired
   optimization experiment.

## Stop Conditions

Stop and report `BLOCKED` instead of changing the experiment when required
weights/data are missing, available devices cannot realize the requested mesh,
or repeated OOM remains after the user-authorized memory mitigations. Do not
silently reduce sequence length, layers, topology, or model precision.
