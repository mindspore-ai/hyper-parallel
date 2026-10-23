---
name: codegen-precision-validation
description: >
  Run the frozen codegen vs native-trainer dual-arm precision validation for
  Qwen3-MoE and DeepSeek-V4.1 on an 8-NPU host: fresh artifact generation from
  the repository-recommended recipe, 10-step training boot from the generated
  module, and per-step loss alignment against the identical native arm. Use
  when validating codegen artifact correctness, re-running the acceptance
  flow after codegen changes, or diagnosing codegen/native divergence.
---

# Codegen vs Native Trainer Precision Validation

Prove that the codegen `gen` backend produces an artifact equivalent to the
native trainer path: same recipe, same 10 training steps, aligned per-step
`training/total_loss` under seed 42 + full determinism.

The frozen flow (single source of truth) lives in
[examples/training_demo/codegen_validation/](../../../examples/training_demo/codegen_validation/):
`run_codegen_validation.sh` plus the native-arm YAML templates and
[README.md](../../../examples/training_demo/codegen_validation/README.md)
(full variable/contract reference).

## Load

1. Read `AGENTS.md` (ambient law).
2. Read the full contract:
   [examples/training_demo/codegen_validation/README.md](../../../examples/training_demo/codegen_validation/README.md).

## When to use

- After codegen pipeline changes (generator, signature, loader, artifact
  layout) — re-run the acceptance flow on the 8-NPU validation host.
- When a model YAML + `codegen: true` must be proven numerically equivalent
  to the native path.
- Not for model-integration acceptance that does not involve codegen — that
  is `hf-model-precision-validation`.

## Procedure

1. **Preconditions** (defaults target the 8x Ascend 910B3 validation host):
   - Checkout of the commit under validation, 8 free NPUs.
   - Cropped model assets on the host (`QWEN_CONFIG`, `DSV41_CONFIG`).
   - Override `PYTHON` / `TORCHRUN` / `VAL_ROOT` / `TREE_ROOT` via env if the
     host differs; all variables are listed in the README.
2. **Run** (select models with `RUN_QWEN` / `RUN_DSV41`):

   ```bash
   bash examples/training_demo/codegen_validation/run_codegen_validation.sh
   ```

3. **Interpret the report**: every check prints `PASS`/`FAIL`; final
   `RESULT: PASSED|FAILED` mirrors the exit code. Logs and the artifact
   bundle land in `$VAL_ROOT/logs/` and `$VAL_ROOT/<model>/generated/`.

## Acceptance criteria (all three parts, per model)

- **Part 1 — fresh generation**: log lines `codegen: generating artifact` →
  `codegen: imported generated modeling file` → `codegen: sharding applied
  by the generated module`; `generated/` contains `modeling_*_gen_npu.py` +
  `codegen_meta.json`. A `reusing` line instead of `generating` means the
  preflight failed to force a fresh bundle.
- **Part 2 — boot from generated module**: codegen arm completes 10 steps
  with a complete loss progression.
- **Part 3 — dual-arm alignment**: native arm (identical YAML minus
  `codegen: true`, asserted by diff) shows zero `codegen:` lines and the
  per-step `training/total_loss` abs diff stays within `LOSS_TOL` (default
  5e-4; reference run recorded 0.0 for both models).

## Failure triage

- Criteria lines missing / `reusing` → check `CLEAN_GENERATED` and
  `$VAL_ROOT/<model>/generated/` state before the run.
- Part 1/2 pass but Part 3 diverges → compare
  `generated/modeling_*_gen_npu.py.diff` against the source modeling module
  to narrow down which injected rule changed numerics.
- Import-path preflight fails → the script requires `hyper_parallel` to
  import from `TREE_ROOT` when run from that cwd; do not rely on a stale
  editable install elsewhere on the host.

## Out of scope

- Model integration / numerical acceptance without codegen →
  `hf-model-precision-validation`.
- Unit tests and distributed ST → `.agent/rules/testing.md`.
- Commit / PR → **autogit**.
