# Training framework bug validation

Validate each FM-Agent candidate against the actual training framework and the supplied
domain knowledge. Preserve the bug ID, target result file, output paths, required JSON
fields and allowed status values specified by the invoking FM-Agent prompt. Its output
schema takes precedence over the default schema below.

## Contract and reproduction

1. Read the candidate's specification, code evidence and trigger. Check the original
   implementation, public API, callers and existing tests. Identify the intended contract
   independently of the alleged violation, and label any inferred assumptions.
2. Construct the smallest in-contract reproduction. Prefer a CPU numeric reference for
   shape/layout/gradient errors. Use the real implementation via normal imports; do not
   copy the allegedly buggy function into a standalone reimplementation, strip its imports
   or mock away the suspected failure. Record the imported module's resolved path.
3. For numeric training errors, use identical initial parameters, seeds, loss reduction,
   dtype and optimizer settings on the reference and parallel paths. Compare the relevant
   outputs, gradients and/or one optimizer step. Record the tolerance and its justification.
4. Use the framework's distributed launcher conventions for communication failures. Set a
   timeout, capture rank observations, and reap workers. A CPU mocked call-order test may
   support a contract finding, but cannot confirm accelerator races, memory reuse or NCCL/HCCL
   deadlocks. When required hardware is absent, record an unresolved validation with that reason.
5. Write and execute a self-contained probe under `fm_agent/bug_validation/`. Record its exact
   command, interpreter, exit code, stdout/stderr and expected/actual values. Limit attempted
   revisions to three unless the invoking prompt specifies a different budget.

Do not modify training source, existing tests, or logic-verification results during validation.
Do not run the entire test suite. Retain successful and unsuccessful probe scripts and raw
outputs so the result is independently reviewable. Keep checkpoint probe files inside the
probe workspace or a temporary directory and report their lifetime. Use the editable framework
checkout in the probe environment, which may differ from the FM-Agent engine interpreter.

## Classification

- `confirmed`: an executed probe against the real implementation demonstrates an in-contract
  violation with a documented oracle. A spec disagreement or speculative trace alone is insufficient.
- `not_confirmed`: executed probes do not demonstrate the claimed violation, or the candidate
  depends on an invalid contract/unsupported input. Describe what was tested and what remains open.
- `error`: validation could not execute reliably, including missing hardware/dependencies,
  import failures and probe errors. Never count these as confirmed bugs or evidence of correctness.

Use any additional statuses and terminal-state rules required by the invoking engine rather
than inventing a new result vocabulary. A collective timeout by itself is insufficient:
distinguish a framework scheduling bug from a failed runtime or worker startup.

## Original FM-Agent artifacts

Write `fm_agent/bug_validation/<bug_id>.md` with the specification claim, actual behavior,
source file/function/lines, trigger inputs, expected/actual output, executed probe, raw output,
revision and remaining assumptions. Preserve the source/function metadata from the candidate.

Unless overridden by the invoking prompt, also write `<bug_id>.result.json` with:

```json
{
  "id": "the provided bug ID",
  "source_file": "the candidate source file",
  "function_name": "the candidate function name",
  "confirmation_status": "confirmed or not_confirmed or error",
  "attempts": 1,
  "probe_script": "fm_agent/bug_validation/probe_<bug_id>.py",
  "detail_file": "fm_agent/bug_validation/<bug_id>.md",
  "probe_stdout": "raw executed probe stdout",
  "trigger_summary": "specific trigger and expected versus observed behavior"
}
```

Retain the engine's original artifacts. Propose a focused repository regression test for a
confirmed bug; actually transferring it into the test suite belongs to the subsequent fix.
