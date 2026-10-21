---
name: formal-verification
description: >
  Use the official FM-Agent to find correctness bugs in distributed training frameworks.
  Defaults to HyperParallel contracts for DTensor, FSDP, pipeline, activation memory,
  tensor parallelism and checkpoints. Use for specification-driven bug hunting and
  verification of training changes; ordinary style review uses code-review.
---

# Training framework formal-methods agent

Reuse [FM-Agent](https://github.com/fmagent-project/FM-Agent) for specification generation,
compositional Hoare-style reasoning, candidate diagnosis and executable bug validation.
The adapter configures its public CLI; it does not implement a second reasoning engine.
LLM reasoning is evidence for investigation, not a machine-checked proof.

## Workflow

1. Select a narrow profile from [profiles.json](profiles.json) and inspect its offline plan:

   ```bash
   python .agent/skills/formal-verification/scripts/verify.py plan --profile dtensor
   ```

2. Read [training-invariants.md](references/training-invariants.md) and, for HyperParallel,
   [hyperparallel.md](references/hyperparallel.md). These supply training-specific questions;
   the current implementation, callers, tests and repository rules determine the actual contract.
3. Follow [engine-setup.md](references/engine-setup.md) for the official checkout and its model backend.
   `doctor` checks the real engine's dependencies and options. `run --mode estimate` invokes
   the official estimate without LLM calls; `plan` and `run --dry-run` do not invoke the engine.
4. Run `--mode verify`. The adapter always adds `--isolate --all-bugs`, the configured domain
   knowledge and [bug-validator.md](references/bug-validator.md). Keep the scope and mode identical
   when resuming. For verification after code changes, use `--incremental INTENT_FILE`.
   Use `--mode spec` when only specifications are wanted; it does not verify bugs.
5. Inspect the original `fm_agent/report.html`, `logic_verification_results/`,
   `bug_validation/summary.json` and per-candidate result/probe files. Inspect pending and
   failed candidates as well as confirmed ones. Neither a zero process exit code nor a
   `MATCH` proves correctness. Missing artifacts mean the run did not complete.
6. For each finding, report the source/function, contract and its provenance, trigger,
   expected/actual behavior, reproduction command/output, framework revision and limitations.
   A confirmed framework bug needs an executed probe against the actual implementation.
   Transfer a confirmed probe into the appropriate regression test when fixing that bug;
   do not alter training source merely to make a probe reproduce.

The framework checkout needs an editable install in the **probe** environment. The FM-Agent
engine can use a separate Python environment. Distributed probes follow the repository's
launcher conventions and must record rank count, backend, timeout and per-rank observations.
CPU mocks can check a calling contract; they do not confirm device-stream behavior.

## Scope and portability

`--submodule DIR [DIR ...]` replaces the profile's source directories. Keep required callees
in scope or document them as assumptions. Do not silently broaden a narrow request to the
whole repository. `training-core` is an explicit larger HyperParallel scope.

For another training framework, supply `--repo ROOT --config CONFIG.json --profile NAME`.
The [configuration example](references/engine-setup.md#other-training-frameworks) explains
how to replace source paths and framework knowledge. The launcher never imports HyperParallel
or PyTorch, and does not change runtime dependencies or training entry points.

If the official engine or model authentication is unavailable, provide the plan and precise
setup failure. Manual contract review remains useful but must be labelled manual; do not
present it as an FM-Agent run. Installation of this skill in a harness is harness-specific;
the repository's agent and command entries also provide explicit discovery.
