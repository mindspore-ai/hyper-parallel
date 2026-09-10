---
name: hyper-rl
description: Entry rule for hyperparallel-RL-owned work.
paths:
  - hyper_parallel/rl/**
  - docs/rl-*.md
  - .agent/rules/hyper-rl.md
  - .agent/rules/rl/module-map.md
---

# hyperparallel-RL

Run RL commands from the repository root; `hyper_parallel/rl/` is a source root imported as `rl.*`. Open [`rl/module-map.md`](rl/module-map.md) only to locate ownership, and load navigation or product docs only when the change affects them.

## Readability

Human readability comes first; agent traceability is the minimum gate.

- Treat hard-to-follow code as a bug; simplify only when readability improves without changing required behavior.
- Add an abstraction only when it reduces reading cost or defines a real contract.
- Do not add configuration, abstractions, or compatibility branches for hypothetical requirements.
- Keep each fact in one authoritative place.
- Preserve features, defaults, performance knobs, and observability.
- Scope bug fixes to supported recipes and public extension contracts; do not expand a change for hypothetical problems in unsupported scenarios.

## Flow

Documentation is event-driven. Update the affected source of truth in the same change: this file when working policy changes, `rl/module-map.md` when subsystem ownership or paths change, `docs/rl-navigation.md` when a config/entry/branch/data/metric/test trace changes, and an existing product doc when its public contract, supported runtime behavior, operating procedure, or cross-project boundary changes. Internal refactoring that leaves those facts unchanged does not require a docs edit.

Create a new doc only for a stable contract or operating procedure with its own scope and maintenance lifecycle that does not fit an existing source of truth. Link it from `rl/module-map.md`. Do not create docs for one-off implementation plans, temporary validation results, or facts already owned elsewhere.

1. **Scope.** Define the RL-owned change and its affected behavior and contracts. For behavior changes, update the applicable config/entry → branch → data → metric → test trace in [`docs/rl-navigation.md`](../../docs/rl-navigation.md), using `—` where a field does not apply. Handle required main HyperParallel changes separately under that module's rules.
2. **Design.** For changes to feature scope, component boundaries, abstractions, or backends, consult [design goals and principles](../../hyper_parallel/rl/docs/design.md). Before changing code, present the approach, affected interfaces and edge cases, touched files, and test method. Wait for approval once.
3. **Implement.** Complete the scoped change and necessary tests under these rules and applicable module contracts. Keep the diff focused on the current problem; avoid unrelated refactoring.
4. **Finish.** For docs or agent rules, run catalog and link checks. For code, run affected tests from the repository root; broaden regression coverage for shared contracts, core flows, or uncertain impact. A full `rl_tests` run is not required for every local change; follow the applicable pre-merge gates and verify their coverage before relying on CI. Run an NPU gate only when the affected product doc requires it. Report changed files, results, and unavailable hardware as skipped.
