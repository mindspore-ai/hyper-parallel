---
name: hyper-rl
description: Entry rule for Hyper-RL-owned work.
paths:
  - hyper_parallel/rl/**
  - hyper_parallel/models/qwen3/**
  - tests/ut/rl/**
  - tests/torch/rl/**
  - tests/common/rl_st_cases.py
  - docs/rl-*.md
  - .agent/rules/hyper-rl.md
  - .agent/rules/rl/module-map.md
---

# Hyper-RL

Start here for RL code, tests, docs, and agent rules. Use the [module map](rl/module-map.md) to locate ownership,
[architecture](../../docs/rl-architecture.md) for project boundaries, and
[feature navigation](../../docs/rl-navigation.md) for configuration, implementation, metrics, and representative tests.
Load the affected product contract rather than every RL document.

## Scope and runtime

- Run commands from the repository root. `hyper_parallel/rl/` is the source root imported as `rl.*`.
  Follow the [runtime installation guide](../../hyper_parallel/rl/docker/README.md) for dependencies
  and vLLM plugin registration; the root package installation alone is not a complete RL runtime.
- RL and `hyper_parallel/models/qwen3/` use native Torch APIs. Do not introduce Platform dispatch or `get_platform()`.
  Qwen3 adapters belong in `models/qwen3/`; main-project callers use the shared AutoModel builder.
  RL construction compatibility belongs in `hyper_parallel/rl/rl/roles/qwen3_builder.py`; value heads and Critic behavior
  belong in `hyper_parallel/rl/rl/roles/policy/critic.py`.
- The built-in algorithms are GRPO and PPO. Model registration accepts Qwen3 dense; training supports FSDP sharding
  with TP1/TP2 and `dp_replicate=cp=pp=ep=1`. Rollout uses shared vLLM in colocated or disjoint mode.
  Check `rl/config.py` and `rl/roles/model_setup.py` for the executable boundary; do not infer RL support from a main-project API.
- Weight synchronization selects `full_gather` or `direct_reshard`, with IPC for colocated and HCCL for disjoint.
  Publication errors propagate; there is no automatic fallback. Preserve this contract when fixing failures.
- Shared HyperParallel modules retain their own rules. For Qwen3 integration, also follow applicable model and
  distributed rules; do not apply RL policy to unrelated code or change shared contracts without examining other callers.

## Readability

Human readability comes first; agent traceability is the minimum gate.

- Treat hard-to-follow code as a bug; simplify only when readability improves without changing required behavior.
- Add an abstraction only when it reduces reading cost or defines a real contract.
- Do not add configuration, abstractions, or compatibility branches for hypothetical requirements.
- Keep each fact in one authoritative place.
- Preserve features, defaults, performance knobs, and observability.
- Scope bug fixes to supported recipes and public extension contracts; do not expand a change for hypothetical problems
  in unsupported scenarios.

## Flow

Documentation is event-driven. Update this file when working policy changes, [module-map.md](rl/module-map.md) when
ownership or paths change, [rl-architecture.md](../../docs/rl-architecture.md) when module boundaries change, and
[rl-navigation.md](../../docs/rl-navigation.md) when configuration, entry points, branches, data, metrics, or representative
tests change. Update the existing product document when its public contract, supported runtime behavior, operating
procedure, or cross-project boundary changes. Internal refactoring that leaves those facts unchanged needs no docs edit.

Create a new document only for a stable contract or operating procedure with its own scope and maintenance lifecycle
that does not fit an existing source of truth. Link it from the module map. Do not create docs for one-off implementation
plans, temporary validation results, or facts already owned elsewhere.

1. **Scope.** Define affected behavior and contracts; update the corresponding navigation rows, using `—` where a field
   does not apply. Treat required shared-project changes under their own module rules.
2. **Design.** For changes to feature scope, component boundaries, abstractions, or backends, consult
   [design goals and principles](../../hyper_parallel/rl/docs/design.md). Before changing code, present the approach,
   affected interfaces and edge cases, touched files, and test method. Wait for approval once.
3. **Implement.** Complete the scoped change and necessary tests. Keep the diff focused; avoid unrelated refactoring.
4. **Finish.** Apply the validation below and report changed files, results, and checks not run. Keep one commit per PR
   as required by [AGENTS.md](../../AGENTS.md); amend subsequent fixes without including unrelated local changes.

## Validation and test ownership

- UT lives in `tests/ut/rl/`; NPU ST and its launch helpers live in `tests/torch/rl/`.
  Shared recipe data and configuration construction live in `tests/common/rl_st_cases.py` so UT can run without the ST
  archive. Do not introduce a UT dependency on `tests/torch/`. The UT conftest also collects `agentic_ut.py`.
- Follow the repository [testing rules](testing.md) and [UT rules](unit-test.md). ST launchers must not import Torch,
  MindSpore, or HyperParallel during collection. Use the [UT guide](../../hyper_parallel/rl/docs/hyper_rl_ut.md) and
  [ST guide](../../hyper_parallel/rl/docs/hyper-rl-st.md) for execution commands and resources.
- For docs and agent rules, run `python3 .agent/scripts/check_agents_catalog.py`, Markdown lint, and checks for changed
  relative links and referenced paths. The catalog script only compares Skills/Agents tables against disk; it does not
  validate navigation symbols, config keys, metrics, or test semantics. Check those against code and tests separately.
- For code, run affected tests from the repository root. Broaden regression coverage for shared contracts, core flows,
  or uncertain impact; a full `tests/ut/rl/` run is not mandatory for every local edit.
- Run the applicable real-NPU acceptance when the changed product contract requires it. RL ST is `level1`/`allcards` and
  requires explicit resources; default PR CI success or mocked UT does not prove every RL recipe passed. Report missing
  hardware or resources as not run, and preserve learning, policy-version, and consistency assertions.
