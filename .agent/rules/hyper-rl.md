---
name: hyper-rl
description: Entry rule for Hyper-RL-owned work.
paths:
  - hyper_parallel/rl/**
  - hyper_parallel/models/qwen3/**
  - tests/ut/rl/**
  - hyper_parallel/rl/tests/st/**
  - tests/common/rl_st_cases.py
  - docs/rl-*.md
  - .agent/rules/hyper-rl.md
  - .agent/rules/rl/module-map.md
---

# Hyper-RL

Start here for RL-owned code, tests and docs. Use the [module map](rl/module-map.md) for ownership,
[architecture](../../docs/rl-architecture.md) for boundaries, and
[feature navigation](../../docs/rl-navigation.md) for configuration, implementation and test traces.
Read the affected product contract rather than every RL document.

## Runtime and ownership

- Run from the repository root. `hyper_parallel/rl/` is imported as `rl.*`; follow the
  [runtime installation guide](../../hyper_parallel/rl/docker/README.md). Installing the main project alone is insufficient.
- Use native Torch APIs. Do not reintroduce Platform dispatch or `get_platform()`.
  Shared HyperParallel modules retain their own model, distributed and coding rules.
- Shared model loading belongs in `hyper_parallel/models/`; RL Qwen3 construction compatibility belongs in
  `rl/roles/qwen3_builder.py`, Critic/value-head integration in `rl/roles/policy/critic.py`, and
  RL-only probability/recomputation adapters in `rl/roles/policy/token_log_probs*.py`.
- `SyncTrainer` owns synchronous execution; `rl/async_training/` owns the independent rayless loop.
  Both reuse the role runtime and publication machinery. CPU preparation overlap is still synchronous RL.
- Supported model/algorithm/topology combinations are checked by `rl/config.py` and `rl/roles/model_setup.py`.
  Consult [support boundaries](../../hyper_parallel/rl/README.md#支持范围),
  [MoE contracts](../../hyper_parallel/rl/docs/moe_code_agent.md#功能与支持边界), and
  [Qwen3.5-family recipes](../../hyper_parallel/rl/examples/swe_smith/README.md#模型配方).
  Main-project capability, successful config parsing and real-device acceptance are different evidence.
- Task prompts, workspace contents and scoring stay in `examples/`. Shared Agent code handles contracts and execution,
  not task-name-specific reward branches. DeepSeek Harness is a program integration, not a supported model claim.
- Detailed experiment results, historical baselines and one-off analysis tools stay outside tracked source.
  Examples retain runnable recipes and concise result boundaries; tracked launchers/tests must not depend on `output/`.

## Contracts to preserve

- Preserve actual sampled tokens, masks, call/episode identity and policy versions. External Codex/DeepSeek programs
  train each call under its actual prompt with episode-level GRPO; segmented PPO is rejected.
  DP padding has zero loss and is excluded from episode statistics. Legacy continuous trajectories require exact prefixes.
- Keep sampling probabilities separate from Actor-recomputed anchors. Follow the configured `old_log_prob_source`;
  missing sampling probabilities must not be presented as a zero discrepancy. See the
  [training contract](../../hyper_parallel/rl/docs/architecture.md#一个训练步).
- Publication supports explicit `full_gather`/`direct_reshard` and IPC/HCCL according to deployment.
  Errors propagate without fallback. Asynchronous admission, lag, group consumption and recovery follow the
  [async contract](../../hyper_parallel/rl/docs/architecture.md#rayless-asynchronous-grpo), not synchronous version assumptions.
- In synchronous Agent execution, infrastructure/unknown failures reject the update; verified model failures may
  remain trainable. Do not infer budget exhaustion from an unexplained CLI exit. Preserve parser evidence and shutdown
  behavior described in the [Agent contract](../../hyper_parallel/rl/docs/agentic_rl.md).
- Private code tests and dataset labels remain outside agent-visible workspaces. Only the TP request owner executes
  internal environments. Repository tasks retain authenticated session relays, isolated candidate/grader workspaces
  and independent frozen-artifact grading. See [code](../../hyper_parallel/rl/examples/code/README.md) and
  [repository tasks](../../hyper_parallel/rl/examples/swe_smith/README.md).
- Optional Reward Model scoring keeps task scorers in examples; `rl/reward_model/` owns the frozen service.
  Rollout sleeps before RM wake; RM sleeps before training/publication. See the
  [RM contract](../../hyper_parallel/rl/docs/reward_model.md). Distillation has its own
  [teacher contract](../../hyper_parallel/rl/examples/distillation/README.md).

## Editing and validation

- Keep each fact in one authoritative document. Prefer readable, focused changes; preserve features and observability.
  Add abstractions only for an existing contract or real reuse, not hypothetical extensions.
- Update this rule for policy changes, the module map for ownership, architecture for boundaries, navigation for
  config/code/test traces, and product docs for public behavior. Internal refactoring does not require narrative history.
  Create a document only when a stable contract cannot fit an existing owner.
- For feature or boundary changes, consult [design goals](../../hyper_parallel/rl/docs/design.md), describe the approach,
  interfaces, edge cases, affected files and validation, and obtain approval once before implementing.
- UT belongs in `tests/ut/rl/`; ST in `hyper_parallel/rl/tests/st/`; shared recipes in `tests/common/rl_st_cases.py`.
  UT must not depend on the standalone ST directory. `conftest.py` also collects `agentic_ut.py`.
  Follow [testing](testing.md), [UT rules](unit-test.md) and the [UT guide](../skills/add-unit-test/SKILL.md).
  ST launchers must not import Torch/HyperParallel during collection.
- Run affected tests and report failures or unrun checks. Broaden coverage for shared contracts or unresolved concerns;
  a full RL UT run is not required for every edit. NPU ST requires explicit invocation and resources and is outside
  the main-project PR gate; mocked/CPU checks do not prove communication, numerical parity or learning.
- For Markdown changes, run `python3 .agent/scripts/check_agents_catalog.py`, Markdown lint and link/path checks.
  The catalog checker validates Skills/Agents lists only; separately verify referenced symbols and configuration semantics.
- Keep one commit per PR under [AGENTS.md](../../AGENTS.md), back up before rewriting history, and exclude unrelated changes.
