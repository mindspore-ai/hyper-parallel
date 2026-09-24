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
- The built-in algorithms are GRPO and PPO. Qwen3 dense retains FSDP sharding with TP1/TP2 and
  `dp_replicate=cp=pp=ep=edp_shard=1`; shared vLLM rollout supports colocated or disjoint mode.
- Qwen3-30B-A3B (`qwen3_moe`) uses the shared AutoModel builder and the existing Qwen3-MoE recipe.
  Its RL scope is GRPO with colocated native vLLM, consistency off and EPLB off; MoE PPO and disjoint rollout are rejected.
  Training supports TP1/TP2 with EP/EDP, subject to world-size and expert-count divisibility;
  `dp_replicate=cp=pp=1` remains required. See the
  [MoE recipe](../../hyper_parallel/rl/examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_vllm.yaml) and
  [MoE validation scope](../../hyper_parallel/rl/docs/moe_code_agent.md#功能与支持边界).
  Check `rl/config.py` and `rl/roles/model_setup.py` for the executable boundary; do not infer RL support from a main-project API.
- Weight synchronization selects `full_gather` or `direct_reshard`, with IPC for colocated and HCCL for disjoint.
  Publication errors propagate; there is no automatic fallback. Preserve this contract when fixing failures.
- Single-turn Python stdio tasks use `examples.code.agent` / `code_stdio` with the internal runner.
  Keep private test cases in structured `PromptRecord.ground_truth`; `data.row_adapter` preserves messages and task identity.
  Only the TP request owner executes the environment. Judge every declared test before assigning binary success;
  execution-service failures propagate and must not become zero-reward candidates. See the
  [code example](../../hyper_parallel/rl/examples/code/README.md) for the fixed SandboxFusion runtime and reviewed data.
- External Codex and DeepSeek programs train each call under its actual prompt with episode-level GRPO.
  Complete call identities, shared rewards and policy versions are required. DP padding has zero loss and is excluded
  from episode reporting; segmented PPO is rejected. Legacy continuous builders require exact sampled-action prefixes.
  External tool attribution uses the pinned Hermes parser evidence: infrastructure or unknown failures reject the update,
  while verified malformed model actions remain in the episode. See the
  [agent contract](../../hyper_parallel/rl/docs/agentic_rl.md) for budgets, shutdown and support boundaries.
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

- UT lives in `tests/ut/rl/`; NPU ST and its launch helpers live in `hyper_parallel/rl/tests/st/`.
  Shared recipe data and configuration construction live in `tests/common/rl_st_cases.py` so UT can run without the ST
  archive. Do not introduce a UT dependency on the standalone ST directory. The UT conftest also collects `agentic_ut.py`.
- RL feature UT belongs in `tests/ut/rl/`, grouped by data, agentic, trainer and weight-sync ownership.
  MoE configuration, structured code tasks, episode padding and tool attribution are covered by CPU contracts.
- Real feature ST uses `hyper_parallel/rl/tests/st/test_feature_st.py`: two-rank Gloo padding, optional real
  SandboxFusion, and explicit MoE/code/agent training workers. Missing runtime resources are not a passing result.
  The existing `test_rl_st.py` dense/consistency/PPO recipes remain independent. See the
  [feature validation guide](../../hyper_parallel/rl/docs/moe_code_agent.md) for configuration and evidence boundaries.
- Follow the repository [testing rules](testing.md) and [UT rules](unit-test.md). ST launchers must not import Torch or HyperParallel during collection. Use the [UT guide](../../hyper_parallel/rl/docs/hyper_rl_ut.md) and
  [ST guide](../../hyper_parallel/rl/README.md#系统测试) for execution commands and resources.
- For docs and agent rules, run `python3 .agent/scripts/check_agents_catalog.py`, Markdown lint, and checks for changed
  relative links and referenced paths. The catalog script only compares Skills/Agents tables against disk; it does not
  validate navigation symbols, config keys, metrics, or test semantics. Check those against code and tests separately.
- For code, run affected tests from the repository root. Broaden regression coverage for shared contracts, core flows,
  or uncertain impact; a full `tests/ut/rl/` run is not mandatory for every local edit.
- Run the applicable real-NPU acceptance when the changed product contract requires it. RL ST is retained under the RL subproject and temporarily excluded from the main-project PR gate.
  It requires explicit invocation and resources; default PR CI success or mocked UT does not prove every RL recipe passed. Report missing
  hardware or resources as not run, and preserve learning, policy-version, and consistency assertions.
