# Hyper-RL Module Map & Contracts

Companion to `.agent/rules/hyper-rl-workflow.md` —
interface contracts that keep the Graph-of-Responsibilities and the shared
deployment model intact. Use it during Scope (step 1) and Implement (step 2).

## Subsystem → File Map

| Subsystem | Files | Responsibilities |
| --- | --- | --- |
| Config | `rl/config.py` | YAML/CLI, model identity, parallelism, deployment & consistency validation. Adapted into HyperParallel config objects. |
| Trainer | `rl/trainer.py`, `rl/evaluation.py` | Sole top-level orchestrator: sync loop, eval, checkpoint, resource lifecycle. |
| Dataset / Agentic | `rl/dataset/`, `rl/agentic/` | Parquet, tokenization, `PromptRecord`, `GenerationResult`; agentic: `core/` (runner, session, types, chat-template), `envs/` (base, environment, reward composition), `tools/` (executor, protocol, registry), plus the `@register_reward("gsm8k")` rule-based reward in `rl/algorithm/reward.py`. |
| Algorithm / Policy | `rl/algorithm/`, `rl/roles/policy/`, `rl/roles/model.py` | Selected-token logprob, GRPO loss/advantage/reward, Actor/Critic, model identity & registration. |
| Rollout | `rl/roles/rollout/` | One shared vLLM server (`vllm.py`), HTTP generation (`base.py`), Qwen3 adapter (`vllm_qwen3.py`), topology (`topology.py`), engine registry (`registry.py`), worker lifecycle (`worker.py`), vLLM plugin entry + rollout-side consistency install (`vllm_plugin.py`, via `HYPER_RL_CONSISTENCY_PROFILE`). |
| Weight sync | `rl/roles/weight_sync/` | Source/destination layout, transport (IPC/HCCL), transaction, publication. |
| Consistency | `rl/consistency/` | Qwen3-Ascend numeric recipe (`qwen3_dense.py`), optimizer-pre-update comparator (`gates.py`), vLLM-Ascend recipe pairing (`vllm_ascend.py`). |
| Utils | `rl/utils/` | Helpers shared across the above (incl. `monitoring/metrics.py`). |

`rl/registry.py` + `rl/roles/rollout/registry.py` handle Algorithm / environment
/ rollout-engine registration. `rl/roles/weight_sync/` is the most
consequence-heavy — 6 modules + `__init__.py` / ~5.4k lines — and the most
fragile:
`layout.py` (placement contract), `hccl.py`/`transfer.py` (transport),
`checkpoint.py`/`sync.py`/`vllm_worker.py` (publication lifecycle).

## Source Layout

`rl` is a source root, not a sub-package — the fact, its cause, and the
"run pytest from the repo root" consequence are stated once, in
[`.agent/rules/hyper-rl-workflow.md` § Two Facts You Must Know](../../../rules/hyper-rl-workflow.md#two-facts-you-must-know).
Not repeated here.

## Interface Contracts (do not break these)

- **Data contract.** `PromptRecord` carries stable prompt identity, messages,
  ground truth, tokenized prompt. Rollout-returned token IDs are the **only**
  authoritative input to the trainer — no decode / re-encode. `GenerationResult`
  contains response token IDs, a response-only mask, FP32 raw sampled-token
  logprobs (aligned to the response tokens), generation time, and worker policy
  identity. EOS is part of the action; post-EOS tokens, padding, and env content
  never enter the policy loss.
- **Batch builder.** `sequences`, `attention_mask`, `action_mask`,
  `loss_action_mask` (= `action_mask[:, 1:]`), `old_log_probs`, `advantages`,
  `reference_log_probs`. All trajectories in a batch must belong to the same
  committed policy version / fingerprint.
- **Shared deployment.** One coordinator, one endpoint, vLLM-managed DP×TP
  workers. No second Router, no fixed Trainer-rank→rollout-worker map, no
  rank-local server, no `rollout.vllm.topology`. Frontend count is vLLM-upstream
  decided.
- **Colocated vs disjoint** are the same config schema, Trainer, rollout
  controller, and policy-transaction interface; `deployment` only selects device
  ownership, residency, and transport (IPC vs HCCL). Colocated requires
  `rollout_dp × rollout_tp == trainer_dp_shard × trainer_tp`; disjoint only
  requires visible devices == `rollout_dp × rollout_tp` and no overlap with the
  trainer set.

## Change boundaries

For changes outside `rl/` and future generalization work, follow
[Scope Boundaries](../../../rules/hyper-rl-workflow.md#scope-boundaries).
This file defines interfaces, not a second scope policy.

## Coding & Verification Notes

- Multi-backend `core/` never imports torch/mindspore directly; RL code is
  Torch-only today, but keep the `get_platform()` boundary where it applies.
- Apache 2.0 header on every `.py` (lines 1–16).
- `rl_tests/` is the CPU gate (the former `hyper_parallel/rl/tests/` dir was a
  stale untracked leftover and has been removed).
