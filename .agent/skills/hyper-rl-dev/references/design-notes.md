# Design Notes (Hyper-RL)

Go-to decisions, invariants and rejection rules, used when turning an approved
design into implementation and when writing the design.

## Invariants (from architecture.md)

- **Unified training architecture.** Trainer loads the Transformers model def
  directly and uses HyperAutoModel's parallelism — currently FSDP/TP, potentially
  any of its parallel modes (FSDP/HSDP/PP/DP) — along with its optimizer,
  checkpoint and grad clipping; no RL-specific Trainer copy, no `trainer_dev` path.
- **Train-inference unity.** Trainer and Hyper-vLLM both take the Transformers
  Qwen3 def as their model-semantics source and reuse HyperParallel's
  ShardingPlanner, TP placement, tied-weight and source-layout contract. The
  vLLM paged-attention / KV cache / worker lifecycle is the only
  inference-specific leaf.
- **One shared rollout.** Colocated and disjoint share one coordinator, one
  endpoint, vLLM-managed DP×TP workers.
- **Strong policy consistency.** Policy version is monotonic. Generation, weight
  transaction, cache reset and resume verify worker-local identity; on failure
  all trainer ranks exit together; an incomplete policy is never visible.
- **Consistency isolation.** `consistency.enabled=false` (default) allows
  Trainer/rollout TP to differ. Only `=true` installs the Qwen3-Ascend recipe
  and requires Hyper-vLLM matched TP.
- **The Reference Actor is a separate frozen model**, not the current Actor in
  temporary eval mode.

## Rejected by design (re-introducing any of these is a regression)

- `rollout.vllm.topology`, `request_concurrency`, `api_server_count`
- Rank-local server, per-rank port, second Router, fixed Trainer-rank→worker map
- Multi-node, async/off-policy rollout, dynamic scaling, transparent generation retry
- A HyperParallel source change made without declaring its public/internal
  boundary and its impact on surrounding consumers — see
  `.agent/rules/hyper-rl-workflow.md` § Scope Boundaries, which permits such
  changes when scoped.

## Model identity

- `model.registry_name` (e.g. `qwen3_4b`) → `ModelRegistration`. Resolved to a
  trainer model and, for rollout, to a vLLM model via
  `normalize_model_implementation` / `resolve_vllm_model`.
- Tied embeddings must keep the same Parameter identity across trainer and
  rollout sharding — see `docs/public_module_changes.md` for the tied-alias
  fixes that make this work.

## Algorithm & loss

- GRPO: advantage from group-normalized rewards (mean/std over the response
  group), advantage epsilon, clip ratio, `loss_aggregation: token-mean`.
- `kl_type: low_var_kl`, `kl_coef` for the reference-KL term.
- `learning_gate.*` controls a zero-update skip — useful for smoke runs, but a
  gated zero-grad step is still a valid (vacuous) step.

## Config boundaries to check when editing `config.py`

- `rollout.vllm.deployment: colocated|disjoint`; `model_implementation:
  hyper|native`; `weight_sync.strategy: full_gather|direct_reshard`;
  `weight_sync.fallback_strategy: none|full_gather`.
- `consistency.enabled=true` only with Qwen3 Hyper-vLLM matched TP.
- Disjoint must provide a `visible_devices` set equal to `rollout_dp*rollout_tp`
  and non-overlapping with the trainer set.
- Config translation is where `_validate_colocated_vllm` /
  `_validate_disjoint_vllm` / `_validate_model_implementation` live — attach
  new validation there, and add a CPU test in `rl_tests/test_config.py`.

## Metric / eval surface

- `logging.backends` (console / wandb), `evaluation.*`, `progress_steps`.
- FP32 raw selected-token logprobs and `0/0/0` bit-exact are **pre-update**
  comparison artifacts — not gradient, optimizer-state, or convergence claims.
