# Weight-Sync Strategy Matrix & Success Criteria

How a Trainer `FSDP/FSDP+TP` source layout publishes weights to the rollout
`DP×TP` destination layout. This is the most consequence-heavy subsystem — read
before touching `rl/roles/weight_sync/`.

## Layout Model

```text
Trainer FSDP/FSDP+TP source layout
  -> rollout DP/TP destination layout
    -> bounded transfer plan
      -> NPU IPC (colocated) or HCCL (disjoint)
        -> worker transaction
          -> identity / source-derived manifest verification
            -> controller publication
```

`placements`, source layout, and `checkpoint→TP` mapping must come from the same
`ShardingPlanner` / `apply_sharding_plan` contract as the trainer — never
hand-write a second mapping.

## Strategy Matrix

| Strategy | TP1 | Qwen3 TP2 |
| --- | --- | --- |
| `full_gather` | Rebuild full logical tensors | Normal strategy, oracle, or fallback |
| `direct_reshard` | Auto-degrades to full-gather | Transfer only source/destination region intersection |

## Choosing a Strategy

Default to **`full_gather`** unless the design explicitly requires
`direct_reshard` (e.g. a TP-mismatch or a bandwidth-motivated change).
`fallback_strategy ∈ {none, full_gather}` — `full_gather` is the only fallback.

## Transaction Semantics

- Policy version is strictly monotonic. Generation, weight transaction, cache
  reset and resume all verify worker-local identity; on failure all trainer
  ranks exit together, and an incomplete policy **must not be visible**.
- Direct failure path: abort the pending transaction **first**, then let
  full-gather overwrite completely. If fallback also fails: keep admission
  closed, do not restore rollout, and **do not publish a new version**.
- Successful full/direct publication resets the KV cache and re-exposes the
  version after worker verification.
- Failure semantics and admission ownership live in
  `hyper_parallel/rl/docs/vllm_rollout.md` — consult before changing anything
  that can block or skip a publication.

## Pass Criteria

- Publication succeeds only when every destination worker verifies against the
  **source-derived manifest** (not "logprobs match, so it's fine").
- A step claiming weight correctness must have a non-zero valid-token count and
  distinct-identity match; zero-advantage steps are correct-but-vacuous, not a
  pass.

## Never

- Re-introduce a rank-local server, an RL-only Router, a second topology config,
  or `rollout.vllm.topology` / `request_concurrency` / `api_server_count`.
- Publish a version without a verified transaction; silent skip is a CI/UX bug.
