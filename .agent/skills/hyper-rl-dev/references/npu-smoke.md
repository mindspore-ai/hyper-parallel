# NPU Smoke Runs (Hyper-RL)

Two launchers, two scopes. Never swap them. Run only on NPUs with
`npu-smi info` showing `Health=OK` and no other running process. All commands
from the `hyper-parallel` repo root.

## Which launcher?

| Your change... | Launcher | Consistency |
| --- | --- | --- |
| Trainer / rollout / dataset / algorithm — ordinary TP | `run_qwen3_tp_docker.sh {colocated\|disjoint}` | **disabled** |
| Weight-sync strategy, layout, transaction, bit-exact | `run_qwen3_consistency_docker.sh {colocated\|disjoint}` | **enabled** |
| A pure-CPU path (config validation, batch builder, math) | CPU-only `rl_tests`; NPU smoke optional | n/a |

> `run_qwen3_tp_docker.sh` explicitly disables consistency — it **cannot** be
> used to claim bit-exact.

## Launcher commands

**Ordinary TP (4 NPU; Trainer `FSDP-shard2×TP2` → rollout `DP2×TP2`):**

```bash
export HYPER_QWEN3_TP_MODEL_ROOT=<Qwen3-4B dir>
export HYPER_QWEN3_TP_DATA_ROOT=<gsm8k dir>
export HYPER_QWEN3_TP_RESULT_ROOT=<results>/normal-tp2
export HYPER_QWEN3_TP_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP_TRAINER_TP=2
export HYPER_QWEN3_TP_ROLLOUT_TP=2
export HYPER_QWEN3_TP_IMPLEMENTATION=hyper   # hyper | native
./hyper_parallel/rl/examples/scripts/run_qwen3_tp_docker.sh colocated
```

`colocated` vs `disjoint` differs only in device ownership / residency /
transport. Disjoint adds:

```bash
export HYPER_QWEN3_TP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_QWEN3_TP_TRAINER_COUNT=4
export HYPER_QWEN3_TP_ROLLOUT_DP=2
./hyper_parallel/rl/examples/scripts/run_qwen3_tp_docker.sh disjoint
```

**Bit-exact (TP1 = 2 NPU, TP2 = 4 NPU):**

```bash
export HYPER_QWEN3_MODEL_ROOT=<Qwen3-4B dir>
export HYPER_QWEN3_DATA_ROOT=<gsm8k dir>
export HYPER_QWEN3_RESULT_ROOT=<results>/consistency-tp2
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3
export HYPER_QWEN3_TP=2
./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh colocated
```

Disjoint TP2 (8 NPU):

```bash
export HYPER_QWEN3_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYPER_QWEN3_TP=2
export HYPER_QWEN3_TRAINER_COUNT=4
export HYPER_QWEN3_ROLLOUT_DP=2
./hyper_parallel/rl/examples/scripts/run_qwen3_consistency_docker.sh disjoint
```

Bit-exact extra env (`direct_reshard` with `full_gather` fallback):

```bash
export HYPER_QWEN3_WEIGHT_SYNC_STRATEGY=direct_reshard
export HYPER_QWEN3_WEIGHT_SYNC_FALLBACK=full_gather
```

## Report this per run

Hardware (e.g. Ascend 910B3), world size, dtype (bfloat16), parallel dimensions
(Trainer `dp_shard×tp`, rollout `dp×tp`), model/checkpoint path, `deployment`,
`model_implementation`, `weight_sync strategy/fallback`, and the exact command.
Then the verdict.

## Pass / fail

- **Ordinary TP:** trainer makes a non-zero update step with non-zero valid
  tokens. A fast smoke whose random batch is all-equal-reward yields zero GRPO
  advantage and zero gradient — that is **correct, not a failure**; run the
  default config to observe a non-zero update if you need the gradient to be non-zero.
- **Bit-exact:** per comparison step `valid-token count > 0` and
  `mismatch_count == max_abs_diff == mean_abs_diff == 0` (rendered `0/0/0`).
  Non-zero exit → no version publish; report the failure, do not re-run silently.
- **If no free/healthy NPU or env is unavailable:** mark the smoke `skipped` in
  the report and say so plainly — never claim a smoke passed when you didn't run
  it.

## Verified matrix (for reference)

`colocated TP1 FSDP → DP×TP1`, `colocated FSDP-shard2×TP2 → DP2×TP2`,
`disjoint pure-TP2 → DP1×TP2`, `disjoint FSDP-shard2×TP2 → DP2×TP2`,
direct-failure→full fallback, direct+fallback double-failure, prefix
cache/chunked prefill, and DCP destroy/resume/refit — see
`hyper_parallel/rl/docs/qwen3_training_inference_consistency.md`.
