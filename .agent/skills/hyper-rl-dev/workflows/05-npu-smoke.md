# Workflow 5: NPU Smoke

## Goal

Verify on real Ascend NPU, per the design's test method, using the launcher that
matches the change scope.

## Steps

### 5.1 Choose the launcher

From [../references/npu-smoke.md](../references/npu-smoke.md): `run_qwen3_tp_docker.sh`
(ordinary TP, consistency disabled) or `run_qwen3_consistency_docker.sh`
(bit-exact). Never swap.

### 5.2 Preflight

```bash
npu-smi info   # only Health=OK, no other process
```

### 5.3 Run & record

Copy the exact command; record hardware, world size, dtype, `dp_shard×tp` /
`dp×tp`, model/checkpoint path, `deployment`, `model_implementation`,
`weight_sync strategy/fallback`, and the outcome.

### 5.4 Verdict

- Ordinary TP: non-zero gradient/update step (or explicitly note that the
  all-equal-reward step is correct-but-vacuous).
- Bit-exact: per step, valid-token count > 0 and `0/0/0`.
- No free NPU → mark **skipped** and say so; **never** claim a smoke passed
  that you did not run.
