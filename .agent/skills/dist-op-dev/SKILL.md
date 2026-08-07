---
name: dist-op-dev
description: >
  Implement distributed ops from a confirmed analysis plan; write UT/ST and
  run until executable tests pass. Goal mode — no step-by-step confirmation.
---

# Distributed Operator Development

## Trigger

Confirmed plan (resolve any `[待确认]` with the user first):

```text
/dist-op-dev implement according to .agent/skills/dist-op-analysis/plans/<Op>_dist_op_plan.md
```

Reports (Chinese): `.agent/skills/dist-op-dev/reports/{OpName}_report.md` (gitignored).

## Execution flow (autonomous)

1. **Read plan** — op name, platform scope, Path A/B/C, files, UT/ST case lists.
2. **Implement** — templates, Path A/B/C, YAML — see
   [references/implement-and-test.md](references/implement-and-test.md).
   Must comply with `.agent/rules/distributed-op-dev.md`.
3. **UT** — `tests/ut/core/shard/ops/test_parallel_{op}.py` per
   `.agent/rules/distributed-op-testing.md`.
4. **ST** (if plan requires) — MindSpore and/or Torch `cases/case_{op}.py`
   (details + commands in implement-and-test).
5. **Run & fix** until all **executable** tests pass; log to report.
6. **Complete** — list files, per-suite status, remind Ascend ST if skipped;
   prompt `/commit`. Then `/gate-doctor` for CI if needed.

## Out of scope

- git commit/push → **autogit**
- CI babysitting → **gate-doctor**
