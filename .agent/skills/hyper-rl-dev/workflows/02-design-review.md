# Workflow 2: Design Review

> This step owns producing a design; the **human** owns approving it.

## Goal

For every code change (features, bug fixes, refactors, docs/config edits):
produce a **design + test method** and stop for human confirmation before
implementing.

## Producing the design

Cover, in Chinese, one page max:

1. **Approach** — where the change lands (subsystem + files), interfaces touched.
2. **Edge cases** — what could make it wrong (zero-advantage steps, TP mismatch,
   transaction abort, layout contract, resume paths).
3. **Test method** — which `rl_tests` (and which NPU launcher, if applicable),
   the **exact pass criteria** (e.g. bit-exact `0/0/0` + non-zero valid tokens).
4. **Files touched** — verify against
   [Workflow 1](01-scope-analysis.md) that nothing outside `hyper_parallel/rl/`
   is needed.

## Gotchas

- If the change might regress the verified Qwen3 + GRPO path, say how you'll
  prove it doesn't (run the existing `rl_tests`, run the default consistency
  smoke, confirm `0/0/0`).
- "Run tests" with no pass threshold is **not** a test method. State the
  threshold.
- If you cannot get a free/healthy NPU, note the NPU gate as "skipped" in the
  design so the human knows before approving.

## Approval gate

Do not implement. Present the design in chat; wait for the human's OK / edit /
cancel. After approval, proceed to [Workflow 3](03-implement.md) autonomously.
