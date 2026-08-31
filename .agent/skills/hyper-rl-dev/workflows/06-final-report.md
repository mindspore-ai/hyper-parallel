# Workflow 6: Final Report & Commit

## Goal

Write the report, then hand off to `autogit` for commit/PR and `gate-doctor`
for CI.

## Steps

### 6.1 Report

Write `.agent/skills/hyper-rl-dev/reports/{Change}_report.md` (Chinese,
gitignored):

- Files touched (with paths)
- Per-suite CPU status (each `rl_tests` file, and full suite)
- NPU smoke status/verdict (or "skipped — no free NPU")
- Bit-exact values if a consistency run was made (`mismatch/max/mean`, valid
  token count)
- Design checklist — every item from the approved design's test method
- Any out-of-scope/failed item called out explicitly

### 6.2 Then prompt the user / hand off

- `/commit` (autogit) — the commit message must be shown to the user first.
- `/create-pr` if a PR is wanted.
- `/gate-doctor` only if a GitCode gate is involved.

## Anti-patterns

- Do not commit without the user seeing the commit message.
- Do not claim an NPU smoke passed when it was skipped.
- Do not silently move past a failed `rl_tests` / smoke: surface it.
