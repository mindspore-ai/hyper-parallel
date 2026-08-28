---
name: add-unit-test
description: >
  Add or extend HyperParallel unit tests under tests/ut. Use when writing UT,
  increasing coverage, or mocking distributed/hardware. Not for multi-card ST
  (see testing.md / distributed-op-testing) or shard-op-specific UT constraints
  (distributed-op-testing rule).
---

# Add Unit Test

Procedural guide for `tests/ut/**`. Hard constraints live in rules; this skill
is the how-to.

## Load

1. Hard rules: `.agent/rules/testing.md`, and for shard ops
   `.agent/rules/distributed-op-testing.md` + `.agent/rules/test-assertion-style.md`
2. Step-by-step + patterns: [references/guide.md](references/guide.md)

## Hard constraints (short)

- **Runner:** pytest. **Authoring:** prefer `unittest.TestCase` where existing UT does.
- UT is hardware-agnostic; mock distributed communication; no real NPU/GPU requirement for logic UT.
- File naming: `test_<src_file_name>.py` under `tests/ut/...`
- Arrange–Act–Assert; clear caches in setUp/tearDown when touching mesh/layout globals.

## Out of scope

- Distributed ST launchers → `testing.md` / `parallel_case`
- Op YAML ST suite cases → `distributed-op-testing`
- Commit/PR → **autogit**
