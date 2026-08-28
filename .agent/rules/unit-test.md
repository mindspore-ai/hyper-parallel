---
name: unit-test
description: Hard constraints for HyperParallel unit tests under tests/ut — runner vs authoring, hardware-agnostic, mock distributed. Procedural how-to is the add-unit-test skill.
paths:
  - tests/ut/**
---

# Unit Test Hard Constraints

Canonical how-to: `.agent/skills/add-unit-test/` (see `SKILL.md` + `references/guide.md`).

- **Runner:** pytest (`pytest tests/ut`, markers via `@arg_mark` only where ST requires — UT typically no `@arg_mark`).
- **Authoring:** prefer `unittest.TestCase` matching existing UT; do not invent a parallel pytest-native style for the same tree without cause.
- UT must be **hardware-agnostic** and **mock distributed** (no real multi-card for logic UT).
- Naming: `test_<src>.py` mirroring source layout under `tests/ut/`.
- Assertion message style: `.agent/rules/test-assertion-style.md`.
- Shard-ops UT extras: `.agent/rules/distributed-op-testing.md`.
