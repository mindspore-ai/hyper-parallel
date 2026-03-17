---
description: Multi-platform and list/collection API consistency — apply when adding or changing APIs that exist on both Torch and MindSpore or accept list/collection inputs.
---

## Multi-Platform & List/Collection APIs

When implementing or changing features that (1) have both Torch and MindSpore paths, or (2) accept list/collection inputs (e.g. `fully_shard([m1, m2])`), ensure:

- **Same semantics on both backends** — Compare the two platform implementations: same arguments to state/scheduler (e.g. single module vs tuple of modules); same “who gets the handle” behavior.
- **List/collection contract** — Decide and document: does every element get a handle? Can any element be used in follow-up APIs (e.g. prefetch)? Implement and test accordingly.
- **State/handle coverage** — If one logical unit spans multiple user-visible objects (e.g. multiple roots), either attach the same handle to all or document that only the first is the control handle.
- **Tests from user perspective** — Include at least one test that uses the “non-first” element (e.g. second root `.unshard()` or as prefetch target); avoid mocking away the path under test.
- **In-place return assertions** — Use a named variable for the container: `in_list = [a, b]; result = api(in_list); assert result is in_list`. Do not use `assert result is [a, b]`.

See also: `.claude/skills/code-review/review-checklist.md` § Multi-Platform & List/Collection APIs.
