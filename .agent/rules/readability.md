---
name: readability
description: Readability hard-rule shortlist — human-readable first, agent-traceable minimum gate. Autoloaded with any change; full expansion in skill readability-first.
---

# Readability (hard rules, canonical shortlist)

The priority order is explicit: **human readability comes first; coding-agent
traceability is the minimum gate.** The full rules + checklist:
`.agent/skills/readability-first/SKILL.md`. This file is the shortlist that is
always in context; do not restate the skill's prose here.

- Over-complex / hard-to-follow **code or docs** is a bug, not a style issue.
- Prefer deleting redundant code and instructions. Extract abstractions when
  they reduce reading cost or define a necessary boundary; avoid trivial wrappers.
- **One fact = one place.** Architecture → `docs/rl-architecture.md`; feature map →
  `docs/rl-navigation.md`; RL contracts → `module-map.md`; hard rules →
  `distributed.md`. Every other file links, never restates.
- Changes to documented feature behavior or references must update the navigation
  in the same diff. Internal-only changes need verification, not a cosmetic edit.
- Do not remove features, performance knobs, or observability for simplicity.
- Scope to the verified recipe; a bug that can't trigger there isn't worth fixing.
