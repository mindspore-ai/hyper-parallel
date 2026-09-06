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
- Prefer deleting over adding; a helper needs 3+ real call sites AND nontrivial
  logic, else inline it. Applies to `.agent/` agents/skills/rules too.
- **One fact = one place.** Architecture → `docs/rl-architecture.md`; feature map →
  `docs/rl-navigation.md`; RL contracts → `module-map.md`; hard rules →
  `distributed.md`. Every other file links, never restates.
- A config-key / module / metric / test change must update `docs/rl-navigation.md`
  in the same diff, or review fails.
- Do not remove features, performance knobs, or observability for simplicity.
- Scope to the verified recipe; a bug that can't trigger there isn't worth fixing.
