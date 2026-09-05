---
name: readability-first
description: Readability + agent-traceability gate for ALL code, docs, or .agent config written, fixed, or reviewed in this repo. Invoke BEFORE writing any change and when reviewing any diff. Treats over-complex or oversized changes as a correctness bug, not a style issue. Companion to .agent/rules/code-style.md (mechanical style) and .agent/rules/readability.md (the hard-rule shortlist this skill expands).
---

# Readability is the first principle

**Human readability comes first; coding-agent traceability is the minimum
gate.** A human should understand the change in one pass. An agent must at least
be able to trace a feature from CLI flag to executed branch, tensor/record,
metric, and test without reconstructing hidden control flow. Every rule below is
an instance of that ordering. These are **hard correctness rules**, not style
preferences — a violation is a bug and must be fixed before the change ships.

The ordering is why this skill exists at all: if a change is hard for an agent to
trace, it means the code is hiding control flow or the docs/agent config are
duplicating a fact — and both are bugs.

## Rules

1. **Code a human can't follow at a glance is a bug.** If a reviewer can't read a
   function top-to-bottom in one pass, restructure or delete it. Nesting,
   indirection, and clever constructs count against correctness — cleverness that
   costs comprehension is a defect, whatever it saves.

2. **Too much / redundant code is a bug.** Solve the problem in the fewest lines
   that stay readable. Prefer deleting code over adding it. A fix that adds more
   than ~20 lines for a problem statable in one sentence is suspect — find the
   smaller fix first.

3. **Simplicity is the core engineering metric.** When two designs both work,
   ship the one with less code, fewer concepts, fewer files.

4. **No over-encapsulation.** No new class / dataclass / module / agent / skill
   / rule file for a single call site or a single reference. A helper needs 3+
   real call sites AND nontrivial logic — otherwise inline it. Never wrap trivial
   code. Never change a function signature or return shape to thread data that
   only one caller needs. This applies to `.agent/` too: an agent that only
   points at a skill, or a rule that only restates another file, does not earn
   its file.

5. **One fact lives in exactly one place.** A source layout, a module table, a
   hard-rule list, a feature→test mapping — each has exactly one authoritative
   file. Everywhere else **links** (with a one-line "what differs here" note).
   Duplicated facts are a bug because they drift. The authoritative homes are:
   architecture → `docs/rl-architecture.md`; feature map → `docs/rl-navigation.md`;
   RL interface contracts → `.agent/skills/hyper-rl-dev/references/module-map.md`;
   hard rules → `.agent/rules/distributed.md` (long form) +
   `project-overview.md` (shortlist). Restate none of these in a second file.

6. **Simplicity is not deletion of capability.** Features, performance knobs,
   and observability are intentional — do not remove them in the name of
   simplicity. Knobs default ON stay ON. Simplify the implementation, keep the
   behavior surface.

7. **A "bug" that cannot trigger under the shipped recipes is not worth
   fixing.** Scope to the verified path (Qwen3 + GRPO, single-node sync,
   Torch/Ascend); don't gold-plate edge cases nothing runs.

## Checklist before finishing any change

- Would a human reading this cold understand it in one pass? That is the gate.
- Could this diff be half the size? If unsure, make it smaller.
- Any new class / file / agent / skill / rule? Justify each with 3+ call sites and
  nontrivial logic, or delete it.
- Did I duplicate a fact that already lives in `docs/rl-architecture.md`,
  `docs/rl-navigation.md`, `module-map.md`, `distributed.md`, or `project-overview.md`?
  If so, **link instead of restating**.
- If I changed a config key, module entry, or added a metric/test — does
  `docs/rl-navigation.md` have a row for it, and is that row updated in this diff?
  If not, review fails.
- Any signature / return-shape change? Verify every caller genuinely needs it.
- Comments: concise "why" only, 2-4 lines max, written for an external reader —
  no job ids, commit hashes, single-run metrics, or internal paths; keep upstream
  issue/PR links.
- One problem = one minimal diff. Do not batch unrelated "improvements".
- A "bug" that cannot trigger under the real recipes is not worth fixing.

## How this gates a diff

Before you write, fix, or review code: (1) load this skill; (2) load
`.agent/rules/code-style.md` for mechanical style; (3) if you touch `docs/` or
`.agent/`, load `.agent/rules/readability.md`; (4) if you touch RL, load
`.agent/rules/hyper-rl-workflow.md` first — design-first is a level above this.
Any change that fails a checklist line is a blocking finding.
