---
name: feature-doc
description: >
  Author or update a HyperParallel feature / model design document from a
  fixed template: feature statement, migration-and-adaptation approach, code
  change scope, usage limits, and the functionality / precision / performance
  verification reports. Use for 特性文档 / 特性设计说明书 / 模型迁移文档 /
  需求串讲 / feature doc / design doc, for either a new feature-or-model doc
  or an update to an existing one. Produces a review-ready Markdown document
  under docs/, never code.
---

# Feature Doc

Turn a feature or model bring-up into a review-ready design document that
carries enough for code review and acceptance: what the feature is, how it
was migrated / adapted, what code changed, where the limits are, and the
functionality / precision / performance evidence.

**This file is the index + hard rules.** The document shape lives in
`templates/feature-design-doc.md`; fill every section's `[FILL]` marker.

## Two paths

| Path | When | What differs |
|---|---|---|
| **new** | A new feature or model doc | Start from the full template; every section is filled from scratch. |
| **update** | An existing doc needs revision | Read the existing doc first, change only the sections the diff touches, and append a dated entry to its change log. Do not rewrite unchanged sections. |

```bash
# new: scaffold a fresh doc from the template
cp .agent/skills/feature-doc/templates/feature-design-doc.md \
   docs/design/<feature-or-model>.md
# update: edit the existing docs/design/<...>.md in place
```

## Required sections (the template enforces these)

Every doc must support both code review and acceptance, so none of these may
be left empty for a `new` doc:

1. **特性/模型说明** — background, goal, scope boundary, and the external
   interface (config fields, entry points, YAML surface).
2. **迁移适配方案** — the approach: delayed init, high-performance module
   replacement, TP/CP/EP adaptation, checkpoint conversion, what was reused
   vs rewritten, and why.
3. **代码修改范围** — the files and the part of each that changed, at the
   granularity a reviewer needs to locate the change (module / function).
4. **使用限制** — supported shapes, dtypes, parallel degrees, sequence
   lengths, hardware; and the known-unsupported cases with the reason.
5. **验证报告** — three sub-reports, each with its command, commit hash and
   environment so the result is reproducible:
   - **功能** — what ran and passed (UT / ST / smoke), with the run evidence.
   - **精度** — against the acceptance spec (see the `precision-acceptance`
     rule): 1000-step mean-absolute-error / mean-relative-error, plus
     grad_norm, first-step, determinism and resume-from-checkpoint checks,
     and the baseline it was compared to.
   - **性能** — the breakdown (compute / communication / free / optimizer)
     and the headline metric, same-node back-to-back where it is an A/B.
6. **组件依赖与兼容性** — upstream / operator / framework dependencies and
   the compatibility constraints (versions, co-features).
7. **变更记录** — dated change log (the `update` path appends here).

## Hard rules

- **Evidence is reproducible or it is not evidence.** Every number in the
  verification report carries its command, commit hash and environment; a
  bare number is rejected (shared discipline with the `autoresearch` and
  `gate-doctor` skills).
- **Positive evidence only.** "No error observed" is not a pass; state what
  ran and what it produced. A fallback run and a real run look identical
  without a positive marker.
- **Precision numbers follow the acceptance spec**, not an ad-hoc bar; link
  to `rules/precision-acceptance.md` and state the baseline explicitly.
- **Do not invent results.** An unmeasured section stays marked `[待测]`
  with the reason, never filled with a plausible-looking number.
- The document is for humans reviewing a change; keep it in the repo under
  `docs/design/`, in Markdown, in the language of the surrounding docs.

## Review

A `new` doc is worth one review pass of the filled template before the
feature merges; an `update` only needs the changed sections reviewed. The
functionality / precision / performance sub-reports are the parts a reviewer
leans on most, so they get the tightest scrutiny.
