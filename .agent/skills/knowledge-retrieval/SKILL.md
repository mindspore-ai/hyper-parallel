---
name: knowledge-retrieval
description: >
  Find the answer inside this repository before answering from memory: search
  the feature docs, the code, the rules and skills, and the official docs for
  a question or a bug, rank hits by relevance, and read enough surrounding
  context to answer correctly. Handles Chinese/English wording and the
  project's abbreviations. Use for 查文档 / 这个特性在哪 / 这个报错是什么 /
  知识检索 / where is X implemented / what does <abbr> mean.
---

# Knowledge Retrieval

Answer a question about this project from **what the repository actually
says**, not from recollection. The failure mode this skill exists to prevent
is a confident answer assembled from a half-remembered API or a single
grep hit read out of context.

**This file is the index + the rules.** Source map and abbreviation list
load on demand.

## When to use

- A question about how something works, where it lives, or what a term means.
- A bug or error message whose meaning is probably already written down.
- Before answering any "does the framework support X" question.

## Search order (cheapest and most authoritative first)

1. **Rules and skills** (`.agent/rules/**`, `.agent/skills/**`) — the
   project's own constraints and procedures; these override general habit.
2. **Feature / design docs** (`docs/**`) — behaviour, interfaces, limits.
3. **Code** (`hyper_parallel/**`) — the ground truth when docs and code
   disagree; say so when they do.
4. **Tests** (`tests/**`) — executable statements of intent; often the
   clearest answer to "what is the contract".
5. **Official docs / upstream** — only after the repo has been searched, and
   only for things the repo genuinely does not define.

## Query construction (Chinese / English / abbreviations)

The same concept appears in several forms; search several, not one:

- **Both languages**: a Chinese question often maps to English identifiers in
  code (重算 → `recompute` / `checkpoint`; 显存 → `memory`; 并行 →
  `parallel`). Search the Chinese term in docs and the English term in code.
- **Abbreviations**: expand and contract (see
  [references/abbreviations.md](references/abbreviations.md)) — searching
  only `SAC` misses `selective activation checkpoint`, and vice versa.
- **Identifier spellings**: snake_case, CamelCase and hyphenated forms are
  different strings to a grep; try the ones the layer in question uses.
- **Error text**: search a distinctive *fragment* of the message, not the
  whole line (paths, ranks and numbers differ between runs).

## Relevance and context rules

- **Rank by authority, then proximity**: a rule or a docstring beats a
  passing mention in a comment; a definition beats a call site.
- **Read enough context to be right.** A single matching line is a pointer,
  not an answer: open the function, the surrounding section, or the test that
  exercises it. Most wrong answers come from quoting one line.
- **Follow the pointer chain.** `AGENTS.md` points at rules and skills; a
  SKILL.md points at its references; docs point at modules. Follow to the
  leaf rather than answering from the index.
- **Prefer the current tree.** Answer from the checked-out code, not from
  what a past version did; if history matters, say which version.

## Answer rules

- **Cite where it came from** — `file:line`, or the doc section — so the
  asker can verify.
- **Say when the repo does not answer it.** "Not found in the repo" is a
  valid, useful answer; inventing a plausible API is not.
- **Flag doc/code disagreement** rather than silently picking one.
- Keep the answer to what was asked, with the citation attached.

See [references/source-map.md](references/source-map.md) for where each kind
of knowledge lives in this repository.
