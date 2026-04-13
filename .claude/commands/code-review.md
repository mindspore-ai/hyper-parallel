# /code-review — Code Review

Delegates to the `code-review` skill for a full review focused on distributed system correctness.
The review must load `.claude/rules/code-style.md` first and treat violations as blocking findings.

## Usage

```bash
/code-review              # review current branch vs upstream/master
/code-review #160         # review PR #160
/code-review branch       # explicit local branch mode
/code-review #160 detailed  # line-by-line detailed review
```

## What It Does

Invokes `/skill code-review` with the provided arguments. See `.claude/skills/code-review/SKILL.md` for full workflow details.

The review covers:

- **Stream synchronization** — async collectives, non_blocking, cross-stream deps
- **Memory lifecycle** — storage free, buffer cleanup, grad nulling
- **DTensor correctness** — layout, placement, redistribution invariants
- **Cross-platform consistency** — torch/mindspore parity
- **Code quality** — conventions, patterns, design
- **Style compliance** — mandatory enforcement of `.claude/rules/code-style.md`
- **Testing** — coverage, distributed test patterns
- **Pylint (review-PR stage)** — Run `autogit pylint-review` on changed Python files and include the report in the Code Quality section; add violations to `.jenkins/check/config/filter_pylint.txt` for unified suppression (do not use inline `# pylint: disable=` except `C0415` for lazy backend imports under `platform/torch/` and `platform/mindspore/` per `code-style.md`)

When the review finds a `code-style.md` violation, the output must:

- explicitly identify the violation
- provide the auto-fixed complete code
- avoid leaving a non-compliant "suggested" version as the final answer
