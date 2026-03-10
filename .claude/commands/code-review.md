# /code-review — Code Review

Delegates to the `code-review` skill for a full review focused on distributed system correctness.

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
- **Testing** — coverage, distributed test patterns
