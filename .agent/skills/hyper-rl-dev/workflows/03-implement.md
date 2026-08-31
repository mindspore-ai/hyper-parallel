# Workflow 3: Implement

## Goal

Turn the approved design into code, respecting the layouts/contracts in
[../references/module-map.md](../references/module-map.md).

## Steps

### 3.1 Setup

- Read `.agent/rules/code-style.md` before editing. Apache 2.0 header on new
  `.py` files (lines 1–16).
- Confirm the exact files from Workflow 1; do not silently widen the diff.

### 3.2 Implement

- Follow the subsystem invariants in
  [../references/design-notes.md](../references/design-notes.md) and the
  contract table in module-map.md — especially the data contract (no
  decode/re-encode of rollout tokens), shared deployment (no new Router), and
  the weight-sync transaction semantics
  ([../references/weight-sync.md](../references/weight-sync.md)).
- If implementing a config change, attach validation in `config.py` and add a
  CPU test in `rl_tests/test_config.py`.

### 3.3 Self-check

- No `import torch`/`mindspore` at `core/` level (RL is Torch-only, but keep the
  platform boundary where it applies).
- Design-consistent: reverify against the approved design's "files touched" list.

## Output

The implementation, ready for the CPU gate (Workflow 4) and NPU smoke
(Workflow 5).
