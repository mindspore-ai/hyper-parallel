# Workflow 4: CPU Gate (rl_tests)

## Goal

Prove the change passes the RL CPU suite (`rl_tests/`, from the repo root).

## Steps

### 4.1 Scoped

```bash
python -m pytest -q hyper_parallel/rl/rl_tests/<changed_test>.py
```

### 4.2 Full

```bash
python -m pytest -q hyper_parallel/rl/rl_tests
```

### 4.3 Diagnose & fix

- Read the failure; if fixable in code, fix and re-run the same file.
- If the same root cause repeats across attempts, stop and surface the output to
  the user — do not silently move past.

## Gotchas

- Run from the `hyper-parallel` repo root (pytest prepend mode resolves `rl.*`
  only when pytest treats `hyper_parallel/rl/` as the basedir).

## Pass criteria

Full suite green. If you had to skip any test (env missing, hardware), list it
in the report explicitly.
