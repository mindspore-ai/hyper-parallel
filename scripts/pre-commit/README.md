# Pre-commit Setup

This directory provides one-click installation scripts for the repository
pre-commit hook. The hook runs `pylint` on staged Python files and
`lizard` on staged source files and `markdownlint-cli2` on staged
Markdown files before a commit.

## Files

- `install.sh` — Linux and macOS installer
- `install.ps1` — PowerShell installer for Windows
- `install.bat` — Command Prompt installer for Windows
- `run_pylint.py` — shared pylint entry used by pre-commit
- `run_lizard.py` — shared lizard entry used by pre-commit
- `.pre-commit-config.yaml` — root pre-commit hook definition
- `.markdownlint.jsonc` — shared Markdown lint rules

## Usage

### Linux / macOS

```bash
bash scripts/pre-commit/install.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts/pre-commit/install.ps1
```

### Windows CMD

```bat
scripts\pre-commit\install.bat
```

## What It Does

1. Ensures `pre-commit` is installed.
2. Installs the repository git hook.
3. Installs hook environments and dependencies declared in the root `.pre-commit-config.yaml`.
4. Uses the root `.pre-commit-config.yaml` to run `pylint`, `lizard`, and `markdownlint-cli2`.

## Manual Verification

```bash
python -m pre_commit run --all-files
```
