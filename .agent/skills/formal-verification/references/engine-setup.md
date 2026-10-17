# Official FM-Agent integration

The integration targets [fmagent-project/FM-Agent](https://github.com/fmagent-project/FM-Agent),
the formal-methods project, rather than other projects with the same name.
Its [public CLI](https://github.com/fmagent-project/FM-Agent#quick-start) provides the engine;
HyperParallel supplies scope and Markdown context through supported extension points.

## Install and check

Use Python 3.12+ for FM-Agent and install its dependencies in a separate environment.
The adapter itself uses only the standard library and does not import the training framework.
The following example keeps the engine outside the framework repository:

```bash
git clone https://github.com/fmagent-project/FM-Agent.git ../FM-Agent
git -C ../FM-Agent checkout 341aed7a60262186a34dc0340283a099d3195951
uv sync --project ../FM-Agent
uv run --project ../FM-Agent python .agent/skills/formal-verification/scripts/verify.py doctor --engine-root ../FM-Agent
```

This is the upstream revision used for this adapter's compatibility checks. An updated
checkout can also be used; `doctor` verifies that its actual CLI supports the selected options.
`uv` is the upstream dependency manager; alternatively invoke the adapter with the engine's
installed Python via `--engine-python /absolute/path/to/engine/python`.
On Windows, this is typically `FM-Agent/.venv/Scripts/python.exe`.

The upstream's ArkTS dependency builds a native extension and may require a C++ compiler.
For Python-only HyperParallel analysis on a Windows host without that compiler, install
the Python engine environment with `uv sync --project ../FM-Agent --no-install-package tree-sitter-arkts`.
Then use its interpreter through `--engine-python` or `uv run --no-sync` so the next launch
does not retry the omitted package. This does not enable ArkTS verification. The adapter
sets UTF-8 mode for engine subprocesses because the upstream reads TOML/prompts using the
locale default and Windows GBK cannot decode some of its UTF-8 files.

Configure model authentication **in the external FM-Agent environment**, following its
[model-backend documentation](https://github.com/fmagent-project/FM-Agent/blob/main/docs/config_llm.md).
The upstream supports OpenCode/API and authenticated local CLI backends. Choose an installed
backend and a model supported by it; the adapter does not override model or credential settings.
`FM_AGENT_HOME` can replace repeated `--engine-root` arguments. Never save API keys in the
HyperParallel configuration. `doctor` checks imports/options, not model authentication.

### Windows native CodeGraph

The original pinned revision above imports POSIX-only `fcntl` while installing
CodeGraph and uses the shell installer and Unix launcher path. Native Windows
analysis therefore needs the compatibility fix in
[FM-Agent PR #242](https://github.com/fmagent-project/FM-Agent/pull/242).
Until it is merged upstream, the tested public fork revision can be selected in
the external engine checkout:

```bash
git -C ../FM-Agent fetch https://github.com/weich97/FM-Agent.git ed31bed6fb230b6604f9d275086073bc05deb90c
git -C ../FM-Agent checkout ed31bed6fb230b6604f9d275086073bc05deb90c
```

This fix builds on the original upstream revision and preserves its CLI. It
installs the pinned native release into FM-Agent's private cache, uses a Windows
installation lock, and preserves qualified method identities through extraction
and incremental cleanup. It does not change the persistent user PATH. Keep the
default `[codegraph].version`; `FM_AGENT_CODEGRAPH_DIR` can select a private cache
directory. Use the same engine interpreter and adapter commands below.

Native Windows x64 installation, HyperParallel source extraction, a small full
LLM pipeline and resume were exercised. Linux aarch64 installation and regression
tests also passed. The fork's Windows/Linux CI passed on this exact revision;
the upstream PR's first-contribution workflow still needs maintainer approval.
These checks do not establish support for every optional language backend or a
machine-checked proof. See the [validation record](validation.md).

## Run from the HyperParallel repository root

```bash
# Offline plan: no engine or model calls, no generated files.
python .agent/skills/formal-verification/scripts/verify.py plan --profile dtensor

# Official estimate: no LLM calls; creates fm_agent_estimate/estimate.json.
uv run --project ../FM-Agent python .agent/skills/formal-verification/scripts/verify.py run --engine-root ../FM-Agent --profile dtensor --mode estimate

# Specifications only: consumes model calls, does not confirm bugs.
uv run --project ../FM-Agent python .agent/skills/formal-verification/scripts/verify.py run --engine-root ../FM-Agent --profile dtensor --mode spec

# Full candidate search and per-candidate validation in an isolated snapshot.
uv run --project ../FM-Agent python .agent/skills/formal-verification/scripts/verify.py run --engine-root ../FM-Agent --profile dtensor --mode verify

# Resume an interrupted verify run with exactly the same source scope.
uv run --project ../FM-Agent python .agent/skills/formal-verification/scripts/verify.py run --engine-root ../FM-Agent --profile dtensor --resume

# Incremental verification after a completed baseline run and a code change.
uv run --project ../FM-Agent python .agent/skills/formal-verification/scripts/verify.py run --engine-root ../FM-Agent --profile dtensor --incremental intent.md
```

Relative `--repo`, `--config`, `--engine-root`, `--knowledge` and intent paths are resolved
from the adapter's launch directory. Knowledge/validator entries **inside the JSON config**
are relative to that config file. `--submodule` paths are relative to the framework root.
Absolute paths work on Windows and Linux. Commands use argument lists without a shell.

`plan` and `run --dry-run` return JSON with `analysis_status: not_run`, the HEAD revision,
local-change indicator, scopes, inputs and exact argv. They are not verification results.
`doctor` executes only the upstream `main.py --help`; it makes no model calls.

The official isolate mode snapshots committed and non-ignored local source changes, runs
in a separate worktree, and copies `fm_agent/` results back. Preserve the snapshot path
printed by FM-Agent; HEAD alone does not identify uncommitted source. The adapter refuses
to overwrite an existing workspace on a fresh run. Archive `fm_agent/` before starting a
new scope/mode; `--resume` and `--incremental` reuse existing engine artifacts.
The adapter records framework, mode and source scope in `fm_agent/training_adapter.json`
and rejects reuse with a different scope. Older runs without this record can be resumed
using the original upstream CLI or archived before a new adapter run.
Do not run two analyses concurrently against the same repository's `fm_agent/` workspace.

The launcher preserves engine exit codes. Some upstream setup failures exit zero;
the adapter additionally requires the estimate artifact or completed-run `version.log`.
A completed pipeline means fresh artifacts were produced, not that all reasoning or validation
succeeded. Inspect pending/error candidates and engine traces before declaring coverage.

The framework needs to be installed editable in the **probe environment**. An isolated
probe must confirm its imported module belongs to the analyzed snapshot rather than an
unrelated checkout. Engine dependencies and framework dependencies can be managed separately.

## Results

Inspect the original artifacts; no second report schema is imposed:

- `fm_agent/report.html`: official browsable report.
- `fm_agent/logic_verification_results/`: reasoning and mismatch candidates.
- `fm_agent/bug_validation/summary.json`: confirmed, unconfirmed and pending counts.
- `fm_agent/bug_validation/*.result.json`, `*.md`, `probe_*.py`: reproduction evidence.
- `fm_agent/trace/`: model/agent execution trace.

The upstream incremental flow may need explicit report regeneration:

```bash
uv run --project ../FM-Agent python ../FM-Agent/report.py ./fm_agent
```

An accelerator-only candidate with no device reproduction remains unresolved. A `MATCH`
is LLM reasoning, not a machine-checked proof. Avoid presenting spec-only or estimated
work as a successful bug hunt.

## Other training frameworks

Create a config with the same schema, for example:

```json
{
  "framework": "AnotherTrainingFramework",
  "domain_knowledge": ["training-invariants.md", "framework-contracts.md"],
  "bug_validator": "bug-validator.md",
  "profiles": {
    "distributed": {
      "submodules": ["src/distributed"],
      "focus": "Gradient synchronization and parameter state transitions"
    }
  }
}
```

Supply Markdown files appropriate to that framework and invoke:

```bash
python /path/to/verify.py plan --repo /path/to/framework --config /path/to/config.json --profile distributed
```

Use the same options with `doctor` and `run`. The generic training contracts are reusable;
replace HyperParallel-specific paths, rules, test launchers and public API assumptions.

## Adapter validation

Run its independent standard-library tests with:

```bash
python -m pytest --confcutdir=tests/ut/agent tests/ut/agent/formal_verification -q
python -m pylint .agent/skills/formal-verification/scripts/verify.py tests/ut/agent/formal_verification/test_verify.py
python scripts/pre-commit/run_lizard.py .agent/skills/formal-verification/scripts/verify.py tests/ut/agent/formal_verification/test_verify.py
python .agent/scripts/check_agents_catalog.py
```

`--confcutdir` keeps this standalone tooling suite independent of the main UT hook's eager
Torch/HyperParallel import. The tests remain under `tests/ut/` and also run within the normal
UT gate when the framework environment is installed. They use temporary Git repositories
and an executable CLI fixture; no model credentials, distributed initialization or device
are needed. The real `doctor`/`estimate` commands above check the official engine separately.
When CI copies only the test tree, the module explicitly skips if the optional `.agent`
source tooling is absent, following the repository's agent-tool testing convention.
The suite includes a separate copied-tree collection regression check.
The initial [validation record](validation.md) lists observed results and their test boundary.
