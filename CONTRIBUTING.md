# Contributing to HyperParallel

**English** | [简体中文](CONTRIBUTING.zh.md)

HyperParallel welcomes bug reports, documentation improvements, tests, model integrations, performance work, and new
parallel capabilities. You can contribute through a fork without write access to the upstream repository.
Use public issues and pull requests to make designs, review decisions, and validation results available to the community.

## Choose a contribution

- **Report a problem:** Search [existing issues](https://github.com/mindspore-ai/hyper-parallel/issues) first, then use the
  [bug-report template](https://github.com/mindspore-ai/hyper-parallel/issues/new/choose).
- **Improve documentation or tests:** Small corrections can go directly into a focused PR. Update English documents and
  their sibling `.zh.md` translations together.
- **Propose a feature or integration:** Open a Feature Issue or RFC before substantial implementation. Describe the use
  case, proposed interfaces, compatibility impact, supported hardware, and acceptance criteria. Ask the relevant
  [module owners](CODEOWNERS) to review the design.
- **Optimize performance:** Explain the workload and bottleneck, then provide reproducible correctness and performance
  comparisons with the same model, hardware, precision, parallel configuration, and measurement method.

If you are unsure where a change belongs, describe it in an issue and refer to [CODEOWNERS](CODEOWNERS).
Keep discussion respectful, respond to technical feedback, and record decisions in the issue or PR.

## Prepare your development environment

Fork [HyperParallel on GitHub](https://github.com/mindspore-ai/hyper-parallel), replace `YOUR_GITHUB_USERNAME` below, and
create a branch from the intended upstream target. Normal development targets `master`; coordinate release fixes with
the release maintainer before targeting a release branch.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/hyper-parallel.git
cd hyper-parallel
git remote add upstream https://github.com/mindspore-ai/hyper-parallel.git
git fetch upstream
git switch -c feat/your-change upstream/master
```

Create and activate an isolated Python environment using your preferred tool, such as `venv` or Conda.
Choose a Python version supported by the installation guide; the environment name is up to you.
Install a supported PyTorch and hardware runtime combination following the [installation guide](docs/installation.md).
Then run the following commands in that environment to install this checkout in editable mode:

```bash
python -m pip install -e .
python -m pip install pytest
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

The printed path must point to this checkout. If it points to another checkout or an old installation, reinstall from
the intended repository. Native components need their own build and environment activation steps in the installation
guide; an editable Python install alone does not build them.

See [development setup](docs/contributing/dev_environment.md) for dependencies and the
[GitHub collaboration manual](docs/contributing/github_development_manual.md) for authentication and account setup.
For RL work, start with the [RL entry rules](.agent/rules/hyper-rl.md), including its separate runtime setup.

## Follow the coding and documentation standards

Read [AGENTS.md](AGENTS.md) and the [coding standards](.agent/rules/code-style.md) before editing.
The coding standards are the authoritative reference, including their documented exceptions.

- Add the Apache 2.0 header to Python files. Follow PEP 8 with lines around 120 characters, public API type hints, and
  Google-style docstrings. C++ follows the repository's `.clang-format` and C++ conventions.
- Use module-level imports and native PyTorch APIs. Follow the
  [distributed implementation rules](.agent/rules/distributed.md) when changing sharding, communication, streams, or
  memory management.
- Keep changes focused. Include tests and documentation for changed behavior, public interfaces, dependencies, and
  checkpoint compatibility.
- Keep English as the default document language and update the corresponding `.zh.md` version. Check relative links,
  anchors, example commands, and API signatures.
- Fix lint and test failures at their cause. Explain necessary exceptions or untested configurations in the PR.

Install the local pre-commit hooks and run the repository checks from the activated environment:

```bash
bash scripts/pre-commit/install.sh
python .agent/skills/autogit/scripts/autogit.py check
python .agent/scripts/check_agents_catalog.py
git diff --check
```

Checks select applicable tools such as pylint, lizard, C++/shell linters, codespell, and markdownlint.
The catalog check is required whenever Markdown changes; it checks the agent catalog, so check documentation links
separately. See the [pre-commit guide](scripts/pre-commit/README.md) for installation and tool coverage.
The `autogit check` command temporarily stages changes to determine its scope; use a focused working tree.
AutoGit's remote PR workflow targets GitCode. Use Git and GitHub tooling for GitHub submissions.

## Validate the change

Run tests appropriate to the changed behavior. Follow the [testing rules](.agent/rules/testing.md) and
[testing guide](docs/contributing/testing.md) for test locations, markers, and launchers.

| Change | Expected evidence |
| --- | --- |
| Documentation only | Markdown lint, catalog check, changed links/anchors, and verification of changed commands or examples where applicable. Explain why runtime tests are not applicable. |
| Independently testable logic | Affected unit tests in `tests/ut/`, including a regression test for a bug fix. |
| Sharding, collectives, parallel combinations, or checkpoint restore | Relevant distributed ST on the affected hardware, including numerical and state-restoration checks where applicable. |
| Performance or memory optimization | Correctness checks plus baseline and changed results under the same configuration; report throughput/step time, peak memory, and measurement conditions as applicable. |
| Package or dependency change | Build, clean-environment installation, imports, and relevant examples on the declared supported combinations. |

For example, run the affected shard UT, or the full unit suite when the scope calls for it:

```bash
python -m pytest -vs tests/ut/core/shard/
python -m pytest -vs tests/ut/
```

Unit tests must not initialize a distributed job. Distributed ST uses the repository launchers described in the testing
rules; follow the selected test's device requirements rather than wrapping every launcher in another `torchrun`.
RL ST runs separately from the main-project PR gate; a main-project CI pass does not establish RL ST acceptance.

In the PR, record the tested commit, commands, model/configuration, Python and PyTorch versions, device model/count, and
results. Include `torch-npu`, CANN, and driver versions for Ascend; include CUDA and driver versions for NVIDIA GPU.
List unavailable hardware and unrun cases explicitly, and coordinate required hardware validation with maintainers.
Do not report a skipped test or a configuration that only parsed successfully as a passed hardware test.

## Submit and iterate on a pull request

1. Keep one independently reviewable change per PR. Use a Conventional Commit such as `docs: clarify installation`,
   with an imperative subject around 80 characters.
2. Keep exactly one commit per PR. Squash work-in-progress changes before submission and amend subsequent fixes into
   that commit. Back up before rewriting history. Update only your PR's source branch with `--force-with-lease` and the
   expected remote SHA; never force-push a shared main or release branch.
3. Push to your fork and open a PR against the intended upstream branch. Use the repository's
   [PR template](.github/PULL_REQUEST_TEMPLATE.md) or [Chinese template](.github/PULL_REQUEST_TEMPLATE.zh-CN.md).
   Explain the problem, resulting behavior, scope, compatibility impact, and actual test results. Link real issues;
   when none applies, state that explicitly. Complete checklist items truthfully.
4. Use a Draft PR while design, implementation, or required evidence is incomplete. Request review from the relevant
   owners in [CODEOWNERS](CODEOWNERS) when ready.
5. Address review feedback, update tests and both documentation languages, and obtain fresh CI and review results for
   the latest commit. Resolve discussions after their concerns have been addressed.

For a PR paired with a GitCode request, update the agreed source branch and let the synchronization service handle its
counterpart. Avoid competing edits on both platforms. Follow the target branch's current
[repository rules](https://github.com/mindspore-ai/hyper-parallel/rules) and any applicable identity/CLA checks.

## CI, review, and merge

Jenkins runs the project pipeline and reports test/build results to the PR. GitHub workflows enforce CI-label validity
and the review quorum. The current `master` gates include:

| Gate | What contributors need to know |
| --- | --- |
| `label-check` | A CI pass label must be valid. `ci-pipeline-passed` and `ci-pipeline-passed-but-overtime` expire 24 hours after being added. New commits require fresh CI results. The overtime label retains the indication of an abnormal duration. |
| `all-reviewers-approved` | One reviewer requires one approval; two or more reviewers require two distinct approvals. Approvals must apply to the current commit, and the author does not count. Outstanding change requests or pending team-review requests still block the gate. |
| Native branch rules | The latest push requires independent approval, stale reviews are dismissed after pushes, and review discussions must be resolved. The target branch's live rules determine additional requirements. |

Use [review and merge rules](docs/contributing/github_auto_merge.md) for the detailed policy. Passing one gate does not
replace the others. Maintainers decide readiness under the effective branch rules.

If validation fails, inspect the failing stage and logs. The author fixes code regressions with the module maintainer;
CI maintainers handle runner, environment, and reporting failures. After fixing the cause, request a retest using the
repository's supported entry point and verify results against the current commit. Do not manually add pass labels or
disable checks to make a PR mergeable.

## Release cadence and compatibility work

The community plans approximately **one Beta per month and one Stable per quarter**. Betas support early adoption and
downstream feedback. Stable releases provide accepted deployment baselines. Actual dates and scope depend on candidate
validation and maintainer review. Follow the [release process](docs/contributing/release.md) for SemVer, packaging,
release notes, documentation, known limitations, and upgrade guidance.

The **Q4 2026 CI/CD plan** expands validation from PyTorch 2.9 through the latest stable version on Ascend NPUs and
NVIDIA GPUs, including tracking the two most recent PyTorch releases. This is a roadmap target; use published results
to determine supported combinations. Record unavailable runtime combinations as pending validation with a tracking
issue. A feature marked `DEMO` retains its stated limitations even in a Stable release.

### Downstream integration, including VeOmni

Downstream projects consume HyperParallel as an optional package dependency and release independently. Integration PRs
should provide an installation path, user configuration, runnable example, compatibility boundaries, and reproducible
correctness/performance results. Update the downstream usage guide alongside the adapter.

The planned VeOmni pipeline refreshes a pinned upstream baseline weekly and validates HyperParallel changes before
merge. Adopt a new baseline only after it passes; otherwise retain the previous passing combination and track the
failure. Required integration cases must report results for both project commits and the exact environment.
The initial VeOmni integration targets Ascend; GPU support follows later. These planned checks become merge
requirements when enabled for the integration and must not be described as already running without evidence.

## Maintenance and issue follow-up

Use a [public issue](https://github.com/mindspore-ai/hyper-parallel/issues/new/choose) for a reproducible defect or a
cross-PR problem; keep PR-specific findings on the PR. Include expected/actual behavior, versions, device details,
configuration, minimal reproduction, and sanitized logs. For numerical or performance reports, include the comparison
method and baseline. Remove credentials and private training data from attachments.

Module owners review design and implementation changes. PR authors address regressions they introduce, while CI
maintainers own infrastructure failures. Record the root cause, affected/fixed versions, and regression results in
the original issue.

For the proposed VeOmni integration, HyperParallel owns the package, adapter, VeOmni usage guide, Ascend test resources,
and weekly compatibility validation. VeOmni maintainers retain review, merge, and release decisions in their repository.

## License

Contributions are made under the repository's [Apache 2.0 license](LICENSE). Include the required attribution and
license information for any third-party code and update the
[third-party notices](Third_Party_Open_Source_Software_Notice) when applicable.
