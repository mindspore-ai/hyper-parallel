# Release and Integration Plan

**English** | [简体中文](release_and_integration_plan.zh.md)

This page collects the Q4 2026 release and integration targets previously listed in the contributing guide.
These are planning targets, not a record of completed validation or additional active merge gates. Follow the
[contributing guide](../../CONTRIBUTING.md) for current contribution steps and the target branch’s live rules for merging.

## Release cadence and compatibility work

The community plans approximately **one Beta per month and one Stable per quarter**. Betas support early adoption and
downstream feedback. Stable releases provide accepted deployment baselines. Actual dates and scope depend on candidate
validation and maintainer review. Follow the [release process](release.md) for SemVer, packaging,
release notes, documentation, known limitations, and upgrade guidance.

The **Q4 2026 CI/CD plan** expands validation from PyTorch 2.9 through the latest stable version on Ascend NPUs and
NVIDIA GPUs, including tracking the two most recent PyTorch releases. This is a roadmap target; use published results
to determine supported combinations. Record unavailable runtime combinations as pending validation with a tracking
issue. A feature marked `DEMO` retains its stated limitations even in a Stable release.

## Downstream integration, including VeOmni

Downstream projects consume HyperParallel as an optional package dependency and release independently. Integration PRs
should provide an installation path, user configuration, runnable example, compatibility boundaries, and reproducible
correctness/performance results. Update the downstream usage guide alongside the adapter.

The planned VeOmni pipeline refreshes a pinned upstream baseline weekly and validates HyperParallel changes before
merge. Adopt a new baseline only after it passes; otherwise retain the previous passing combination and track the
failure. Required integration cases must report results for both project commits and the exact environment.
The initial VeOmni integration targets Ascend; GPU support follows later. These planned checks become merge
requirements when enabled for the integration and must not be described as already running without evidence.

## Integration responsibilities

For the proposed VeOmni integration, HyperParallel owns the package, adapter, VeOmni usage guide, Ascend test resources,
and weekly compatibility validation. VeOmni maintainers retain review, merge, and release decisions in their repository.
