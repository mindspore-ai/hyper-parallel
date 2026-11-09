# Proposal to Integrate HyperParallel into VeOmni

**English** | [简体中文](hyperparallel_community_operations.zh.md)

**Q4 2026 · For contributors, framework maintainers, and downstream integration teams · Updated November 5, 2026**

How an acceleration capability reaches the main branch, passes multi-device validation, and becomes an optional backend for VeOmni users training on Ascend.

HyperParallel is a distributed acceleration library built on native PyTorch for Ascend NPUs and NVIDIA GPUs. Training frameworks can adopt its parameter sharding, tensor parallelism, pipeline parallelism, context parallelism, expert parallelism, and distributed checkpointing as needed. The community uses public issues and PRs to discuss requirements, review interfaces, and record validation results. [Project homepage](https://github.com/mindspore-ai/hyper-parallel) · [Q4 technical roadmap](https://github.com/mindspore-ai/hyper-parallel/issues/1158)

The community plans **approximately one Beta per month and one Stable per quarter**. Betas allow contributors and downstream projects to adapt early; Stable releases provide accepted, reproducible integration baselines. In Q4 2026, the community plans to expand CI/CD coverage from PyTorch 2.9 through the latest stable version on Ascend NPUs and NVIDIA GPUs, while strengthening collaboration with PyTorch ecosystem projects.

## Community Proposal for Integrating HyperParallel into VeOmni

1. **Capabilities submitted for approval.** This proposal asks VeOmni to approve optional Python package integration for HyperParallel FSDP, the Muon optimizer, high-performance modules, and MegaMoE. The GPU backend is outside this initial scope. The RFC will list each high-performance module, how to enable it, and applicable models.
2. **Reproducible benefits and correctness.** Extend the existing experiments to cover scenarios and acceptance requirements requested by the VeOmni community. The full report records models and configurations, both code versions, the Ascend environment and device count, native-backend comparisons, performance gains, and numerical error. Both the RFC and merge PR include the full report; PR results correspond to the final tested commit. [Existing experiments](https://github.com/mindspore-ai/hyper-parallel/issues/637)
3. **Initial support matrix and user boundaries.** The first release targets ByteDance internal users training with VeOmni on Ascend NPUs. For each model, the RFC support matrix lists validated acceleration capabilities, Ascend devices, VeOmni and HyperParallel versions, PyTorch, torch-npu, and CANN versions, with links to the corresponding results.
4. **A minimal example reviewers can run.** Submit a “Using HyperParallel as an Ascend Acceleration Backend” guide to the VeOmni repository with the adaptation PR. Include commands to install validated versions, a minimal training configuration, launch commands, and expected output. Reviewers can then start multi-device training from a clean environment and confirm that the selected acceleration is active.
5. **Maintenance responsibilities and incident handling.** HyperParallel owns routine Ascend backend maintenance and first-line incident handling. It maintains packages, adaptation code, and the guide in VeOmni, supplies Ascend test resources, and validates compatibility with VeOmni weekly.

### Integration Method and Progress

HyperParallel's Q4 roadmap includes an optional FSDP2 backend and Muon optimizer for VeOmni. An implementation exists on an adaptation branch; upstream integration into VeOmni remains a tracked task. VeOmni retains training orchestration and DeviceMesh construction, while HyperParallel supplies capabilities explicitly selected by users. The projects can keep independent version numbers and publish tested compatibility tables showing which combinations can be installed together. [Integration tracking issue](https://github.com/mindspore-ai/hyper-parallel/issues/1124) · [VeOmni releases](https://github.com/ByteDance-Seed/VeOmni/releases)

**Integration uses an optional Python package dependency.** HyperParallel builds and releases its own wheels; adaptation code calling its public APIs ships with VeOmni. For formal integration, declare an optional dependency in VeOmni's `pyproject.toml` and record validated versions in `uv.lock`. Users install dependencies for the capabilities they enable; the default PyTorch path remains independently usable. [VeOmni dependencies](https://github.com/ByteDance-Seed/VeOmni/blob/main/pyproject.toml)

### Shared Q4 2026 Roadmap Priorities

VeOmni's overall roadmap spans multiple hardware platforms, while its Ascend-specific plan identifies NPU requirements. HyperParallel's Q4 roadmap explicitly proposes optional FSDP2 and Muon support for VeOmni. The table compares their technical goals; adopting HyperParallel implementations still requires VeOmni community review. [VeOmni Q4 roadmap](https://github.com/ByteDance-Seed/VeOmni/issues/1365) · [VeOmni Ascend plan](https://github.com/ByteDance-Seed/VeOmni/issues/1405) · [HyperParallel Q4 roadmap](https://github.com/mindspore-ai/hyper-parallel/issues/1158) · [Integration proposal](https://github.com/mindspore-ai/hyper-parallel/issues/1124)

| Shared direction | VeOmni Q4 goal | HyperParallel Q4 goal and integration boundary |
| --- | --- | --- |
| Low-precision FSDP2 communication and training | Quantize parameter AllGather on Ascend and adapt low-precision ReduceScatter for HCCL; the overall roadmap also includes FP8 computation and quantization-aware training (QAT). [VeOmni low-precision RFC](https://github.com/ByteDance-Seed/VeOmni/issues/1360) | Advance FP8 weight communication for FSDP2 and MXFP4 QAT. Reducing parameter AllGather communication is the most direct overlap; HCCL ReduceScatter and quantized export require separate validation. [FP8 weight communication](https://github.com/mindspore-ai/hyper-parallel/issues/1165) · [MXFP4 QAT](https://github.com/mindspore-ai/hyper-parallel/issues/1170) |
| Muon state offload | Asynchronously offload AdamW and Muon states on Ascend, overlapping transfers with computation while preserving update accuracy and checkpoint restorability. [VeOmni NPU memory optimization](https://github.com/ByteDance-Seed/VeOmni/issues/1405) | Store Muon momentum in pinned CPU memory and move it to the device for updates. Share a multi-device training case comparing peak device memory, step time, parameter updates, and restored optimizer states. [Muon offload RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1077) |
| MoE expert load balancing and execution | Reduce expert hotspots and memory peaks caused by uneven token distributions. Qwen3.5-MoE already has work using temporary expert replicas to share hotspots. [VeOmni expert replication](https://github.com/ByteDance-Seed/VeOmni/pull/1112) | Plan hot/cold expert migration and load balancing; MegaMoE provides fused expert dispatch, computation, and combine. Agree on ownership of expert layout decisions, and measure scheduling benefits separately from fused-operator benefits. [Expert migration RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1129) · [MegaMoE RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1121) |
| Long-context parallelism | Extend hybrid Ring Attention/Ulysses and validate KV-AllGather context parallelism (CP) on Ascend sparse-attention models. [Ascend hybrid parallelism](https://github.com/ByteDance-Seed/VeOmni/pull/1183) · [KV-AllGather CP](https://github.com/ByteDance-Seed/VeOmni/pull/1290) | Also plan Ring + Ulysses, KV AllGather, and multimodal CP. Share variable-length packing boundary, gradient, and memory tests, and validate composition with FSDP and expert parallelism. [Hybrid CP](https://github.com/mindspore-ai/hyper-parallel/issues/1107) · [Multimodal CP](https://github.com/mindspore-ai/hyper-parallel/issues/1118) |
| Selective recomputation and activation offload | Extend layer- and operator-level recomputation, add asynchronous activation offload, and skip redundant backward computation where applicable. [VeOmni selective recomputation](https://github.com/ByteDance-Seed/VeOmni/pull/1166) · [Ascend plan](https://github.com/ByteDance-Seed/VeOmni/issues/1405) | Plan fine-grained MoE recomputation, Activation Swap improvements, and recomputation-boundary memory optimization. Compare gradients, peak memory, and step time under the same configuration; accept backward skipping separately. [MoE recomputation](https://github.com/mindspore-ai/hyper-parallel/issues/1161) · [Activation offload](https://github.com/mindspore-ai/hyper-parallel/issues/1079) |

#### Models and Tests That Can Be Shared

- **Models and high-performance modules.** MoE/VL scenarios in DeepSeek V4.1-Flash, Kimi K3, and the Qwen3 family can serve as shared validation targets. VeOmni supplies training entry points and model configurations; HyperParallel supplies replaceable attention, expert, or MLP modules. Fix one model and replacement location per experiment and record numerical and performance results before and after replacement. [VeOmni model plan](https://github.com/ByteDance-Seed/VeOmni/issues/1405) · [HyperParallel model list](https://github.com/mindspore-ai/hyper-parallel/issues/1166) · [High-performance module RFC](https://github.com/mindspore-ai/hyper-parallel/issues/1126)
- **Multimodal data balancing.** Both projects address DP-rank load differences caused by varying sample lengths and multimodal inputs. Use the same samples to measure the slowest rank's time and memory. This is a different scheduling stage from token balancing between MoE experts. [VeOmni multimodal balancing](https://github.com/ByteDance-Seed/VeOmni/issues/1074) · [HyperParallel data balancing](https://github.com/mindspore-ai/hyper-parallel/issues/1155)
- **Large-model loading and checkpoint restoration.** VeOmni plans reads by target parameter shard; HyperParallel plans parallel DCP reads, broadcasts, and restoration. Share load-time, peak-memory, and resumed-training consistency tests; validate checkpoint-format compatibility separately. [VeOmni sharded loading](https://github.com/ByteDance-Seed/VeOmni/issues/1297) · [HyperParallel DCP loading](https://github.com/mindspore-ai/hyper-parallel/issues/1178) · [Restoration scenarios](https://github.com/mindspore-ai/hyper-parallel/issues/1179)
- **Compilation and hardware regression.** VeOmni focuses on compilation compatibility for dynamic-length attention and real Ascend validation of models, numerics, and checkpoints; HyperParallel advances full-graph compilation for EP and CP. Representative training cases can be reused for weekly integration and PR validation. [VeOmni compilation goals](https://github.com/ByteDance-Seed/VeOmni/issues/1365) · [VeOmni hardware validation](https://github.com/ByteDance-Seed/VeOmni/issues/1405) · [HyperParallel EP compilation](https://github.com/mindspore-ai/hyper-parallel/issues/1112) · [CP compilation](https://github.com/mindspore-ai/hyper-parallel/issues/1113)

**Prioritize two groups of Ascend integration scenarios.** First, validate numerics, device memory, and checkpoint restoration with multi-device FSDP2 + Muon training. Second, enable high-performance modules and MegaMoE separately on one MoE model, retaining VeOmni's training entry point and expert layout, and measure replacement benefits. The initial review still covers four capabilities: FSDP, Muon, high-performance modules, and MegaMoE. Low-precision QAT, long-context CP, and other directions require separate agreement.

## HyperParallel Community Merge Workflow

| Stage | Contributor and maintainer responsibilities |
| --- | --- |
| Report a problem or propose a design | Bug reports include versions, hardware, and minimal reproduction steps. New features or public-interface changes use a Feature Issue or RFC to explain use cases, compatibility, and acceptance criteria; relevant module maintainers review the design. |
| Implement and check standards | Authors implement changes according to repository conventions and check style, complexity, and documentation formatting before submission. |
| Submit a PR | Each PR focuses on one independently verifiable change, organized as one commit with prefixes such as `feat:`, `fix:`, or `docs:`. Link the issue and explain motivation, scope, and measured results; update documentation for API or user-visible changes. |
| Provide test evidence | UT covers independently testable logic. Changes involving communication, sharding, parallel combinations, or resumed training include distributed ST with devices, device count, software versions, and results. Performance changes include a same-configuration baseline and measurement method. Explicitly list unvalidated scenarios. |
| Review and merge | Request module-owner review through CODEOWNERS. The current GitHub main branch requires Code Owner approval, independent approval of the latest push, resolved review discussions, and both CI and reviewer-count checks. Obtain valid results for the new commit after pushing changes. |

**Coding standards and pre-commit checks**

- **Languages and formatting.** Python files begin with the Apache 2.0 header, follow PEP 8, and use approximately 120 columns. Classes use PascalCase; functions and variables use snake_case. C++ follows the repository's .clang-format with a 120-column limit.
- **Interfaces and implementation.** Public functions require type annotations; public functions and methods use Google-style docstrings. Imports normally appear at module scope, ordered as standard library, third-party libraries, and project modules. Distributed implementations use native PyTorch APIs directly.
- **Pre-commit checks.** After installing repository hooks, commits run pylint, lizard, and markdownlint on applicable files. Authors fix findings and explain necessary exceptions in the PR. Reviewers check that public-interface documentation and types match implementations.

[Repository coding standards](https://github.com/mindspore-ai/hyper-parallel/blob/master/.agent/rules/code-style.md) · [Pre-commit configuration](https://github.com/mindspore-ai/hyper-parallel/blob/master/.pre-commit-config.yaml)

Use the repository's contribution templates. The PR template covers five self-check areas: design, testing, outcome validation, public interfaces, and documentation. [Issue templates](https://github.com/mindspore-ai/hyper-parallel/issues/new/choose) · [PR template](https://github.com/mindspore-ai/hyper-parallel/blob/master/.github/PULL_REQUEST_TEMPLATE.zh-CN.md) · [Code ownership](https://github.com/mindspore-ai/hyper-parallel/blob/master/CODEOWNERS) · [Contribution rules](https://github.com/mindspore-ai/hyper-parallel/blob/master/AGENTS.md)

### Current Merge Gates

The GitHub main branch has two independent required checks. Jenkins runs tests and reports results to the PR; GitHub checks whether those results remain valid. Another workflow checks whether the current commit has enough formal approvals. [Effective main-branch rules](https://api.github.com/repos/mindspore-ai/hyper-parallel/rules/branches/master) · [Review and merge workflow](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/github_auto_merge_zh.md)

- **Test gate `label-check`.** Authors complete the template's description checks; after passing, Jenkins test, coverage, and build details are available on the PR. The labels `ci-pipeline-passed` and `ci-pipeline-passed-but-overtime` remain valid for 24 hours after being added. The overtime label preserves evidence of abnormal duration. New commits or expired results require fresh valid results for the code to be merged. [Label validity rules](https://github.com/mindspore-ai/hyper-parallel/pull/1365) · [Jenkins reporting example](https://github.com/mindspore-ai/hyper-parallel/pull/1438#issuecomment-6604110750)
- **Review gate `all-reviewers-approved`.** With one reviewer, one approval is required; with two or more, two different reviewers must approve. Approvals must target the current commit, and the author does not count. Outstanding change requests and pending team-review requests still block success. Native rules such as Code Owner approval remain required after the count is reached. [Review rules and examples](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/github_auto_merge_zh.md)
- **Failure handling.** Authors diagnose failures from the stage and logs; module maintainers help fix code regressions, while CI maintainers handle runtime environment failures. After retesting, use results for the latest commit. Main-project UT and distributed ST have different execution requirements; RL system tests currently run separately and require separate result checks. [Testing standards](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/testing.md) · [RL test scope](https://github.com/mindspore-ai/hyper-parallel/blob/master/AGENTS.md)

## VeOmni Validation Before Merge

The Q4 plan adds a VeOmni compatibility pipeline to **HyperParallel CI/CD**. Weekly jobs track VeOmni; every HyperParallel PR trains with VeOmni using a package built from that commit, exposing downstream regressions before merge.

| Trigger | Pipeline work | Passing criteria |
| --- | --- | --- |
| Weekly VeOmni update | Pin a specific VeOmni upstream commit and record its release tag, if any. Until adaptation is upstream, also pin its matching patch. Run integration tests with the HyperParallel main-branch package. | Adopt the combination as the PR baseline only after every test passes. On failure, retain the previous passing baseline and record the failing combination, logs, and owner; validate again after fixing. |
| Before every HyperParallel PR merges | Build a wheel from the latest PR commit and install it with pinned VeOmni in a clean environment. Validate multi-device Ascend FSDP2/HSDP forward/backward, parameter updates, same-topology save/restore, and numerical comparison with the native backend. Run the RFC-defined cases separately for integrated Muon, high-performance modules, and MegaMoE. | Report both commits, PyTorch version, devices, and per-case results to the PR. Required cases that fail, skip, or lack results block merge. Revalidate after new PR commits or a changed VeOmni baseline. |
| Before Beta / Stable releases | Install the candidate package in a clean environment; validate imports, quick start, and declared supported combinations. Stable also validates upgrade paths and representative downstream workloads. | Deliver packages, exact dependencies, compatibility tables, known limitations, and upgrade instructions. |

## Supporting Multiple PyTorch Versions and Hardware Platforms

The Q4 goal covers **PyTorch 2.9 through the latest stable version at that time**, with inspectable results on Ascend NPUs and NVIDIA GPUs. This VeOmni integration accepts only Ascend combinations. Each release line should specify exact PyTorch minor versions. NPU records also include device model, torch-npu, and CANN versions; GPU records include device model and CUDA version. If a platform runtime has not adapted to a new PyTorch release, mark the combination “pending validation” and link a tracking issue.

Meeting PyTorch Ecosystem Project standards is a Q4 goal. For compatibility, the community will track the two most recent PyTorch releases and provide auditable CI results for each claimed architecture. Formal inclusion depends on the ecosystem working group's review. [Official PyTorch criteria](https://github.com/pytorch-fdn/tac/blob/main/docs/governance/PyTorch_Ecosystem_Process.md#ecosystem-projects-requirements)

| Test target | Hardware | Q4 validation scope |
| --- | --- | --- |
| HyperParallel | Ascend NPU | Basic regression from PyTorch 2.9 to the then-current stable release, by exact minor version and NPU runtime. |
| HyperParallel | NVIDIA GPU | Basic regression over the same PyTorch version range, by exact minor version and CUDA environment. |
| HyperParallel + VeOmni | Ascend NPU | Select measured baselines where both project versions and runtimes match, and execute this integration's acceptance tests. |
| HyperParallel + VeOmni | NVIDIA GPU | Outside this VeOmni support scope. |

Every validation records the model, enabled capabilities, both versions or commits, PyTorch version, device model and count, runtime, results, and report link. Actual supported combinations are defined by the corresponding report.

### Test Schedule

| Suggested timing | Validation scope | Results delivered to the community |
| --- | --- | --- |
| Every PR | Check affected modules' style, UT, and necessary multi-device ST, prioritizing framework versions and hardware touched by the change. | Results for the current commit, failure logs, and reproduction conditions. |
| Regular main-branch regression | Rotate through the version matrix on both hardware types, focusing on numerical accuracy, communication correctness, throughput, peak memory, and checkpoint restoration. | Latest successful record for each combination, regressions, and responsible modules. |
| Before Beta / Stable releases | In clean environments, test candidate installation, imports, and quick start, then multi-device regression for combinations declared supported. Stable additionally checks upgrade paths and representative downstream workloads. | Installable artifacts, exact dependencies, compatibility tables, known limitations, and upgrade instructions. |

This table is a proposed Q4 implementation schedule. Users should select environments from the measured combinations in the relevant release notes. Existing strict builds, package validation, and human release review provide a basis for matrix expansion. [Release process](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/release.md)

## Release Cadence

The projects version and release independently. Each commit checks its PR, and weekly jobs refresh the VeOmni integration baseline. HyperParallel plans monthly Betas and quarterly Stable releases, with exact dates determined by candidate acceptance results.

| Channel | Primary users | How the community uses the window |
| --- | --- | --- |
| **Beta · Approximately monthly** | Feature contributors, hardware adapters, and downstream maintainers | Provide installable candidates listing new capabilities, interface changes, and known limitations. Downstream projects report integration issues before the next Stable, including versions, environments, and reproductions. |
| **Stable · Approximately quarterly** | Training teams needing repeatable deployment and downstream official releases | Finalize scope based on Beta feedback. After validating target combinations and completing human review, publish tags, packages, and release notes with changes, limitations, upgrade guidance, contributors, and merged PRs. |

Versions follow `vMAJOR.MINOR.PATCH`: incompatible APIs increment MAJOR, backward-compatible features increment MINOR, and backward-compatible fixes increment PATCH. Monthly and quarterly windows are plans; dates, features, and scope are finalized with candidates. Release notes also distinguish maturity: a capability marked `DEMO` retains its own usage boundaries even when shipped in Stable. [Versioning and release standards](https://github.com/mindspore-ai/hyper-parallel/blob/master/docs/contributing/release.md) · [Official release-note example](https://github.com/mindspore-ai/hyper-parallel/blob/master/RELEASE_CN.md)

### Downstream Release Coordination

1. **Beta integration.** Each project assigns maintainers and selects a VeOmni commit and HyperParallel Beta package. Begin with Ascend multi-device FSDP2/HSDP, comparing numerics with the native backend and saving/restoring model and optimizer states under the same parallel topology. Record Muon acceptance separately.
2. **Stable finalization.** HyperParallel publishes an accepted package; VeOmni retests its candidate containing adaptation code. Both release notes list exact project versions, installation sources, PyTorch and hardware runtimes, supported models, and parallel topologies.
3. **Subsequent upgrades.** When one project releases a compatible patch, the other retests its existing version and updates the compatibility table after success. Announce public-interface or checkpoint-format changes during the monthly Beta and retain upgrade guidance for users on older combinations.

**HyperMulticore now has a separate repository.** As part of the Q4 2026 community plan, HyperParallel created a standalone HyperMulticore repository on October 28. It maintains Ascend multicore operators and the PyTorch runtime for MegaMoE, MegaGate, and MegaMHC. Contributors can find build guides and examples or submit related issues and PRs there. [HyperMulticore repository](https://github.com/mindspore-ai/hyper-multicore)

## VeOmni Community Maintenance

HyperParallel owns routine Ascend backend maintenance and first-line incident handling. VeOmni maintainers review adaptation changes in their repository and decide merges and releases.

Record primary and backup owners from both projects in the RFC. HyperParallel appoints an integration maintainer responsible for user reports, reproduction, and tracking through closure. VeOmni appoints an interface maintainer to review configuration, training entry points, and adaptation changes. For MegaMoE implementation issues, the HyperParallel integration maintainer coordinates a HyperMulticore fix while the user keeps the original issue.

| Scenario | Lead and response | Recovery criteria |
| --- | --- | --- |
| Regression introduced by a HyperParallel PR | The PR author fixes it; the integration maintainer supplies Ascend resources and logs. Add regression coverage for the failing scenario. | The current PR commit passes required checks before merging into HyperParallel. |
| Weekly tracking of a new VeOmni version fails | The HyperParallel integration maintainer retains the previous passing combination, records the failing commit, and reproduces the issue; for VeOmni interface changes, provide its interface maintainer with a minimal reproduction and adaptation proposal. | Switch baselines only after the new combination passes the same cases; record the validation date in the compatibility table. |
| Post-release training interruption or numerical anomaly | HyperParallel handles Ascend issues; for GPU issues, the maintainer of the failing module leads. | Record root cause, affected and fixed versions, and regression results in the original issue; restore the recommendation after both projects confirm. |

HyperParallel maintains wheels, adaptation code, the VeOmni guide, and Ascend test resources, and runs weekly compatibility validation. VeOmni retains merge and release authority over its repository; each project decides its own blocking scope. Before the first release, define reporting entry points and on-duty contacts, then rehearse a simulated incident from report and rollback through retesting.
