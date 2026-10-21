# Integration validation

Implementation: the official FM-Agent CLI adapter, framework profiles, training contracts,
custom candidate-validation instructions, repository skill/agent/command entries.

## Initial interface checks — 2026-10-11

| Check | Result |
|-------|--------|
| Standalone adapter suite | 36 tests and 23 subtests passed |
| Adapter statement/branch coverage | 95% combined coverage, `coverage.py --branch` |
| Python style, Pylint and Lizard | Passed |
| CI Pylint 3.3.7 configuration and Codespell 2.0.0 | Passed locally |
| Skill frontmatter and AGENTS catalogs | Passed |
| Markdown lint and local documentation links | Passed |
| Official `doctor` | Passed against `341aed7a60262186a34dc0340283a099d3195951` |
| Official DTensor `estimate` | Passed; 27 source files, approximately 526 functions |
| Official launch with invalid model credentials | Rejected as a failed run; no successful analysis reported |

The unit suite executes a strict CLI fixture in temporary Git repositories. It covers
custom training-framework profiles, config-relative knowledge, source boundaries,
paths with spaces, fresh/resume/incremental modes, scope preservation, engine setup
errors, subprocess exit codes, stale artifacts and independence from Torch imports.

The real-engine tests used a separate Python 3.12 environment. The Windows host lacked
the C++ compiler required by the optional ArkTS parser, so that package was excluded
for this Python-only integration check. No official engine source was modified.
The adapter's UTF-8 environment handling resolved the upstream TOML decoding failure
under the Windows GBK locale.

## Reproduce

Use the commands in [engine-setup.md](engine-setup.md#adapter-validation) for unit and
style checks. For coverage, put the data file outside the repository or clean up your
local coverage output after inspection:

```bash
python -m coverage run --branch --include '*/formal-verification/scripts/verify.py' -m pytest --confcutdir=tests/ut/agent tests/ut/agent/formal_verification -q
python -m coverage report -m
```

Install the referenced official checkout and run `doctor`, followed by `run --profile
dtensor --mode estimate`, to repeat the real interface checks. `estimate.json` retains
the exact included/excluded scope; it contains no invented cost estimate when prior
run history is unavailable.

## Initial test boundary

No LLM-backed full verification or accelerator bug reproduction was performed in the
initial interface checks. The estimate is source preflight, not verification of the 526 functions. A full
bug hunt requires a configured model backend and a framework probe environment; device
claims additionally require the appropriate hardware. The results above validate the
integration and its failure handling, not the correctness of the training framework.

## Runtime validation — 2026-10-12

The seven built-in profiles resolve existing source directories through `plan`. These
plans make no model calls. A separate editable HyperParallel checkout on pod116-209
passed the 36 adapter tests, a single-NPU tensor operation and a two-rank HCCL sum.
The probe environment used Python 3.10.20, PyTorch 2.9.0+cpu and torch-npu 2.9.0.post2;
FM-Agent used a separate Python environment. Normal framework imports were checked
against that checkout before candidate execution.

The framework revision analyzed was `27bb47f8050ea956c09868ba930f0575296c0b48`, with
`HEAD:hyper_parallel` equal to `cbc39efe64e462bcfc518861d2620e7e89800bc6`. Later documentation
changes preserve this training-source tree. The LLM run used upstream FM-Agent
`341aed7a60262186a34dc0340283a099d3195951`, the authenticated local Codex CLI backend,
and remote framework probes. Its scope was five files in `hyper_parallel/core/utils`,
extracting 56 functions. This run selected the upstream fallback extractor/call graph
with `[codegraph].version = ""`; it did not use the later native Windows fix.

Candidate validation executed real implementations with analytical or independent
reference controls. Examples of reproduced findings and their boundaries are:

| Function | Executed observation | Contract and scope |
|----------|----------------------|--------------------|
| `differentiable_all_reduce` | Two-rank `op="avg"` returned `[6, 10]` instead of `[3, 5]`; gradients and one SGD step also differed | Arithmetic average is inferred from the accepted operation name. CPU/Gloo and supplementary NPU/HCCL probes reproduced it. SUM and caller-normalized SUM controls passed; callers that explicitly normalize are outside this claim. |
| `p2p_exchange` | In a four-rank Gloo world, subgroup `[1, 2]` returned rank 2's own payload/storage instead of rank 1's payload | The API uses global peer ranks but compares them with a group-local rank. Native subgroup P2P and default-group forward/gradient/SGD controls passed. The payload violation, rather than a timeout alone, establishes the mismatch. |
| `_str_to_reduce_op` | Publicly accepted `Partial("prod")` and `Partial("all")` names raised `ValueError`; public clipping with `Partial("prod")` failed before communication | This is a metadata/name-mapping gap. Fourteen controls passed. No product/all collective or accelerator numerical claim was tested. |
| `_get_total_norm` | On NPU, a parameter without a gradient returned norm `1.0` for finite `norm_type=1e50`; the empty-sum oracle and Torch returned `0.0` | CPU and ordinary norm-order controls passed. This is an extreme finite-order boundary on the tested NPU stack, with no distributed collective. |
| `MoEMonitorCallback.on_step_end` | Two finite float64 auxiliary losses of `1e308` produced stored mean `inf`, while the high-precision reference was `1e308` | Real bias updates and count resets passed. The input loss states were constructed fixtures; the ordinary built-in MoE forward emits float32 losses and did not generate this trigger. |

Candidate reports also rejected extraction-fragment syntax claims and side effects
introduced by caller-defined Python subclasses. Those observations do not establish
supported training-framework bugs. A supplementary manual audit of native AVG
reduce-scatter backward found a gradient/SGD mismatch on the tested Torch/NPU versions;
the native Torch control showed the same behavior. This separate dependency-bound
finding does not change the original FM-Agent candidate's `not_confirmed` result.

Keep the original report, per-function results, all validator attempts, probes, raw
stdout/stderr and exact source/runtime metadata in the run artifacts. Completion must
be audited separately: every extracted function needs a finished reasoning result,
every candidate needs a terminal validation result, and the official report/version
record must exist. A zero exit code alone is insufficient. Generated run artifacts
remain outside the committed agent module.

## Native Windows engine fix

[FM-Agent PR #242](https://github.com/fmagent-project/FM-Agent/pull/242), tested at
`ed31bed6fb230b6604f9d275086073bc05deb90c`, adds native Windows CodeGraph provisioning
and preserves qualified method identities. See [engine setup](engine-setup.md#windows-native-codegraph)
for selecting that external engine revision.

| Check on the patched engine | Observed result |
|-----------------------------|-----------------|
| Provisioning regression suite, Windows x64 / Python 3.12 | 25 passed |
| Provisioning regression suite, Linux aarch64 / Python 3.13 | 24 passed; one Windows-only lock test skipped |
| Native pinned CodeGraph installation, Windows and Linux | Passed |
| Native extraction of the five HyperParallel utility files | 56 functions, 42 resolved call edges; source bodies preserved |
| Two-class fixture with duplicate `forward` method names, Unicode comments and a path containing spaces | Full LLM pipeline completed with two finished `MATCH` results, report and version record |
| Resume of that completed fixture | Passed; both results retained, no pending verification |
| Fork CI, Windows and Ubuntu, at the tested revision | Passed |

The upstream first-contribution workflow still requires maintainer approval. Fork CI
is not an upstream approval. The complete native Windows fixture and HyperParallel
extraction checks are separate from the larger HyperParallel fallback run above.
Neither `MATCH` nor these integration checks constitute a machine-checked proof,
whole-framework coverage, or support for every optional language backend.
