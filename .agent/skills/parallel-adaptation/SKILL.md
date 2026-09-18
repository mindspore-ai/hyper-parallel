---
name: parallel-adaptation
description: >
  Add, review, or standardize model-family TP, CP, and EP support in
  HyperParallel using ModelAdapterSpec, ShardingPlan/ModuleShardingSpec,
  semantic ownership contracts, and layered parity tests. Use for new model
  parallel adaptation, 通用化, capability matrices, or fast TP/CP/EP onboarding.
---

# Parallel Adaptation

Adapt model semantics through the existing registry, plan, and injection contracts. Do not add model-class branches to
generic builders or copy an entire model implementation when an adapter can express the difference.

Load `.agent/rules/model-development-validation.md`, `.agent/rules/distributed.md`, and the applicable platform/testing
rules.

## Required Inputs

- model family, architecture/model-type identifiers, and exact runtime module classes;
- model adapter registration and available provider fields;
- parameter names/layouts, tied weights, attention/MLP/MoE forward boundaries, and cached metadata;
- desired mesh axes and legal TP/CP/EP combinations;
- batch/sequence/media fields and which axes own samples, tokens, heads, vocabulary, and experts;
- loss, gradient synchronization, checkpoint, and generation requirements;
- available accelerator topology for smoke and parity.

Ask only when the model identity, target axes, or hardware ceiling cannot be derived.

## Workflow

1. Read [standard-contract.md](references/standard-contract.md). Produce a capability declaration with supported,
   unsupported, and unverified axes/combinations before editing.
2. Register model-family behavior through `hyper_parallel/models/adapter_spec.py` and `models/<family>/adapter/`. Keep
   generic behavior in `distributed/` or `components/`; keep family naming/forward glue in the adapter.
3. Express parameter and boundary layouts through `ShardingPlan` / `ModuleShardingSpec`. Use existing semantic templates,
   placement rules, `tp_divide_attrs`, and explicit injection points. Avoid path-only heuristics when a runtime class or
   adapter provider can make the contract explicit.
4. Implement one axis at a time in this order unless the model requires otherwise: TP, CP, EP. Prove the single-axis path
   before adding pairwise or hybrid plans.
5. Preserve data and result ownership: TP/CP/EP ranks in one model group read the same logical batch; only data replicas
   receive distinct samples; only one rank/group owner writes results and metrics.
6. Add tests from [test-matrix.md](references/test-matrix.md). Run structure/contract first, then real smoke and
   single-device-vs-parallel parity with `accuracy-validation` tolerances.
7. Update the capability matrix only for combinations actually executed. Unsupported combinations fail early with a
   descriptive error; unverified combinations remain explicit.

## Axis Checklist

- TP: column/row sharding, deferred row bias, local head/expert metadata, vocab/loss parallel, tied weights, partial
  gradients, identical batches inside the TP group, and checkpoint resharding.
- CP: global-to-local sequence mapping, attention offset/mask semantics, packed/varlen metadata, positions and labels,
  tail padding, output ownership, sequence-local gradients, and generation cache layout.
- EP: global expert-to-rank-to-local-expert mapping, variable token counts, dispatch/combine inverse mapping, routing and
  aux-loss semantics, expert-data-parallel groups, zero-token experts, and grouped-kernel fallback.
- Hybrids: unique mesh-axis meaning, communication order, no duplicate gradient reduction, loss/token denominator groups,
  tied/shared parameters, tail batches, checkpoint metadata, and cleanup after failure.

## Output

- List adapter, plan, component, config, and test files changed.
- Print the final capability matrix and plan explanation for the target boundaries.
- State sample/token/head/expert/result ownership and all reduction groups.
- Report each evidence level and actual topology; list unverified combinations as gaps.

## Read On Demand

- [standard-contract.md](references/standard-contract.md): adapter/plan boundary, capability declaration, and axis contracts.
- [test-matrix.md](references/test-matrix.md): minimum TP/CP/EP and hybrid validation.
- `.agent/skills/accuracy-validation/SKILL.md`: quantitative parity workflow.
- `hyper_parallel/distributed/recipe_spec.py` and `hyper_parallel/distributed/plan.py`: source-of-truth plan contract.
