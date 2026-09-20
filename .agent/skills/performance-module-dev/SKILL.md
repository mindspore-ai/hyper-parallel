---
name: performance-module-dev
description: >
  Add, integrate, track, or review HyperParallel high-performance functions and
  modules with accuracy-first acceptance, model-adapter replacement wiring,
  benchmark evidence, fallback behavior, and default-on/open-box performance.
  Use for fused kernels, attention, MoE, norms, loss, optimizer, quantization,
  performance-module roadmaps, or 开箱性能.
---

# High-Performance Module Development

Keep reusable computation in `hyper_parallel/components`, model-family mapping in `models/<family>/adapter`, and parallel
ownership in `distributed`. A fast kernel does not own Trainer lifecycle or benchmark semantics.

Load `.agent/rules/model-development-validation.md`, `.agent/rules/code-style.md`, and the applicable distributed/platform
rules.

## Required Inputs

- operation/module semantics and reference implementation;
- target model families, hardware, backend, dtype, shapes, and parallel axes;
- source parameter names/layouts and replacement/conversion requirements;
- optional dependency and fallback behavior;
- accuracy tier, representative workloads, and performance objective;
- expected default policy: opt-in, auto-selected, or default-on.

Ask only when the mathematical reference, hardware scope, or default policy cannot be derived.

## Workflow

1. Read [module-lifecycle.md](references/module-lifecycle.md). Search existing `components/functional`, `components/modules`,
   model replacements, and optional operators before adding a new abstraction.
2. Freeze reference semantics and edge cases. Add the function first when it is stateless; add a module only when parameters,
   buffers, state, or a drop-in module contract are required.
3. Keep imports lazy for optional hardware operators. Unsupported hardware/dtype/shape combinations either use a tested
   fallback or fail early; never silently run a different mathematical operation.
4. Add family-specific replacement factories and weight transforms under `models/<family>/adapter`. Apply replacements
   before optimizer construction and sharding/checkpoint materialization in the order required by the Trainer.
5. Run standalone forward/backward parity, then the 2x2 optimized-module/parallel matrix through `accuracy-validation`.
6. Read [acceptance.md](references/acceptance.md) and measure warmup, time-to-first-step, steady-state step time/throughput,
   memory, and MFU when meaningful. Compare the same effective batch and workload.
7. Promote to auto-selected/default-on only after the default policy passes the declared hardware matrix, has a visible
   capability decision and fallback, and improves an end-to-end metric without unacceptable startup or secondary-workload
   regression.
8. Update the module inventory and algorithm watchlist with source revision, license, owner, last validation, supported
   matrix, and next action. Tracking an upstream algorithm is not evidence that HyperParallel supports it.

## Initial Inventory And Watch Areas

- Existing/common: RMSNorm/offset norm, RoPE, SwiGLU/MLP, attention variants, grouped experts/matmul, token
  permute/unpermute, auxiliary loss, chunked/linear/vocab-parallel CE, mixed-precision optimizers, quantized linear/GMM.
- Track by model impact: sparse/linear attention, MLA/DSA, delta/recurrent attention, MTP/speculative objectives, DeepEP or
  equivalent dispatch, fused optimizer/update, offload/prefetch, low-precision training, and newly published model-family
  blocks.

The watchlist is prioritized by supported models, user demand, measured bottlenecks, implementation maturity, license,
backend availability, and a credible fallback. Do not add a module solely because an algorithm is new.

## Output

- State component layer, public API, adapter mappings, dependency/fallback, and supported matrix.
- Report standalone and composition accuracy evidence before performance.
- Report raw repeated measurements and aggregate statistics, not only a speedup percentage.
- State default policy and why it is safe, or keep the feature opt-in.
- List remaining hardware/model/topology gaps and watchlist follow-ups.

## Read On Demand

- [module-lifecycle.md](references/module-lifecycle.md): component/adapter boundaries and upstream tracking.
- [acceptance.md](references/acceptance.md): numerical, performance, fallback, and default-on gates.
- `hyper_parallel/components/modules/README.md` and `components/functional/README.md`: current public components.
- `.agent/skills/accuracy-validation/SKILL.md`: parity and 2x2 composition matrix.
