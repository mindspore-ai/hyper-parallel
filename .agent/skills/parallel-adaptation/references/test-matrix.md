# Parallel Adaptation Test Matrix

## Common Layers

| Layer | Minimum checks |
| --- | --- |
| Structure | Adapter registration, provider discovery, plan construction, config parsing, and fail-fast capability checks |
| Contract | Placement/ownership tables, sample identity, shapes, masks, token counts, group membership, and plan explanation |
| Smoke | One real forward/backward/update on a nontrivial axis size |
| Parity | Same weights and deterministic batch on reference vs parallel path; outputs, loss sum/count, gradients, and update |
| Resume | Uninterrupted vs save/resume next batch and next update |
| Impact | Communication/memory/throughput only after parity passes |

## TP

- TP sizes 1 and 2; add 4 when the model/default topology claims it.
- Column/row linears, bias, Q/K/V head metadata, MLP, vocab head/loss parallel, tied weights, and uneven forbidden shapes.
- Rank-identical batch fingerprints within each TP group.
- Targets on both sides of a vocab partition and `ignore_index` for per-token cross entropy.
- Replicated parameters with partial gradients and parameters that must not be summed.

## CP

- CP sizes 1 and 2 with sequence lengths divisible and non-divisible by CP.
- Left/right padding, causal offset for rank greater than zero, packing/varlen metadata, positions, shifted labels, and masks.
- The same logical sample on all CP ranks before slicing; local slices reconstruct the global sequence exactly once.
- Loss and post-sync gradient-norm parity. Start with relative loss error at most `0.005` and gradient-norm error at most
  `0.02`, consistent with the standard accuracy tier; tighter pointwise checks apply where deterministic.
- Generation/prefill/cache tests are separate from training attention parity.

## EP

- EP sizes 1 and 2, top-k 1 and greater than one, balanced and highly imbalanced routing, and zero-token local experts.
- Global expert owner, local expert index, send/receive counts, dispatch order, combine inverse, gate weights, and aux loss.
- Reference expert loop vs distributed experts. Use the standard tier; a grouped/fused kernel may use the relaxed tier only
  when declared before execution.
- Dense gradients reduce over the dense data group; expert gradients reduce only over expert replicas, excluding owner EP.

## Hybrids

Test changed pairwise combinations and one flagship supported topology. Include:

- sample identity within model groups and distinct samples across data replicas;
- unique parameter ownership and no duplicate gradient SUM/all-reduce;
- token denominator counted once despite CP or TP replication;
- tail micro-batches where some data owners have padding but all model ranks preserve collective order;
- high-performance module off/on when the model adapter enables a replacement by default;
- checkpoint metadata and restore on the same topology, plus cross-topology restore only when explicitly supported.

Use `torchrun_case`, `msrun_case`, or `parallel_case` according to `.agent/rules/testing.md`. Launcher modules remain free of
framework and HyperParallel imports.
