# Fused-operator reference

> **Evidence provenance:** Mixed. General integration practice, with measured
> constraints and fallback behavior from this repository's fused operators.
> See [fusion methodology](fusion-methodology.md) and
> [evidence provenance](../SKILL.md#evidence-provenance).

Fused operators combine small operations to reduce intermediate tensors and kernel
launches. Before integration, verify the **operator constraints**. Unsupported inputs
may fall back to the reference implementation on every call, removing the expected
benefit without producing a warning. Positive evidence of path engagement is required.

> This reference defines the methodology and registration fields; it is not an
> exhaustive catalog. For each new integration, record its constraints and use cases
> here and provide layer-level and end-to-end measurements in the feature document.

## Registration fields

| Field | Description |
|---|---|
| Operator name | Name of the dispatched custom operator |
| Constraints | Supported head counts, `head_dim`, dtypes, shapes, sequence-length limits, and other restrictions |
| Applicable scenario | Replaced computation chain and targeted bottleneck |
| Engagement evidence | Positive confirmation of the fused path, such as an `engaged` flag or INFO log |
| Fallback conditions | Behavior when a constraint is not met, including per-call reference fallback |
| Measurements | Layer-level speedup and end-to-end gain from consecutive runs on the same node, with commands, commit, and environment |

## Integration rules

- **Verify engagement positively.** Both reference fallback and successful fused
  execution may produce no warnings. Require an engagement marker such as `engaged=1`
  emitted by the measured process.
- **Make eligibility checks explicit.** Unsupported head counts, dtypes, shapes, or
  lengths must use the documented fallback and must not be reported as fused execution.
- **Declare numerical criteria.** Prefer elementwise equality for selection results.
  Record BF16 tie differences below 1 ULP as known numerical noise. Do not silently
  reduce score precision.
- **Asynchronous failures escape synchronous fallback.** An adapter that catches
  non-OOM exceptions can handle only synchronous failures. Errors such as
  `ACL stream synchronize failed` surface at stream synchronization, outside the
  original `try` scope. Re-raise OOM explicitly to avoid invalid A/B comparisons
  caused by silent degradation.
- **Inspect postprocessing.** Use TopK kernels where integer ArgSort falls back to
  AI CPU. Where equivalent, replace an AI CPU `scatter_reduce` path with `scatter`
  and `view/amax`. Search runtime logs for `npu_cpu_fallback` to locate these hotspots.

## Related references

- Versioned interfaces and training compatibility: [Ascend fusion index](ascend-fusion/index.md).
- Evidence-driven training fusion experiments: [mfu-riser](../../mfu-riser/SKILL.md).
- Automated fusion and postprocessing experiments: the `autoresearch` skill.
- Environment variables affecting operator execution: `cann-pta-envs.md`.
