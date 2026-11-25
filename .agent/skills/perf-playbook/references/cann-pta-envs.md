# CANN and PTA environment variables

> **Evidence provenance:** Mixed. Registration conventions and the framework rule
> separating infrastructure settings from feature configuration, with pitfalls
> observed in this repository. See [evidence provenance](../SKILL.md#evidence-provenance).

These variables affect performance and debugging. **Use environment variables for
infrastructure and diagnostics.** Configure component behavior through recipe fields
rather than introducing environment variables as feature switches.

> This is a methodological reference that grows with verified usage. Record each
> variable's effect, values, applicability, and pitfalls. Do not set variables with
> uncertain effects in production recipes.

## Registration fields

| Field | Description |
|---|---|
| Variable | Environment-variable name |
| Effect | Execution behavior it changes |
| Values | Typical or recommended values |
| Applicability | When to set it and when not to |
| Pitfalls | Known side effects or consequences of misuse |

## Observed pitfalls

- **Long first-run AOT compilation.** A new operator shape can take tens of minutes
  to compile. The default execution timeout can terminate the run and resemble an
  operator failure. Increase the first-run timeout when appropriate, for example
  `HCCL_EXEC_TIMEOUT=7200`. Before treating a quiet process as a deadlock, inspect
  kernel-cache timestamps and process CPU activity to determine whether it is compiling.
- **Concurrent compilation of one shared library.** Multiple ranks compiling the
  same `.so` on shared storage can fail with
  `ld: final close failed: Input/output error`. Compile it once serially first.
- **Diagnostic-log visibility.** `logger.warning` may be hidden by the training
  configuration. Use a directly visible diagnostic marker to confirm path engagement;
  absence of warnings does not establish that a path ran.
- **TorchNPU on CPU test hosts.** A host can have `torch_npu` installed without an NPU.
  Non-reentrant checkpoint recomputation during `backward` may trigger lazy NPU autocast
  initialization and fail with `aclInit` 507008. CPU tests must skip device-specific
  recomputation or stub the device probe.

## Measurement discipline

- Compare A and B under the **same environment-variable set**. Changing a variable
  changes the experimental conditions and requires remeasurement.
- Record every performance-related variable and value in
  `templates/tuning-summary.md`.

## Related references

- Techniques by bottleneck: `feature-scenarios.md`.
- Fused-operator integration constraints: `fused-ops.md`.
