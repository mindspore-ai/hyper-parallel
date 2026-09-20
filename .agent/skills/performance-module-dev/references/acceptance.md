# High-Performance Module Acceptance

## Accuracy Gate

- Standalone reference vs optimized forward, backward, parameter gradients, and one optimizer update.
- Edge cases for empty/zero-token routes, uneven shapes, padding/masks, noncontiguous inputs, and unsupported configs.
- Reference module vs optimized module crossed with target parallelism using the `accuracy-validation` 2x2 matrix.
- Standard tolerance by default. Relaxed tolerance is allowed only for declared fused/grouped/low-precision accumulation and
  still requires benchmark quality evidence when user-facing model behavior can change.

## Performance Method

- Freeze model, batch, sequence/packing, topology, dtype, compilation, checkpointing, and kernels unrelated to the change.
- Measure time-to-first-step separately from steady state.
- Warm up until compilation and allocator transients settle, then record at least five independent measurement windows.
- Report median and P95 step time, tokens/s or samples/s, allocated/reserved peak memory, and MFU when the FLOPs contract is
  valid. Record per-rank maximums where one slow or memory-heavy rank limits the job.
- Synchronize only at measurement boundaries. Do not add `.item()` or device synchronization to every hot-path step.
- A microbenchmark supports diagnosis; default policy requires end-to-end evidence on a representative workload.

## Promotion Defaults

| Policy | Minimum evidence |
| --- | --- |
| Experimental/opt-in | Standalone accuracy, smoke, clear dependency/error behavior |
| Supported opt-in | Accuracy plus parallel composition for declared matrix and reproducible end-to-end impact |
| Auto-selected | Supported opt-in plus capability detection and tested fallback on unsupported shapes/hardware |
| Default-on | Auto-selected plus at least 5% median throughput gain or 5% peak-memory reduction on the primary workload, no more than 2% regression on declared secondary workloads, and acceptable startup cost |

Different product targets may predeclare another threshold. Do not lower it after results are known. Memory-only modules may
accept neutral throughput when the memory reduction enables a larger batch/sequence that is demonstrated end to end.

## Fallback And Observability

- Log the selected implementation once with the capability reason; do not log every step.
- Missing optional dependencies produce an actionable message or a documented fallback.
- A fallback must preserve math and state-dict behavior and receive its own smoke test.
- Expose enough runtime evidence to distinguish reference, optimized, and fallback paths in validation artifacts.
- Do not describe a silently disabled optimization as enabled or supported.
