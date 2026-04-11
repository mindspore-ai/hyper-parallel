# PR Review Checklist

This checklist covers areas that CI cannot check. Focus on distributed system correctness, cross-platform consistency, and code quality.

## Distributed System Correctness

### Stream Synchronization

- [ ] **Async collective handles waited** — `handle` from `async_op=True` must be waited via `handle.wait()` before accessing output tensor
- [ ] **non_blocking transfers synced** — `tensor.to(device, non_blocking=True)` destination not read until stream completes
- [ ] **Cross-stream deps use events** — `event.record(stream_A)` → `event.wait(stream_B)`; CPU code order does NOT guarantee GPU order
- [ ] **No stale data from stream mismatch** — Verify tensor is not accessed on wrong stream after async op
- [ ] **Activation swap stream safety** — `launch_offload/launch_load` on `copy_stream`; `wait_offload/wait_load` before compute stream access

### Memory Lifecycle

- [ ] **Storage freed after use** — `tensor.untyped_storage().resize_(0)` called to free device memory (e.g., after all-gather consumed)
- [ ] **Communication buffers cleared** — `clear_reduce_scatter_output()` / `clear_all_reduce_output()` after consuming reduced gradients
- [ ] **Gradient references nulled** — `param.grad = None` and `unsharded_accumulated_grad = None` after gradient consumed
- [ ] **Buffer reuse over reallocation** — `resize_(expected_size)` preferred over allocating new tensors
- [ ] **Pipeline buffers cleaned** — `_clear_recv_buffer()` and `clear_cache()` after each micro-batch
- [ ] **Activation swap lifecycle complete** — `wait_offload()` frees device storage, `wait_load()` frees CPU storage; missing either side causes memory growth
- [ ] **WeakRef for shared resources** — `weakref.WeakSet` or similar for storage references that should auto-release

### DTensor Invariants

- [ ] **`is_partial()` with parentheses** — It is a method, not a property (defined in `layout.py:473`)
- [ ] **`reduce_partial` before redistribute** — Must reduce partial state before redistribution
- [ ] **ReduceScatter before AllReduce** — Ordering rule in `tensor_redistribution.py`
- [ ] **`SkipDTensorDispatch` in grad hooks** — Use when operating on raw local tensors inside gradient hooks
- [ ] **Layout consistency** — `_build_layout` called with correct `device_mesh`, `placements`, `tensor_dim`

### HSDP/FSDP Specifics

- [ ] **Correct scheduler path** — `grad_sync_stream` only in legacy `HSDPSchedulerV2 + comm_async=True`; new `TorchHSDPSchedulerV2` does not use it
- [ ] **All-gather output lifecycle** — `free_unsharded_param()` called after resharding via `resize_(0)`
- [ ] **Init guard for buffers** — `init_all_gather_outputs()` respects `force_recreate=False`

## Cross-Platform Consistency

### Platform Abstraction

- [ ] **Uses `get_platform()`** — Platform-agnostic code never imports torch/mindspore directly
- [ ] **Both backends updated** — Changes in `platform/torch/` have corresponding `platform/mindspore/` changes (or explicit justification)
- [ ] **Base class updated first** — New platform APIs added to `platform/platform.py` abstract class before implementations
- [ ] **Collective ops via platform** — `all_reduce`, `all_gather`, `reduce_scatter` go through `platform.*`, not raw framework calls
- [ ] **No `self.platform`** — Platform is always referenced via module-level `platform = get_platform()`, never stored as an instance attribute
- [ ] **`differentiable_*` in autograd paths** — Code in forward/backward computation (e.g., `TensorRedistribution`, op dispatch) must use `platform.differentiable_all_reduce` / `platform.differentiable_reduce_scatter`, not the non-differentiable variants
- [ ] **`group` vs `group_info` type correct** — `platform.all_reduce/all_gather_into_tensor/reduce_scatter_tensor` expect `group_info` (object with `.group` attr); `platform.differentiable_*` expect raw `group`; `platform.create_group()` returns raw `group` — verify callers wrap/unwrap correctly

### Common Cross-Platform Pitfalls

- torch-specific tensor APIs used in platform-agnostic code
- Device string handling differences between backends
- Gradient computation API differences
- Process group creation/management differences
- Using `self.platform` instead of module-level `platform` (hides bugs, may reference stale or nonexistent attribute)
- Mixing `differentiable_*` and non-differentiable collective APIs in autograd paths (breaks gradient flow)
- Passing raw `ProcessGroup` to APIs that expect `group_info` wrapper, or vice versa (causes `AttributeError` at runtime)

### Multi-Platform & List/Collection APIs

When an API supports multiple backends (e.g. Torch + MindSpore) or list/collection inputs (e.g. `fully_shard([m1, m2])`), verify:

- [ ] **Same semantics on all backends** — Torch and MindSpore paths receive the same logical inputs (e.g. single module vs tuple of modules); compare state/scheduler construction and who gets the handle
- [ ] **List/collection contract clear** — If API accepts a list, document and implement whether every element gets a handle, can be used in follow-up APIs (e.g. prefetch), and participates in state
- [ ] **State/handle covers all managed objects** — When one logical unit spans multiple user-visible objects (e.g. multiple roots), either every object gets the same handle or docs/tests make “only first is handle” explicit
- [ ] **Tests use real user scenarios** — At least one test exercises “non-first” element (e.g. second root `.unshard()`, or second root in prefetch list); avoid mocking away the code path under test
- [ ] **In-place return assertions** — When asserting “return is the same container as input”, use a named variable: `in_list = [a, b]; result = api(in_list); assert result is in_list`; never `assert result is [a, b]`

## Code Quality

### Abstractions and Design

- [ ] **Clear abstractions** — State management is explicit; no dynamic attribute setting/getting
- [ ] **Match existing patterns** — Code follows architectural patterns already in the codebase
- [ ] **No over-engineering** — Only requested changes are made; no speculative features
- [ ] **No premature abstraction** — Helpers and utilities are only created when reused; three similar lines is better than a one-use helper
- [ ] **No trivial helpers** — Avoid 1-2 LOC helper functions used only once (unless significantly improves readability)

### API Design

When a change introduces new API patterns, evaluate broader implications:

- [ ] **No flag-based internal access** — Reject patterns like `_internal=True` kwargs that gate internal functionality; use a separate private function instead
- [ ] **Pattern already exists?** — Search the codebase to check if this pattern is already established; new conventions need stronger justification
- [ ] **BC implications** — Will this pattern create future backward compatibility constraints?
- [ ] **Discoverability** — Is this pattern understandable to users? Will it appear in autocomplete or docs in confusing ways?

### Code Clarity

- [ ] **Self-explanatory code** — Variable and function names convey intent; minimal comments needed
- [ ] **Useful comments only** — Comments explain non-obvious context that cannot be inferred locally
- [ ] **No backward-compatibility hacks** — Unused code is deleted completely, not renamed with underscores or marked with "removed" comments
- [ ] **Appropriate complexity** — Solutions are as simple as possible for the current requirements

### Initialization and Module Design

- [ ] **No fragile init ordering** — If multiple imports/calls must happen in a specific undocumented order, flag the design; dependencies should be explicit
- [ ] **Idempotent global state** — Registries and global lists that accumulate entries must handle multiple calls safely (no duplicate registration)

### Conventions

- [ ] **License header** — All `.py` files start with Apache 2.0 header (lines 1-16)
- [ ] **Line length** — ~120 chars for Python, 120 for C++
- [ ] **Naming** — Classes `PascalCase`, functions/vars `snake_case`, private `_leading_underscore`
- [ ] **Docstrings** — Google-style with `Args:`, `Returns:`, `Raises:`, `Example:` sections on public APIs
- [ ] **Type hints** — Present on all public function signatures
- [ ] **Imports** — **Non-platform code:** module-level imports only; flag imports inside methods unless documented exceptions (`TYPE_CHECKING`, optional dependency, circular import). **`platform/torch/` and `platform/mindspore/`:** lazy framework imports inside methods with `# pylint: disable=C0415` are expected
- [ ] **Pylint compliance** — Run `pylint` on changed `.py` files; add violations to `.jenkins/check/config/filter_pylint.txt` for unified suppression (do not use inline `# pylint: disable=` except `C0415` on lazy backend imports in `platform/torch/` and `platform/mindspore/` per `code-style.md`)

### Common Issues to Flag

- Dynamic `setattr`/`getattr` for state management (prefer explicit class members)
- GPU-CPU synchronization in hot paths (`.item()`, `.numpy()`, `print(tensor)`)
- Magic numbers without explanation
- Unused imports, variables, or dead code paths
- Copy-pasted code that could be shared
- Overly defensive error handling for impossible cases

## Thread Safety & Concurrency

### Python Threading

- [ ] **No unprotected shared mutable state** — Shared data structures accessed from multiple threads are protected by locks or are inherently thread-safe
- [ ] **Lock ordering** — When multiple locks are acquired, ordering is consistent to avoid deadlocks
- [ ] **No GIL-reliant correctness** — Code that mutates shared state should not rely on the GIL for thread safety

### C++ Threading (if applicable)

- [ ] **No data races** — Shared mutable state is protected by mutexes or uses atomics with appropriate memory ordering
- [ ] **RAII lock guards** — Prefer `std::lock_guard` or `std::unique_lock` over manual `lock()`/`unlock()`
- [ ] **No lock-order inversions** — Consistent global ordering when acquiring multiple locks
- [ ] **Correct atomic memory ordering** — `std::memory_order_relaxed` only when ordering with other operations is genuinely unnecessary

### Distributed Concurrency (HyperParallel-specific)

- [ ] **CUDA/NPU stream synchronization** — Operations across different streams require explicit synchronization; missing sync causes silent data corruption
- [ ] **Collective operation ordering** — All ranks must call collectives in the same order to avoid deadlocks
- [ ] **Pipeline stage deadlock prevention** — Send/recv pairs must be correctly matched across stages; mismatched order causes hangs
- [ ] **Gradient hook thread safety** — Per-parameter gradient hooks may fire from autograd worker threads; shared state must be protected
- [ ] **Process group lifecycle** — Groups created during init must remain valid through training; premature destruction causes NCCL/HCCL errors

## Performance

### Obvious Regressions

- [ ] **No unnecessary allocations** — Tensors are not repeatedly created in hot loops; prefer pre-allocation or buffer reuse
- [ ] **Appropriate in-place operations** — Use in-place ops where possible in performance-critical paths
- [ ] **No Python loops over tensors** — Prefer vectorized operations over iterating tensor elements
- [ ] **No GPU-CPU sync in hot paths** — Avoid `.item()`, `.numpy()`, `print(tensor)`, `tensor.tolist()` in training loops

### Device Handling

- [ ] **Device consistency** — Operations don't unexpectedly move tensors between devices
- [ ] **Async where possible** — Use `non_blocking=True` for device transfers, `async_op=True` for collectives, with proper sync points
- [ ] **NPU/GPU compatibility** — Device-specific optimizations are gated by platform checks, not hardcoded

### Memory Patterns

- [ ] **No memory leaks** — Temporary tensors are freed, no circular references, buffers resized to zero after use
- [ ] **Efficient data structures** — Appropriate containers for access patterns
- [ ] **Gradient memory** — Proper use of `no_grad()`, `detach()` to avoid unnecessary graph retention
- [ ] **Communication buffer reuse** — `resize_()` existing storage instead of allocating new tensors

### Common Performance Issues

- Creating new tensors inside training loops instead of pre-allocating
- Synchronous device operations where async would work
- Keeping computation graph alive longer than needed
- Redundant clones or copies
- Calling `torch.cuda.synchronize()` or `torch.npu.synchronize()` unnecessarily (serializes all streams)

## Testing

### Test Existence

- [ ] **Tests exist** — New functionality has corresponding tests
- [ ] **Right test location** — `tests/torch/ut/` for PyTorch unit tests, `tests/mindspore/ut/` for MindSpore unit tests

### Test Patterns

- [ ] **Proper markers** — `@arg_mark(plat_marks=..., level_mark=..., card_mark=..., essential_mark=...)`
- [ ] **Distributed tests use helpers** — `torchrun_case()` for PyTorch, `msrun_case()` for MindSpore
- [ ] **Graceful hardware skip** — Tests skip cleanly when required hardware unavailable
- [ ] **Self-contained** — No shared mutable state between tests

### Test Quality

- [ ] **Edge cases covered** — Empty tensors, single rank, partial state, boundary dimensions
- [ ] **Error conditions tested** — Expected exceptions tested with proper assertions
- [ ] **Both backends tested** — Or explicit justification for single-backend test

### Common Testing Issues

- Tests that only check happy path without error cases
- Distributed tests that hardcode rank count instead of parameterizing
- Missing cleanup of distributed process groups
- Tests that rely on specific GPU count without graceful degradation

## Op Registration

### YAML + Python Consistency

- [ ] **YAML entry exists** — New ops registered in `core/shard/ops/yaml/`
- [ ] **Python impl exists** — Implementation in `core/shard/ops/parallel_*.py`
- [ ] **YAML and Python match** — Op signatures, input/output placements consistent
- [ ] **Redistribution rules correct** — Input/output placement inference handles all cases
