# Push-compatible MegaMoe memory optimizations

Date: 2026-09-17. Branch: `feat/megamoe-push-memory`.
Base: freshly fetched `upstream/master`, `5642a098`.

## Selected changes

| Optimization from the pull experiments | Applies to push? | This branch |
| --- | --- | --- |
| Backward receive storage reused for expert input gradients | Yes; depends on the local reader/event order | Ported with a plan-time safety check and native alias declaration |
| Release backward scratch before token-gradient restoration | Yes | Ported for `act_grad`, `swiglu_grad`, and `gate_dx` |
| Round symmetric heap capacity to 2 MiB physical pages | Yes | Ported while preserving push receive-capacity sizing |
| Convert FP32 parameters directly into their existing BF16/FP16 storage | Yes; shared optimizer also benefits native experts | Ported using `destination.copy_(source)` |
| Allocate intermediate activations to the local received capacity; avoid redundant zeroing | Yes | Already present upstream |
| Reuse forward `down_proj` as the owned saved dispatch copy | Yes | Already present upstream; the buffer must remain saved for backward |
| Small legacy workspace placeholders and shared execution resources | Yes | Already present upstream |
| Pull source heap and GET transport | No | Excluded |
| Unpermute-grad directly producing the pull symmetric source | No equivalent staging copy in push | Excluded; no new unpermute operator or autograd boundary |
| Early clearing of saved tensors for the integrated unpermute boundary | Needed by that pull implementation | Excluded |

The PUT transport, task ordering, communication/DMA granularities, profiler,
native MegaMoe argument lists, and forward permutation APIs remain those of upstream.
The merged permutation-gradient fix adds a metadata-only ACLNN adapter for managed input gradients.
The native schema now permits `dispatch_target` and `gate_dx` to alias; rebuilding the Torch adapter is required.
The loader rejects an older schema before launching the shared-storage path.

## Why backward reuse also works for push

Push completes the remote writes before publishing the dispatch completion signal.
After reception, the same expert rows are read by W2-gradient and activation-gradient GMM tasks.
On every Cube worker, W2-gradient precedes activation-gradient; the activation completion event joins all workers,
then SwiGLU-gradient joins all expert slices before any gate-gradient task writes dX.
The device worker publishes completion after its compute call has finished.

[storage.py](../modules/mega_moe/backward/storage.py) checks this ordering in the final physical queues.
Additional receive readers, mixed workers, changed event thresholds, or changed ordering disable reuse.
The check runs when constructing a plan and adds no runtime barriers or task reordering.

Push uses the prefix `expert_buffer[:local_capacity]` for dX, retaining the full symmetric allocation for remote PUT.
The saved forward input is an independent owned copy, so delayed and repeated backward cannot overwrite it.
Releasing a temporary view never frees the workspace's symmetric allocation.

This removes one ordinary BF16 `[R, H]` allocation per backward, where R is the local aligned receive capacity.
For H5120, T4096, TopK8, this is roughly 320 MiB per unit of receive load.
It is a local allocation reduction, not a prediction of full-model peak savings.

## Heap sizing retains push capacity

Only the final rounding changes from 64 MiB to 2 MiB. The receive buffer still reserves either
the configured capacity factor or the default lossless EP maximum, plus the original return and event buffers.

For one resource group, H5120 / T4096 / TopK8 / E48 / BF16:

| Receive capacity | Old configured heap | New configured heap |
| --- | ---: | ---: |
| EP4, default lossless | 1664 MiB | 1602 MiB |
| EP8, default lossless | 2944 MiB | 2882 MiB |
| Explicit factor 1.0 | 704 MiB | 642 MiB |
| Explicit factor 1.5 | 832 MiB | 802 MiB |

Explicit heap values must cover all live resource groups and be multiples of 2 MiB.
The SHMEM internal extra allocation remains separate; a configured heap is not the complete process HBM footprint.

## Initial port validation

- CPU multicore and mixed-precision optimizer regression: 121 tests and 460 subtests passed.
  The schedule matrix covers all EP4/EP8 ranks, 20/24 Cube workers, and equal dispatch/combine splits
  of 128, 512, and 1024. Upstream's PUT scheduler does not support the pull branch's asymmetric split policy.
- Native component rebuilt from this branch for `ascend910b`, with Release optimization.
- A two-rank device test on physical devices 4–5 passed with a 2 MiB configured heap.
  It covers independent/reused dX, odd tails, empty receiving ranks, two outstanding forwards,
  reverse-order backward, `retain_graph=True`, and repeated backward.
  Actual native pointers are checked; each rank makes 32 real backward calls and compares output,
  Router gradients, expert gradients, and applicable hidden gradients against native experts.
- CPU lifecycle tests mock the backend gradient operation and verify that scratch references are gone
  before restoration. They do not establish availability of the backend API.
- After moving the parameter-copy test to a separate file, nine focused tests and 22 subtests passed.
- The repository pre-commit Pylint, complexity (CCN <= 15, NLOC <= 100), and Markdown checks passed.
  The broader AutoGit file checks still report upstream formatting/docstring/marker issues and flag the valid
  product name CANN as a spelling error. The original registration file also fails clang-format and the original
  Python files reproduce the spelling findings; unrelated formatting and lint-policy changes are not included.

## ACLNN permutation-gradient merge

Merged `origin/fix/megamoe-permute-grad-aclnn` at `947b4ed5` into this branch.
The adapter calls the existing `aclnnMoeTokenPermuteGrad` using the gradient, mapping, and token/Top-K metadata.
It returns an owned `[T, H]` tensor without saving the original hidden values.
This removes the dependency on the unavailable Torch-NPU `npu_moe_token_permute_grad_v2` Python API.
The receive-buffer alias declaration, safety proof, early scratch release, and heap changes remain intact.
The unit-test conflict was resolved by retaining the scratch-lifetime assertions and mocking the new adapter.

The push-memory device matrix now covers trainable managed inputs in addition to frozen managed inputs
and the public expert-major entry point, with 48 real backward calls per rank.
All three entry modes run with both independent and reused gradient storage.

The Release rebuild and CPU regression passed: 121 tests and 459 subtests.
The adapter's real-device test compares 18 cases per rank against the existing permutation backward
with exact equality across BF16, FP16, FP32, odd sizes, noncontiguous gradients, and a separate stream.
Empty-input and Meta dispatch checks also pass.
All seven two-rank system tests passed on physical devices 4–5 (224.75 seconds): adapter precision,
layer precision, push-memory reuse, local-capacity lifetime, poisoned buffers, device-ready/checkpoint lifecycle,
and shared resources across streams. The receive-reuse case passed with the new managed dX path enabled.
The device audit found no foreign work on the reserved devices.
Repository Pylint, complexity, Markdown, and clang-format checks for the new adapter passed.
The imported C++ adapter received whitespace-only formatting after these system tests and was rebuilt.

Merge validation evidence is in `build/validation/push_memory_aclnn_merge_20260917/`.

No new full-model push/native/pull timing comparison was performed for this port.
The previous pull experiment's allocated/reserved peaks and speedups must not be presented as push measurements.

Local evidence: `build/validation/push_memory_port_20260917/` (ignored build artifacts):
`unit.log`, `unit_final.log`, `build.log`, `st.log`, `precommit_checks.log`,
`autogit_check.log`, `baseline_lint_comparison.json`, and `device/test_mega_moe_push_memory_reuse.json`.
