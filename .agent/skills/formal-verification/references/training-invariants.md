# Distributed training contracts

Use these questions while extracting specifications for a **training framework**.
Derive each precondition, postcondition and invariant from public API documentation,
callers and tests, with file/line evidence. An inferred contract must be labelled as such.
Do not infer intended behavior solely from the implementation being verified.

Express a function contract as `{P} f {Q}` with state ownership, exceptional outcomes,
and relied-on callee contracts. For asynchronous code, include the happens-before graph;
ordinary Python statement order alone is not a device-stream ordering proof.
Distinguish assumptions, LLM reasoning, bounded checks, executed regressions and any
actual solver proof. State the modeled domain and bounds when using a solver.

## Shard geometry and layout

- For valid rank, shape and mesh inputs, shard ranges stay within the global tensor,
  collectively cover it, and have no unexpected overlaps or gaps. Include empty tensors,
  uneven dimensions, more ranks than elements, non-contiguous tensors and negative dimensions.
- Balanced chunking and ceil chunking are different contracts: determine which the API uses.
  Do not impose balanced geometry on FSDP padded/ceil chunks.
- Mesh shape, rank identity, dimension mapping and placement must agree. Round trips should
  preserve the documented layout semantics, including ragged, strided and partial placements.
- Redistribution preserves the reconstructed tensor and its gradient semantics. Make the
  pending reduction explicit when changing partial placement into replication or sharding.

## Forward, backward and optimizer semantics

- Establish a small unpartitioned PyTorch reference under the same dtype, seed, initial
  parameters, loss reduction and optimizer configuration. Compare forward outputs,
  loss, parameter gradients and one optimizer update with justified numeric tolerances.
- Check gradient scaling across data-parallel averaging, loss scaling, HSDP replication,
  microbatch accumulation and `no_sync`. Exactly where averaging occurs is API-specific.
  A factor-of-world-size discrepancy is a candidate, not evidence until reproduced.
- Include shared parameters, unused parameters, repeated forward calls, gradient accumulation,
  mixed precision, and exceptions during backward when the contract supports them.
- Every collective in an autograd path needs the intended backward semantics. Identify
  differentiable wrappers versus intentionally eager communication of parameter/buffer state.

## Parameter and activation state

- Build a transition table for sharded/unsharded/offloaded states using actual enum names
  and hook entry points. Computation must see materialized parameters for the required phase.
- A tensor view must not outlive the backing storage needed for it. Alias ownership and
  reuse must remain valid through resharding, buffer clearing, backward and the optimizer.
- Offload/load completion precedes readers or storage reclamation. Include cross-stream
  dependencies, repeated cycles, exceptions and interrupted operations.
- Activation recomputation must preserve required RNG, autocast and gradient behavior
  relative to the documented uncheckpointed computation.

## Pipeline schedules and communication

- Model events by `(stage, microbatch, direction)` and state the schedule's legal bounds.
  Each supported microbatch must get the required forward/backward work exactly once.
- Forward predecessors precede consumers; backward needs its forward state and downstream
  gradient. Pair sends and receives by peer, microbatch, shape, dtype and ordering.
- Verify schedule-specific rules for warmup, steady state and cooldown, including one stage,
  few microbatches, virtual stages, uneven stage partitions and activation swapping.
- Deadlock candidates need a concrete cycle with rank/stream/event evidence. A reproduction
  must use a bounded timeout rather than leaving hung worker processes.

## Checkpoint consistency

- Save/load round trips preserve the documented parameter and optimizer state. Record dtype,
  global shape, placement, shard offsets and any permitted change in topology.
- Shard metadata covers the logical tensor consistently. Empty/ragged shards, shared weights
  and interrupted writes need explicitly supported or explicitly rejected behavior.

## Evidence boundary

Out-of-contract inputs, undocumented assumptions, unavailable hardware, parser omissions
and unsupported native callees are limitations, not automatically framework bugs. Identify
the missing contract or test needed to resolve them. Finite testing cannot establish
unbounded equivalence or liveness; LLM `MATCH` must never be described as a formal proof.
