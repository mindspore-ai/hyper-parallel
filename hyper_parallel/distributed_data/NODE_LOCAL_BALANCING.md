# Node-local balancing

External raw-step loading uses default backbone FLOPs,
capacity-constrained LPT, configurable node-local data exchange, automatic
one-step buffering, and final H2D on a copy stream. Set
`communication_backend="hccl"` (the default) with an NPU `device` to encode
control objects as device tensors and route all data-plane collectives through
HCCL. Set `communication_backend="gloo"` for encoded control/payload exchange on CPU.
Gloo defaults to CPU batches; an explicit NPU/CUDA `device` enables final H2D
without changing the data-plane backend.
Set `balance_group_size` to partition the root mesh into fixed-size independent
balancing groups. For example, `balance_group_size=1024` creates ten groups on
a 10240-rank pure-DP mesh. Leave it unset to retain launcher/node grouping.
This option applies only to external-step loading;
native `batch_sampler` loading keeps its global data plane.
There is no enable switch: every step evaluates a candidate, but raw samples move only when
the candidate's relative objective improvement is strictly greater than
`min_balance_gain` (default `0.0`). Equal, worse or insufficient-gain candidates
retain the original distribution. Metadata gathering and planning still occur
when sample exchange is skipped.

A fresh CPU process owns the source, metadata, planning, sample codecs and
pack/collate. A lightweight thread in the training process executes serialized
HCCL requests on an independent data stream; control and payload groups remain
separate from model groups. Metadata AG uses a continuous receive tensor:
one length D2H and one content D2H, independent of group size. CPU decoding
happens in the producer. Gloo uses the same process protocol on CPU tensors.

The loader automatically prefetches the next step before returning the current
batch, so preparation overlaps forward as well as backward. `prefetch_plan()`
and `prefetch()` remain idempotent compatibility hooks; no scheduling switch is
required. H2D uses a copy stream and a consumer event. Its pinned source remains
alive until the copy completes. Shared buffers grow with the payload and are
reused only after the receiver copies them into independently owned storage.
There is no fixed payload-size limit.

Sources and callbacks must be serializable with `cloudpickle` before workers
start. A source may implement `__iter__` as a generator method, but an already
created generator cannot be moved to the child process. Source code, workers
and callbacks must stay CPU-only; disable source `pin_memory` and let HP pin the
final batch. The process path requires Linux shared memory (`/dev/shm`).

Checkpoint through the loader after training a returned step. If the source
implements `state_dict()` and `load_state_dict()`, HP captures its progress and
retains any prefetched batch at the delivered cursor. Full source state is
transferred only for explicit save/load, never attached to ordinary batches.
Call `close()` on every data rank before destroying its communication groups.

## External-step integration

Pass the existing selected-step source and its data callbacks directly to the
loader builder. Hyper owns balancing, buffering and device handoff; the
application defines metadata extraction and packing.

```python
from contextlib import closing

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
)

config = DistributedDatasetConfig(
    seq_len=seq_len,
    local_batch_size=microbatches_per_step,
    packing_budgets=packing_budgets,
    min_balance_gain=0.0,
    balance_group_size=1024,
)


def pack_samples(samples, seq_len):
    # The model collator accepts payload fields only.
    payloads = [{key: value for key, value in sample.items() if key != "metadata"} for sample in samples]
    return model_collator(payloads)


with closing(build_distributed_dataloader(
    None, mesh, config,
    external_step_source=source,
    metadata_fn=lambda sample: SampleMetadata(len(sample["input_ids"]), features=sample["metadata"]),
    pack_fn=pack_samples,
    collate_fn=list,
    model_config=model_config,
    device=device,
    max_steps=train_steps,
)) as loader:
    for microbatches in loader:
        for batch in microbatches:
            train_microbatch(batch)
```

- `external_step_source` accepts an existing selected-step source.
  It preserves the model's processor, sampler and token-budget pack selector.
  Each source output is `[[sample, ...], ...]`, with one bin per microbatch.
  Use the original loader with final collation disabled to retain its selection
  behavior. A flat map-style dataset alone does not define those step boundaries.
- `metadata_fn` maps each raw sample to `SampleMetadata`. It can read embedded
  feature counters or compute metadata from the sample. Raw payloads must use
  codec-supported values such as tensors, scalars, dictionaries, lists and tuples;
  construct `SampleMetadata` in the callback rather than storing it in the payload.
  Set `metadata_mode=False`
  (the default) for this route, including when samples carry precomputed metadata.
  The source is not consumed during loader construction.
- `pack_fn(samples, seq_len)` constructs one accepted bin. Remove any embedded
  metadata fields here if the model collator does not accept them; avoid mutating
  source samples. `collate_fn` assembles all packed bins into one local step;
  use `list` to yield a list of microbatches.
- Use tensor/scalar trees built from dictionaries, lists and tuples for final
  batches so tensor payloads use shared storage. Accelerator prefetch recursively
  moves tensor leaves to the device, preserving dtype and scalar metadata.
  CPU execution keeps batches on the host.
- Balance logs report per-bin sample counts, token counts and predicted costs,
  plus per-rank send/receive counts and the balancing decision.
  Configure the application's Python logging to include INFO messages.
- Iteration returns ready device microbatches. The loader waits on the copy
  event and records storage on the consumer stream internally. No explicit
  device-consumption hook or application-side H2D is needed.
  `loader.last_host_batch` retains the corresponding CPU microbatches for
  optional host-only metering without D2H. Moving previously CPU-only metering
  to the device outputs may introduce synchronization; use that host view if
  needed. A dataloader cannot remove device-wide waits from an existing trainer.

## Runtime contract

- Each source yield contains `local_batch_size` non-empty raw-sample bins.
  Source sampling, worker count and `prefetch_factor` remain source concerns.
- Omit `cost_model` to construct `DefaultCostModel(model_config)` automatically.
  An explicit callback replaces it. Missing default-model dimensions are an
  error, never a fallback to metadata cost. Supply the actual `mlp_layer_types`.
- Default-model features are per-sample `P`, `D`, and conditional-image token
  runs for blockwise attention. `P + D == pack_tokens`. Packing budgets remain
  hard limits separate from predicted FLOPs.
- The default objective is maximum predicted rank workload. Accept a candidate
  only when `(original_score - candidate_score) / original_score` is strictly
  greater than `min_balance_gain`; a zero original score keeps the original.
  `0.05`, for example, requires more than a 5% reduction. Failed LPT packing
  returns the reference layout. Equal maximum load is not a makespan improvement.
- Only global rank zero logs its node's before/after packs, transfers and cost.
  Per-bin `cost` and rank `cost_before`/`cost_after`, maxima and spread retain the
  `WorkloadCost.llm` component for compatibility (`cost_component=llm`). Decision
  fields `original`, `candidate` and `relative_gain` use the algorithm's actual
  objective, which may depend on different components and need not be additive.
- H2D selects the current NPU/CUDA device automatically; `device` can specify it.
  CPU-only execution keeps host batches. Use compute-stream synchronization,
  not device-wide synchronization, to avoid draining the next step's copy.
- This local-step route requires pure DP, equal step counts and a raw-step source.
  It supports source-backed checkpoint/resume; use native sampler mode for
  shared-metadata reads. Both paths use the same cost/algorithm contracts.

## Independent cost and assignment policies

Both policies are optional builder arguments. Omitting `cost_model` constructs
`DefaultCostModel(model_config)`; omitting `balancing_algorithm` constructs
`LPTBalancingAlgorithm()`. Neither callback needs communication or H2D code.

```python
from hyper_parallel.distributed_data import LPTBalancingAlgorithm

loader = build_distributed_dataloader(
    None, mesh, config, device=device,
    external_step_source=source,
    metadata_fn=lambda sample: SampleMetadata(len(sample["input_ids"]), features=sample["metadata"]),
    pack_fn=pack_samples,
    collate_fn=list,
    cost_model=my_cost_model,
    balancing_algorithm=LPTBalancingAlgorithm(objective="makespan"),
)
```

A user algorithm implements the `BalancingAlgorithm` protocol:

```python
class MyBalancingAlgorithm:
    algorithm_id = "my-placement-v1"
    objective_name = "makespan"

    def assign(self, samples, *, reference_bins, constraints,
               data_parallel_size, local_batch_size):
        # samples contain metadata.cost already computed by the cost model.
        # Return DP-major bins of SampleKey values, not payloads or new costs.
        return my_placement(samples, reference_bins, constraints,
                            data_parallel_size, local_batch_size)

    def objective(self, rank_costs):
        # A finite, nonnegative scalar; smaller is better.
        return max(cost.llm for cost in rank_costs)
```

The result must have `data_parallel_size * local_batch_size` nonempty bins,
contain every input occurrence exactly once, and respect token/stage budgets.
Hyper reconstructs actual rank costs from the scored samples, applies the same
objective to original and candidate layouts, and owns the threshold gate. A
custom algorithm cannot replace costs or force an inferior placement through.
The default LPT assignment and its objective live in `balancing_algorithm.py`;
`planner.py` handles cost evaluation, plan construction and acceptance.

For checkpointable loading, stateful cost/algorithm objects
should provide configuration-versioned `model_id` / `algorithm_id` attributes.
Those identities participate in cross-rank build and checkpoint fingerprints.

## Native sampler integration

Plain Dataset + native BatchSampler retains its selection and
checkpoint mechanism, but now receives the same cost and algorithm parameters.
It also requires `model_config` when no custom cost model is supplied.

A runnable CPU example is
`examples/torch/distributed_data/external_dataset.py`.
