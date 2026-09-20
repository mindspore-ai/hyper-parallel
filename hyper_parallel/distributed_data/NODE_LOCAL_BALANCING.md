# Node-local balancing

Dataset-owned or external raw-step loading uses default backbone FLOPs,
capacity-constrained LPT, configurable node-local data exchange, automatic
one-step buffering, and final H2D on a copy stream. Set
`communication_backend="hccl"` (the default) with an NPU `device` to encode
control objects as device tensors and route all data-plane collectives through
HCCL. Set `communication_backend="gloo"` for CPU object/payload exchange.
Gloo defaults to CPU batches; an explicit NPU/CUDA `device` enables final H2D
without changing the data-plane backend.
There is no enable switch: every step evaluates a candidate, but raw samples move only when
the candidate's relative objective improvement is strictly greater than
`min_balance_gain` (default `0.0`). Equal, worse or insufficient-gain candidates
retain the original distribution. Metadata gathering and planning still occur
when sample exchange is skipped.

With HCCL, source loading starts one step ahead. Metadata and payload
collectives are issued by the training thread on an independent data stream;
control and payload groups are separate from model groups. CPU planning,
decoding and collation run in the producer. H2D uses a copy stream, and the
consumer waits on its ready event without a host event synchronization.
Pinned copy sources are retained until completion, including early close.
Gloo keeps its complete background producer.

To overlap the full pipeline with model computation, call the two optional
hooks on **every rank at the same training boundaries**:

```python
for microbatches in loader:
    for index, batch in enumerate(microbatches):
        loss = forward(batch)
        if index == 0:
            loader.prefetch_plan()  # HCCL metadata, then background CPU planner
        loss.backward()
        if index == 0:
            loader.prefetch()       # HCCL plan/payload, then decode/collate/H2D
        log_loss(loss.item())       # Host synchronization belongs after launch
```

`prefetch()` completes `prefetch_plan()` if needed; repeated calls before
consumption do not advance the source twice. Ordinary `next(loader)` also
completes omitted hooks, preserving the iteration contract. Without early
hooks, accelerator data exchange starts only when the batch is requested.
The first step has no prior compute to overlap. End-of-source and `max_steps`
stop speculative work. All ranks must still consume equal step counts.

Launch these hooks while previously submitted device computation is still
pending. A separate group alone does not create overlap: using the model's
current stream, calling device-wide synchronization, or launching after a
blocking loss read can remove that window. Metadata sizes and the plan still
need to reach the host, and payload encoding/split exchange run on the caller;
the amount hidden depends on the remaining compute and shared bandwidth.

## Dataset-based integration

Construct the config, bind the source's existing data contract once, and build
the loader. The application does not implement a balancing wrapper, a pack
adapter, logging, or a device-prefetch consumption hook.

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    build_distributed_dataset,
    build_distributed_dataloader,
)

config = DistributedDatasetConfig(
    seq_len=seq_len,
    local_batch_size=microbatches_per_step,
    packing_budgets=packing_budgets,
    min_balance_gain=0.0,
)
dataset = build_distributed_dataset(
    source,
    metadata="metadata",       # SampleMetadata already produced by the transform
    collate_fn=model_collator,  # Existing collator, called once per accepted bin
    cpu_fields=("cu_seqlens",),
    log_fields=("P", "D"),     # Optional additive metadata.features fields
)
with build_distributed_dataloader(
    dataset, mesh, config,
    model_config=model_config,
    device=device,
    max_steps=train_steps,
) as loader:
    for microbatches in loader:
        for batch in microbatches:
            train_microbatch(batch)
```

- `DistributedDataset` wraps a selected-step source, not a new file reader.
  It does not replace a model's processor, sampler or token-budget pack selector.
  Each source output is `[[sample, ...], ...]`, with one bin per microbatch.
  Use the original loader with final collation disabled to retain its selection
  behavior. A flat map-style dataset alone does not define those step boundaries.
- `metadata` accepts either an existing `sample -> SampleMetadata` callable,
  or a mapping field name containing precomputed `SampleMetadata`. A named
  metadata field is omitted from the dictionaries passed to the collator.
  The source is not consumed during dataset or loader construction.
- `cpu_fields` names top-level fields of the collated mapping. Their complete
  subtrees stay on CPU; other tensor leaves move recursively. Metadata and
  collation remain application semantics, not model names embedded in Hyper.
- `log_fields` selects numeric `SampleMetadata.features` to sum per bin.
  Generic sample/sequence/cost and send/receive logs need no extra callback.
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
  It does not support loader checkpoint/resume or native shared-metadata reads.
  The dataset facade does not add checkpoint/resume. Use the reader/sampler
  entry for checkpointable loading; it uses the same cost/algorithm contracts.

## Independent cost and assignment policies

Both policies are optional builder arguments. Omitting `cost_model` constructs
`DefaultCostModel(model_config)`; omitting `balancing_algorithm` constructs
`LPTBalancingAlgorithm()`. Neither callback needs communication or H2D code.

```python
from hyper_parallel.distributed_data import LPTBalancingAlgorithm

loader = build_distributed_dataloader(
    dataset, mesh, config, device=device,
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

For checkpointable reader/sampler loading, stateful cost/algorithm objects
should provide configuration-versioned `model_id` / `algorithm_id` attributes.
Those identities participate in cross-rank build and checkpoint fingerprints.

## Existing integrations

The original `external_step_source`, `metadata_fn`, `pack_fn`, `move_fn` and
`bin_stats_fn` arguments remain supported for existing users. That entry also
returns device-ready batches and retains CPU views in `last_host_batch`.
Do not mix those arguments with a `DistributedDataset`: the dataset owns all
data callbacks. Plain Dataset + native BatchSampler retains its selection and
checkpoint mechanism, but now receives the same cost and algorithm parameters.
It also requires `model_config` when no custom cost model is supplied.

A runnable CPU example is
`examples/torch/distributed_data/external_dataset.py`.
