# Node-local balancing

External raw-step loading uses default backbone FLOPs,
capacity-constrained LPT, configurable node-local data exchange, automatic
one-step buffering, and final H2D on a copy stream. Set
`communication_backend="hccl"` (the default) with an NPU `device` to encode
control objects as device tensors and route all data-plane collectives through
HCCL. Set `communication_backend="gloo"` for CPU object/payload exchange.
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
The first step has no prior compute to overlap. Source exhaustion stops
speculative work. To limit iteration, truncate the external source before
wrapping it, for example with `itertools.islice(source, train_steps)`. All ranks
must still consume equal step counts.

Launch these hooks while previously submitted device computation is still
pending. A separate group alone does not create overlap: using the model's
current stream, calling device-wide synchronization, or launching after a
blocking loss read can remove that window. Metadata sizes and the plan still
need to reach the host, and payload encoding/split exchange run on the caller;
the amount hidden depends on the remaining compute and shared bandwidth.

## External-step integration

Pass the existing selected-step source and its data callbacks directly to the
loader builder. Hyper owns balancing, buffering and device handoff; the
application defines metadata extraction and packing.

```python
from contextlib import closing
from itertools import islice

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
    external_step_source=islice(source, train_steps),
    metadata_fn=lambda sample: SampleMetadata(len(sample["input_ids"]), features=sample["metadata"]),
    pack_fn=pack_samples,
    collate_fn=list,
    model_config=model_config,
    device=device,
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
- Accelerator prefetch recursively moves tensor leaves to the training device,
  preserving dtype and non-tensor metadata. Batch objects with a `to` method use
  that method. CPU execution keeps batches on the host.
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
  It does not support loader checkpoint/resume or native shared-metadata reads.
  Use the native sampler entry for checkpointable loading; it uses the same
  cost/algorithm contracts.

## Independent cost and assignment policies

Both policies are optional builder arguments. Omitting `cost_model` constructs
`DefaultCostModel(model_config)`; omitting `balancing_algorithm` constructs
`FeasibleLPTBalancingAlgorithm()`, which scores window placements by scheduled round maxima and
refines feasible pack pairs within a fixed work budget. Neither callback needs communication or H2D code.

```python
from hyper_parallel.distributed_data import FeasibleLPTBalancingAlgorithm

loader = build_distributed_dataloader(
    None, mesh, config, device=device,
    external_step_source=source,
    metadata_fn=lambda sample: SampleMetadata(len(sample["input_ids"]), features=sample["metadata"]),
    pack_fn=pack_samples,
    collate_fn=list,
    cost_model=my_cost_model,
    balancing_algorithm=FeasibleLPTBalancingAlgorithm(objective="makespan"),
)
```

A user algorithm implements the `BalancingAlgorithm` protocol:

```python
class MyBalancingAlgorithm:
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

For checkpointable native sampler loading, callers must keep cost/algorithm
configuration consistent across ranks and compatible with restored state.
Policies do not need identity attributes and are not fingerprinted.

## Native sampler integration

Plain Dataset + native BatchSampler retains its selection and
checkpoint mechanism, but now receives the same cost and algorithm parameters.
It also requires `model_config` when no custom cost model is supplied.

A runnable CPU example is
`examples/torch/distributed_data/external_dataset.py`.
