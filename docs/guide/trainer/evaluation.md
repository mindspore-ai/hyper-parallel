# Standalone loss/PPL evaluation

**English** | [简体中文](evaluation.zh.md)

`examples/training_demo/run_text.py` runs this task with `run_mode: eval`. The default remains `train`.

Supports one device, DP and dense FSDP/HSDP without an optimizer or scheduler.
TP/CP/EP/PP/SP, training compilation, activation checkpointing/swapping, HyperOffload and worker recovery
are currently rejected.

## Launch

Update paths in the [example configuration](../../../examples/training_demo/qwen3/eval.yaml), then run:

```bash
torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/eval.yaml
```

Set `checkpoint.restore_from` to a DCP checkpoint path, or use `LATEST` with `checkpoint.checkpoint_dir`.
Missing checkpoints raise an error. Without a restore request, use the weights loaded by the model builder.
Ascend uses `hccl`; NVIDIA uses `nccl`.

## Evaluation

### Loss and generation evaluation

`evaluation.mode` defaults to `loss`: provide existing text, evaluate the likelihood
of its true labels under causal attention, and aggregate loss and perplexity. The
model runs in `eval()` mode without gradients and calls
`model.forward(..., use_cache=False)` directly. It does not generate an answer
token by token through `infer.generate`.

`mode: generation` is reserved and currently raises `NotImplementedError` at the
run-mode boundary. Generative evaluation would generate an answer from a prompt,
then score it against a reference or judging rule. Its generation stage should reuse
the same `infer.generate` as prediction, with scoring and aggregation added afterward.

Standalone loss evaluation can compare checkpoints, for example weights saved after
training steps 1000, 2000 and 3000, on the same validation data. Hold the data range,
tokenizer, template, truncation, supervision masks, statistics and numerical precision
constant. When using `max_batches`, also fix batching and DP topology so that the
evaluated sample set does not change. Lower loss/PPL means better prediction of the
true tokens on this dataset, not necessarily better performance on every downstream
task. Training log losses can use different data batches, auxiliary terms or reduction
rules and are not a substitute for this independent score.

### Data ranges

`evaluation.dataset` and `evaluation.dataloader` can override the top-level data
configuration. `evaluation.split` accepts `valid`, `test` or `dataset`; the last
option selects a single unsplit Dataset. Tasks never implicitly select training data.

- `max_samples` bounds the selected Dataset's global index range before DP partitioning.
- Online Mapping caps the request at the natural split size; `max_samples: null` traverses that entire split without a training iteration plan. Weighted sources are sampled without replacement: weights determine order while exhausted sources yield to the remaining sources. Changing the cap preserves the selected prefix for a fixed seed.
- Iterable sources require `max_samples`; the finite prefix is materialized before partitioning.
- Indexed builders require an explicit sample plan, supplied through `max_samples` or the builder's explicit split sizes.
- `max_batches` limits forward rounds. Dynamic token batching determines each round's sample count from its token budget.
- Incomplete tails are retained. Valid samples are never duplicated to fill DP lanes. Exhausted lanes execute zero-weight forwards to keep FSDP rounds aligned.

For example:

```yaml
evaluation:
  mode: loss
  split: valid
  batch_size: 2
  max_samples: 1000
  max_batches: 20
  metrics:
    - _target_: hyper_parallel.evaluation.metrics.LossMetric
    - _target_: hyper_parallel.evaluation.metrics.PerplexityMetric
```

This selects up to 1000 source records before DP partitioning and executes at most
20 forward rounds per lane. Evaluation ends on exhaustion or the round limit; both
limits set to `null` traverse the selected finite dataset. `num_samples` counts model
samples with valid supervision, including individual packed sequences. If a transform
splits one source record into multiple windows, each window counts separately, so the
reported count need not equal the source-record count.

### Dynamic batching and padding

The eval example uses the existing
[`TokenBatchLoader`](../../../hyper_parallel/data/batching/build_dataloader.py),
which groups encoded samples by input token count. Its default budget per DP lane is:

```text
token_budget = evaluation.batch_size × data_transform.max_seq_len
```

Set `token_budget` explicitly in the selected dataloader configuration to override it.
With this dynamic loader, `batch_size` controls source reads and derives the default
budget; it is not a cap on the number of samples in the final forward. For example,
`batch_size: 2` and `max_seq_len: 1024` give 2048 tokens, which can accommodate two
1024-token samples or four 512-token samples. This counts input tokens, not the valid
supervision tokens used as the loss denominator.

The loader scans buffered candidates in order. A candidate that fits is selected;
one that does not fit stays for a later round while the scan continues. The loader
does not sort by length or split a model sample to fill the remaining budget.
The preceding data transform determines whether a source record becomes multiple
model windows. With a budget of 2048 and buffered lengths `900, 700, 600, 400`, it selects
`900 + 700 + 400 = 2000`, leaves the 600-token sample for later, and need not fill
the remaining 48 positions. A single sample exceeding the budget is emitted alone,
so the budget is not a hard length limit; model context and collator limits still apply.

`min_buffered_samples` controls the minimum candidate pool, not the forward's sample
count. Normally both the candidate-count and buffered-token thresholds must be reached
before selection. On source exhaustion, the loader drains all remaining candidates.

The accompanying `TextPackingCollator` concatenates selected samples into
`[1, packed_length]` and records their boundaries in `cu_seq_lens`. By default, it
pads only to the required execution alignment, not to the token budget. An explicit
`pad_to_length` pads to that fixed length. Padding labels use `IGNORE_INDEX` and do
not contribute to loss/PPL denominators. Different rounds can have different input
lengths, so `max_batches` cannot be converted to a fixed number of samples.

### Metrics and statistical meaning

The built-in metrics are `LossMetric` and `PerplexityMetric`, with default output names
`loss` and `perplexity`. Configure both in one evaluation to obtain both values. The
report also contains sample/token counts, forward rounds, elapsed time and status.

```text
loss = sum of valid-token losses / number of valid tokens
PPL = exp(global mean NLL)
```

For example, batches with 10 and 90 valid tokens and mean losses of 2 and 1 produce
`(10 × 2 + 90 × 1) / 100 = 1.1`, not the unweighted batch mean of `1.5`.

Statistics accumulate numerators and denominators before reduction, rather than
averaging batch or rank means. Built-in loss/PPL and sample counts share one numerical
reduction after exchanging schemas and error status. Repeated `compute()` calls do
not reduce again. CPU/CUDA accumulation uses FP64. NPU accumulation uses compensated
FP32 sums to avoid precision loss in torch_npu's FP64 cast fallback. Sample counts remain
integer-valued on device. Communication uses FP32 high/low pairs for HCCL compatibility;
the final components are combined in CPU FP64 after reduction. This keeps memory bounded
without synchronizing each batch to the host. Compensation limits local summation error;
the collective still sums each channel in FP32 and does not provide exact FP64 arithmetic.

Loss modules declare `loss_normalization` as `token_mean`, `sum` or `global_mean`.
Already-reduced `global_mean` values are not accumulated again.
`LossStatistics.is_lm_nll` is a boolean: only a pure language-model NLL should declare
`True` so that PPL may exponentiate it. Other objectives, including a loss with auxiliary
terms mixed in, keep `False` and can still be reported as loss. A loss module may declare
`statistic_is_lm_nll` as a mapping or callable of the loss key. The chunked causal-LM
loss reports one local token mean; when the model folds a router auxiliary term with
a nonzero coefficient into it, it declares `foundation` as `False`, so PPL is refused.
An absent or zero auxiliary coefficient preserves the pure NLL and its perplexity.
Fully masked batches contribute zero weight and do not propagate empty-cross-entropy
NaNs to the report.

### Custom metrics

`evaluation.metrics` accepts importable
[`Metric`](../../../hyper_parallel/evaluation/metrics/base.py) class targets, for example
`_target_: my_project.metrics.CustomMetric`. The lifecycle is:

```text
reset() → update(statistics) for each round → synchronize(context, device) → compute()
```

`update()` receives named `LossStatistics` with a numerator, denominator and
`is_lm_nll`, not logits, labels or generated text. `compute()` returns a flat
`{"metric_name": value}` mapping and reads results without another collective.
Metric output names must be distinct.

For a transformation of an existing loss, subclass `LossMetric` and override only
`compute()`. For example, dividing pure NLL by `ln(2)` gives bits per token and still
requires checking the statistic's `is_lm_nll`. Subclasses share the built-in reduction
while they inherit both `update` and `synchronize` unchanged. Overriding either method
keeps their own accumulation and synchronization. More general metrics can implement
all four `Metric` lifecycle methods. A custom metric may explicitly return `True` from
`supports_shared_reduction` and implement `set_reduced_statistics` when the raw shared
statistics alone are sufficient.

Accuracy, F1, BLEU and ROUGE are not built in and cannot be enabled just by adding a
metric target. Token accuracy needs a step that also produces correct-token and
valid-token counts; text metrics need generation and scoring. The underlying
`Evaluator` accepts a custom step, but `TextTrainer.evaluate()` currently constructs
`LossEvalStep` with no YAML setting to replace it. Such extensions need both the
step's data production and a corresponding Trainer-side integration.

## Boundaries

`evaluation/` does not import Trainer. Trainer constructs `LossEvalStep` and passes it to `Evaluator`.
The evaluator calls `step(round_index)` to receive statistics. The Trainer-side step reads batches and calls
the model and loss function, while the evaluator owns iteration and aggregation. This runtime callback
does not require a dependency on a concrete Trainer. Loss evaluation does not call `infer.generate`.
`get_build_steps(run_mode)` returns enabled component names; it neither builds them nor defines execution order.

## Validation entry points

```bash
pytest -q tests/ut/evaluation
pytest -q tests/torch/evaluation/test_eval_e2e.py
```

System tests use real two-layer randomly initialized FP32 models and synthetic data. They do not establish
coverage for full-size models, BF16, quantization or replacement-operator combinations. System tests require
the matching device/backend; CPU tests cannot replace NPU acceptance.
