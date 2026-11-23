# Standalone evaluation and prediction

**English** | [简体中文](eval_predict.zh.md)

`examples/training_demo/run_text.py` assembles and executes the task selected by
`run_mode: train | eval | predict`. The default remains `train`, preserving existing
training configurations and callback execution.

## Scope

| Mode | Features | Supported topology |
| --- | --- | --- |
| eval (P0) | Global token-weighted loss, NLL perplexity, independent data ranges, model-only restore | Single device, DP, dense FSDP/HSDP |
| predict (P1) | File input, conversation history, length buckets, batched generation, ordered JSONL, manifest | Single device, independent DP lanes holding complete models |

Model adapters explicitly declare `ModelAdapterSpec.supports_generation`. Native
Qwen3 and Qwen3-MoE text models currently declare this capability. Each additional
model family must validate its cache and generation contracts. Model replacements
and quantization combinations require separate acceptance; training support alone
does not establish generation support.

Standalone tasks reject training compilation, activation checkpointing and activation
swap. TP/CP/EP/PP/SP remain unsupported. Prediction also rejects FSDP/HSDP sharding
and omits training FSDP wrapping. Generative evaluation, beam search, multiple
candidates, resumable runs and online serving are outside this phase.

## Launch

Replace the model, input and output paths in the examples, then run:

```bash
# Use nproc_per_node=1 for one device; use training.backend=nccl for NVIDIA.
torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/eval.yaml

torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/predict.yaml
```

Neither eval nor predict requires an optimizer. Predict also needs no dataset,
dataloader or loss function. Set `checkpoint.restore_from` to a DCP checkpoint
path, or use `LATEST` with `checkpoint.checkpoint_dir`. Missing checkpoints raise
an error rather than silently using initial weights. Without a restore request,
the task uses the pretrained weights loaded by the model builder.

## Evaluation

`evaluation.dataset` and `evaluation.dataloader` can override the top-level data
configuration. `evaluation.split` accepts `valid`, `test` or `dataset`; the last
option selects a single unsplit Dataset. Tasks never implicitly select training data.

- `max_samples` bounds the selected Dataset's global index range before DP partitioning.
- Online Mapping caps the request at the natural split size; `max_samples: null` traverses that entire split without a training iteration plan. Weighted sources are sampled without replacement: weights determine order while exhausted sources yield to the remaining sources. Changing the cap preserves the selected prefix for a fixed seed.
- Iterable sources require `max_samples`; the finite prefix is materialized before partitioning.
- Indexed builders require an explicit sample plan, supplied through `max_samples` or the builder's explicit split sizes.
- `max_batches` limits forward rounds. Dynamic token batching determines each round's sample count from its token budget.
- Incomplete tails are retained. Valid samples are never duplicated to fill DP lanes. Exhausted lanes execute zero-weight forwards to keep FSDP rounds aligned.

Statistics accumulate numerators and denominators before reduction, rather than
averaging batch or rank means. Built-in loss/PPL and sample counts share one numerical
reduction after exchanging schemas and error status. Repeated `compute()` calls do
not reduce again. CPU/CUDA accumulation uses FP64. NPU accumulation uses compensated
FP32 sums to avoid precision loss in torch_npu's FP64 cast fallback. Sample counts remain
integer-valued on device. Communication uses FP32 high/low pairs for HCCL compatibility;
the final components are combined in CPU FP64 after reduction. This keeps memory bounded
without synchronizing each batch to the host. Compensation limits local summation error;
the collective still sums each channel in FP32 and does not provide exact FP64 arithmetic.

Custom metrics default to their own `update`/`synchronize` lifecycle. `LossMetric`
subclasses share the reduction while they inherit both methods unchanged. Overriding
either method keeps the subclass's own accumulated statistics and synchronization.
A custom metric may explicitly return `True` from `supports_shared_reduction` and
implement `set_reduced_statistics` when the raw shared statistics alone are sufficient.

`num_samples` counts model samples with valid supervision, including individual packed
sequences. When a transform splits one source record into multiple windows, each window
counts as a model sample; `max_samples` still bounds the original Dataset indices.

Loss modules declare `loss_normalization` as `token_mean`, `sum` or `global_mean`.
Already-reduced `global_mean` values are not accumulated again. PPL requires an
`lm_nll` statistic; auxiliary objectives cannot serve as PPL. A loss module may also
declare `statistic_kinds` (a mapping or a callable of the loss key) to reclassify a
statistic: the chunked causal-LM loss reports one local token mean and, when the
model folds a router auxiliary term with a nonzero coefficient into it, declares `foundation` as an objective,
so perplexity is refused instead of exponentiating an aux-contaminated value.
An absent or zero auxiliary coefficient preserves the pure NLL and its perplexity.
Fully masked batches contribute zero weight and do not propagate empty-cross-entropy
NaNs to the report.

## Prediction input and output

Inputs support `.txt` (one nonempty prompt per line), `.jsonl` (one object per line)
and `.json` (an array of objects). Example JSONL:

```jsonl
{"id":"plain","prompt":"Introduce Beijing."}
{"id":"chat","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"How big is Beijing?"},{"role":"assistant","content":"About 16,400 square kilometres."},{"role":"user","content":"What about its population?"}]}
```

Supply exactly one of `prompt` or `messages`. Missing IDs use stable input ordinals.
Duplicate IDs, empty input and malformed messages fail before generation. Messages
support system/user/assistant/tool roles and preserve template fields such as
`tool_calls` and `tool_call_id`. The task generates only the next assistant response.
Explicitly configure `model_assets.chat_template`; template settings belong to
`model_assets`, not the tokenizer target. Plain prompts are encoded without a chat template.

`batch_size` limits requests per DP lane. `max_batch_tokens` bounds the padded prompt
budget: request count multiplied by the longest prompt length. Oversized requests fail
without silent truncation. `bucket_by_length` can change execution order, while final
output always follows input order.

Generation configuration accepts only serializable `GenerationOptions` fields:
`max_new_tokens`, `temperature`, `top_k`, `top_p`, `do_sample`, `eos_token_id`,
`pad_token_id`, `repetition_penalty` and `use_cache`. Special tokens resolve from
explicit task options, model generation metadata, then the tokenizer. An unspecified
pad token may fall back to EOS. Multiple EOS IDs are supported. Runtime adapters,
devices, process groups and RNGs are injected by code and cannot be passed through YAML.

Each output line contains:

```json
{"id":"plain","ordinal":0,"sample_idx":0,"generated_text":"Beijing is the capital of China.","prompt_tokens":4,"generated_tokens":8,"finish_reason":"eos"}
```

`generated_tokens` includes a terminating EOS and excludes padding. Text decoding
removes special tokens. `finish_reason` comes from the generation loop and is `eos`,
`length` or `stop`; token limits produce `length`. Code-injected legacy stopping
criteria produce `stop`. This phase emits only `sample_idx=0` per request.

DP requires shared storage. Each lane writes a shard and completion manifest. The
coordinator verifies checksums, ownership, IDs, ordinals and exact input coverage,
then merges in input order and atomically publishes JSONL. Existing output raises an
error unless `overwrite: true`. Failures retain `.<output_name>.<run_id>/` without
automatic resume. An adjacent `.lock` prevents concurrent writers; remove a stale lock
only after confirming the previous process has stopped.

`<output_file>.manifest.json` records input/output checksums, model and checkpoint
sources, tokenizer vocabulary fingerprint, template, resolved generation options,
seed, batching policy and environment versions. JSONL is the atomic commit unit;
the manifest is a separate sidecar whose `output_sha256` verifies consistency.
Sampling is reproducible with the same configuration and topology, but changing
DP size or batch composition does not preserve per-request random streams.

## Boundaries and compatibility

`evaluation/` and `predict/` import neither `trainer/` nor each other. The task host
owns the model, tokenizer, restore, configuration and topology validation. Evaluation
receives statistics through an injected step. Predictor owns requests, batches,
results and persistence. Prefill/decode, caching, sampling, stopping and padding
removal use the single generation loop in `infer/`.

Legacy `generate(model, input_ids, generation_config, attention_mask)` calls continue
to return a Tensor by default. New callers can supply `options=GenerationOptions(...)`,
`runtime=GenerationRuntime(...)` and `return_dict=True` to receive `GenerationResult`
with actual prompt/generated lengths and finish reasons.

## Validation entry points

```bash
pytest -q tests/ut/evaluation tests/ut/infer tests/ut/predict
pytest -q tests/torch/predict/test_predict_e2e.py
pytest -q tests/torch/evaluation/test_eval_e2e.py::test_native_eval_range_and_empty_lanes
# These two cases require available NPU/GPU hardware for the FSDP path:
pytest -q tests/torch/evaluation/test_eval_e2e.py::test_eval_matches_reference_single_card
pytest -q tests/torch/evaluation/test_eval_e2e.py::test_eval_dp_ranks_report_identical_global_metrics
```

System tests use real two-layer randomly initialized Qwen3/Qwen3-MoE models, synthetic
data and real Torch backends. Coverage includes HF greedy token parity, cache on/off,
nondivisible DP inputs, empty lanes, failure propagation, the actual TextTrainer,
LATEST model-only restore and an independent evaluation reference. Small-model
acceptance does not replace full-size, replacement-operator or accelerator validation.
