# Offline text prediction

**English** | [简体中文](predict.zh.md)

`examples/training_demo/run_text.py` runs this task with `run_mode: predict`. The default remains `train`.

Supports one device and independent DP lanes holding full model replicas. No dataset, dataloader, loss function
or optimizer is required. Native Qwen3 and Qwen3-MoE declare `ModelAdapterSpec.supports_generation`.
Prediction skips training FSDP wrapping and rejects FSDP/HSDP parameter sharding.
TP/CP/EP/PP/SP, training compilation, activation checkpointing/swapping, HyperOffload and worker recovery
are currently rejected.

## Launch

Update paths in the [example configuration](../../../examples/training_demo/qwen3/predict.yaml), then run:

```bash
torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/predict.yaml
```

Set `checkpoint.restore_from` to a DCP checkpoint path, or use `LATEST` with `checkpoint.checkpoint_dir`.
Missing checkpoints raise an error. Without a restore request, use the weights loaded by the model builder.
Ascend uses `hccl`; NVIDIA uses `nccl`.

## Prediction input and output

### Input and multi-turn conversations

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

The caller can maintain complete `messages`, append a user turn, run prediction and
append the generated text as the next assistant message. `TextTrainer.predict()` reads
the configured input file and returns a run summary; it has no in-memory
`predict(messages)` interface. Each turn therefore writes an input file and reads
`generated_text` from output JSONL, using a new output path or explicitly enabling
overwrite. The same process can reuse the loaded model. The caller owns conversation
history, the interaction loop and context-length management.

### Batched generation, length buckets and KV cache

[`batch_size`](../../../hyper_parallel/predict/batching.py) caps request count per DP lane.
With `bucket_by_length: true`, each lane
sorts its encoded prompts from shortest to longest before forming batches. Model inputs
are rectangular `[request_count, longest_prompt_in_batch]` tensors: shorter prompts
receive left padding and `attention_mask` marks valid positions. Final output still
follows the original input order.

For four requests in the same lane with prompt lengths `100, 900, 120, 880`, a
two-request batch cap and a sufficient token budget:

| Batch policy | Two batches | Total padding |
| --- | --- | --- |
| Input order | `(100, 900)`, `(120, 880)` | 1560 tokens |
| Sorted by length | `(100, 120)`, `(880, 900)` | 40 tokens |

`max_batch_tokens` bounds `request_count × longest_prompt_in_batch`, including prompt
padding but excluding subsequently generated tokens. A batch may therefore contain
fewer than `batch_size` requests. A single oversized request fails without silent
truncation. For example, `batch_size: 4`, `max_batch_tokens: 8192` and
`max_new_tokens: 256` allow up to four requests, at most 8192 padded prompt tokens,
and at most 256 newly generated tokens per request.

Current batching is static: the whole batch completes before the next one starts.
Requests that finish early remain in their batch and are not replaced by new requests.
Continuous batching would remove completed requests and admit new ones during execution;
it is not implemented. This is a scheduling feature. Whether padding is needed depends
on the execution backend's support for variable-length sequences, a separate concern.

Prediction generates tokens through `infer.generate`. `generation.use_cache: true`
caches Key/Value states within one generation call so that later decode steps reuse
the prefix; it can also be disabled. The example enables it. Separate predict calls
do not share a KV cache, so each turn of a caller-managed conversation prefills the
complete history submitted for that turn.

Generation configuration accepts only serializable `GenerationOptions` fields:
`max_new_tokens`, `temperature`, `top_k`, `top_p`, `do_sample`, `eos_token_id`,
`pad_token_id`, `repetition_penalty` and `use_cache`. Special tokens resolve from
explicit task options, model generation metadata, then the tokenizer. An unspecified
pad token may fall back to EOS. Multiple EOS IDs are supported. Runtime adapters,
devices, process groups and RNGs are injected by code and cannot be passed through YAML.

### Output and manifests

Each output line contains:

```json
{"id":"plain","ordinal":0,"sample_idx":0,"generated_text":"Beijing is the capital of China.","prompt_tokens":4,"generated_tokens":8,"finish_reason":"eos"}
```

`generated_tokens` includes a terminating EOS and excludes padding. Text decoding
removes special tokens. `finish_reason` comes from the generation loop and is `eos`,
`length` or `stop`; token limits produce `length`. Code-injected legacy stopping
criteria produce `stop`. This phase emits only `sample_idx=0` per request.

DP requires shared storage. Each lane writes its own shard and completion manifest
in the same per-run staging directory. The
coordinator verifies checksums, ownership, IDs, ordinals and exact input coverage,
then merges in input order and atomically publishes JSONL. Existing output raises an
error unless `overwrite: true`. Failures retain `.<output_name>.<run_id>/` without
automatic resume. An adjacent `.lock` prevents concurrent writers; remove a stale lock
only after confirming the previous process has stopped.

`<output_file>.manifest.json` records input/output checksums, model and checkpoint
sources, tokenizer vocabulary fingerprint, template, resolved generation options,
seed, batching policy and environment versions. JSONL is the atomic commit unit;
the manifest is a separate sidecar whose `output_sha256` verifies consistency.
Sampling uses a per-lane RNG. Even with the same seed, changing DP size, batch
composition or execution order can change a request's sampled answer.

The following B0 extensions in the design are not implemented:

| Extension | Meaning | Current behavior |
| --- | --- | --- |
| `n > 1` | Generate multiple candidates per request, identified by `(input_id, sample_idx)` | One answer per request, with `sample_idx=0` |
| Per-request RNG | Derive an independent random stream from the global seed, request ID and candidate index | Requests in one lane share an RNG; regrouping does not preserve request-level random streams |
| Resume manifest | Validate task identity, detect durably completed candidates and generate only missing ones | Existing manifests record provenance and integrity; retained failure shards do not trigger resume |

Candidate-level resume would not itself restore a partially generated answer's KV
cache or decode state.

## Boundaries

`predict/` does not import Trainer. Trainer owns model assets, restoration and configuration/topology validation.
`Predictor` owns request batching and output, and reuses the generation loop through `infer.generate`.
The legacy `generate(model, input_ids, generation_config, attention_mask)` still returns a Tensor by default.
New calls can use `options=GenerationOptions(...)`, `runtime=GenerationRuntime(...)` and `return_dict=True`
to obtain detailed generation results.
`get_build_steps(run_mode)` returns enabled component names; it neither builds them nor defines execution order.

## Validation entry points

```bash
pytest -q tests/ut/infer tests/ut/predict
pytest -q tests/torch/predict/test_predict_e2e.py
```

System tests use real two-layer randomly initialized FP32 models and synthetic data. They do not establish
coverage for full-size models, BF16, quantization or replacement-operator combinations. System tests require
the matching device/backend; CPU tests cannot replace NPU acceptance.
