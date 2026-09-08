# Generate example

This directory contains a minimal validation path for
`hyper_parallel.infer`. The example covers greedy/top-k/top-p generation,
KV-cache decode, no-cache fallback, HuggingFace alignment, and a small
performance baseline script.

## Functional tests

```bash
python -m pytest tests/torch/generate -q
```

Expected result:

```text
44 passed
```

## Python API

```python
import torch

from hyper_parallel.infer import GenerationConfig, generate

model.eval()
input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
output_ids = generate(
    model,
    input_ids=input_ids,
    generation_config=GenerationConfig(
        max_new_tokens=16,
        do_sample=False,
        eos_token_id=None,
    ),
)
```

If the model returns `past_key_values`, decode uses KV cache. If the model
does not return `past_key_values`, generation falls back to full-sequence
recompute.

Sampling supports greedy, top-k, and top-p modes through `GenerationConfig`.
Custom `logits_processor` and `stopping_criteria` callables can be supplied as
lists. Each logits processor receives `(input_ids, scores)` and returns updated
scores; each stopping criterion receives `(input_ids, scores)` and returns a
scalar boolean or scalar tensor.

## Benchmark

Run a small deterministic baseline:

```bash
python examples/generate/benchmark_generate.py \
  --device npu \
  --batch-size 4 \
  --prompt-len 32 \
  --max-new-tokens 64 \
  --warmup 2 \
  --repeat 5 \
  --output /tmp/generate_baseline.json
```

The JSON output records latency and token throughput for:

- `prefill`: prompt-only forward latency.
- `with_cache`: model returns `past_key_values`; decode uses single-token steps.
- `no_cache`: model does not return `past_key_values`; decode recomputes the
  whole sequence.

Key output fields:

- top-level: `device`, `batch_size`, `prompt_len`, `max_new_tokens`,
  `vocab_size`, `warmup`, `repeat`.
- `prefill`: `mean_seconds`, `min_seconds`, `max_seconds`,
  `tokens_per_second`.
- `with_cache` and `no_cache`: `calls`, `first_call`, `last_call`,
  `output_shape`, `mean_seconds`, `min_seconds`, `max_seconds`,
  `tokens_per_second`.

Use the same command and hardware when comparing future generate changes.

## HuggingFace alignment

Run a real causal-LM alignment check with a local HuggingFace model:

```bash
export MODEL_PATH=/data/models/Qwen3-4B-Instruct-2507

python examples/generate/hf_alignment_check.py \
  --model "$MODEL_PATH" \
  --prompt "用一段话简单介绍Mindspore。" \
  --max-new-tokens 128 \
  --logits-compare-steps 128 \
  --device npu \
  --trust-remote-code \
  --output /tmp/generate_hf_alignment_128.json
```

The helper runs HuggingFace greedy `model.generate(...)`,
`hyper_parallel.infer.generate(..., use_cache=True)`, and
`hyper_parallel.infer.generate(..., use_cache=False)` on the same prompt. It
fails if:

- HuggingFace ids and HyperParallel cache ids differ;
- HyperParallel cache and no-cache ids differ;
- cache/no-cache decode logits cosine similarity falls below the configured
  threshold.

The output JSON should contain the following validation fields:

```json
{
  "prompt": "用一段话简单介绍Mindspore。",
  "generated_new_tokens": 128,
  "hf_vs_hyper_cache_ids_match": true,
  "hyper_cache_vs_no_cache_ids_match": true,
  "logits_compare_steps": 128,
  "logits_cosine_min": 0.9996622800827026,
  "logits_cosine_pass": true,
  "logits_cosine_threshold": 0.999
}
```

The same JSON also includes decoded `hf_text`, `hyper_cache_text`, and
`hyper_no_cache_text` fields, which can be inspected to confirm the generated
text for the prompt.

## Transformers model smoke

The CPU test suite covers generation against a Transformers causal language
model initialized from a small offline config:

- `LlamaForCausalLM` through the functional `generate(model, ...)` API.
- `LlamaForCausalLM` with `GenerateMixin`, using the `model.generate(...)`
  method form.

These tests validate the Transformers integration without requiring checkpoint
downloads. Cache behavior is also covered by cache-capable test models and the
KV cache unit tests.

## Distributed boundary

The current implementation provides the model-agnostic generate loop and KV
cache container. Tensor-parallel inference can gather vocab-sharded logits
before sampling by enabling `GenerationConfig(gather_logits=True)` and passing
the tensor-parallel process group through `logits_process_group` when needed.

Context-parallel prefill can select final-token logits from the rank that owns
the global last prompt token by setting `context_logits_rank` and passing the
context-parallel process group through `context_process_group`. This keeps
sampling decisions identical across ranks after sequence-sharded prefill.

Context-parallel KV cache decode is available as an opt-in path by setting
`GenerationConfig(context_parallel_cache=True)`. The model forward may either
return a full prefill cache, which `generate` shards into a
`ContextParallelKVCache`, or return a local cache together with
`sequence_shard_info`. During cached decode, CP-aware models must accept the
local `past_key_values`, the current `sequence_shard_info`, and
`global_seq_len`, then return updated local K/V tensors plus the next
`sequence_shard_info`.

Contiguous prefix cache reuse is supported by passing
`prefix_past_key_values` and `prefix_attention_mask` in `GenerationConfig`.
The prefix cache is treated as a continuous history immediately before
`input_ids`; returned token ids contain only `input_ids` and newly generated
tokens, not the reused prefix tokens. For CP prefix cache, callers may pass a
full prefix cache or a local prefix cache with `prefix_sequence_shard_info`.

Supported CP cache boundary:

- contiguous sequence shards only;
- append-only decode positions;
- contiguous prefix cache reuse before `input_ids`;
- `context_logits_rank` is local to `context_process_group`;
- packed/non-contiguous cache layouts are not part of this generic generate
  path because they require explicit per-token position or segment metadata
  and model-side attention support.

Run the end-to-end CP cache check with:

```bash
python -m torch.distributed.run --nproc_per_node=2 examples/generate/cp_generate_check.py
```

For models that already return full replicated logits on each rank, the
single-process API can be used directly without enabling logits gathering.
