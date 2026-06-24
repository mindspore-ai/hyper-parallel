# Generate example

This directory contains a minimal validation path for
`hyper_parallel.infer`. The example covers greedy/top-k generation,
KV-cache decode, no-cache fallback, and a small performance baseline script.

## Functional tests

```bash
python -m pytest tests/torch/generate -q
```

Expected result:

```text
33 passed
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
recompute so existing project models can still run.

Sampling supports greedy, top-k, and top-p modes through `GenerationConfig`.
Custom `logits_processor` and `stopping_criteria` callables can be supplied as
lists. Each logits processor receives `(input_ids, scores)` and returns updated
scores; each stopping criterion receives `(input_ids, scores)` and returns a
scalar boolean or scalar tensor.

## Benchmark

Run a small deterministic baseline:

```bash
python examples/generate/benchmark_generate.py \
  --batch-size 4 \
  --prompt-len 32 \
  --max-new-tokens 64 \
  --warmup 2 \
  --repeat 5 \
  --output /tmp/generate_baseline.json
```

The JSON output records latency and generated-token throughput for:

- `prefill`: prompt-only forward latency.
- `with_cache`: model returns `past_key_values`; decode uses single-token steps.
- `no_cache`: model does not return `past_key_values`; decode recomputes the
  whole sequence.

Example output fields:

```json
{
  "batch_size": 4,
  "prompt_len": 32,
  "max_new_tokens": 64,
  "prefill": {
    "mean_seconds": 0.0034,
    "tokens_per_second": 37647.0
  },
  "with_cache": {
    "mean_seconds": 0.0123,
    "tokens_per_second": 20813.0
  },
  "no_cache": {
    "mean_seconds": 0.0456,
    "tokens_per_second": 5614.0
  }
}
```

Use the same command and hardware when comparing future generate changes.

## Project model smoke

The CPU test suite covers no-cache generation against real repository model
classes with small random configurations:

- `Qwen3_5ForCausalLM` through the functional `generate(model, ...)` API.
- `Qwen3_5ForCausalLM` with `GenerateMixin`, using the `model.generate(...)`
  method form.
- `Qwen3_5MoeForCausalLM` through the functional API and the
  `model.generate(...)` method form.

These tests validate that at least one repository model can use the generate
flow without requiring checkpoint downloads. The current Qwen project-model
checks use no-cache fallback because those model forwards do not return
`past_key_values`; cache behavior is covered by cache-capable test models and
the KV cache unit tests.

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
