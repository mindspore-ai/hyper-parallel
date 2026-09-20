# OpenCompass Mechanism Demo

HyperParallel provides `HyperOpenCompassModel` for single-process MMLU PPL evaluation against OpenCompass `0.5.4`. The
distributed and generation paths remain unsupported and must not be inferred from this first adapter.

## First Task

Use the pinned OpenCompass MMLU PPL configuration as the first alignment task. It exercises sequence scoring before the
more complex autoregressive generation path.

Responsibilities:

```text
OpenCompass MMLU config
  -> dataset / retriever / prompt template
  -> PPLInferencer
  -> HyperOpenCompassModel.get_ppl(prompts, ...)
  -> Hyper execution group scores complete candidate sequences
  -> one owner returns scores in request order
  -> OpenCompass evaluator and summary
```

Do not assume `get_ppl` means answer-token-only NLL. Match the pinned OpenCompass reference model's mask, denominator,
normalization, and return value exactly. A faster four-label-logit path is allowed only after proving equivalence to the
selected task configuration.

## Adapter Surface

The adapter provides:

```python
class HyperOpenCompassModel:
    def get_token_len(self, prompt: str) -> int: ...
    def get_ppl(self, prompts: list[str], **kwargs): ...
```

The adapter owns Hyper model loading, tokenization, full-sequence causal NLL scoring, mask-length handling, and ordered
NumPy results. It fails fast for `world_size > 1` and for generation until those paths have independent parity evidence.

## Configuration Pattern

Use `examples/evaluation/opencompass_hyper_mmlu.py`. It reads explicit environment variables, imports the pinned MMLU PPL
config, and defaults to a 32-sample single-subject smoke run.

Run through the OpenCompass CLI used by the pinned revision, for example:

```bash
opencompass examples/evaluation/opencompass_hyper_mmlu.py \
  --work-dir <output>/opencompass --debug
```

Use the revision's documented CLI if it differs. Do not encode external checkout paths in the repository example.

## Validation Sequence

1. Native/reference adapter on a fixed 32-sample subset.
2. Hyper single-device adapter on the same expanded prompts and candidates.
3. Add distributed execution only after result ownership and distributed-vocabulary NLL have dedicated parity tests.
4. Full MMLU only after candidate scores and selected labels pass subset parity.

Save expanded prompts or digests, token IDs, scoring masks, valid-token denominators, candidate scores, predicted labels, and
per-subject plus micro accuracy. Verify that task-worker count reserves whole Hyper execution groups instead of allocating
one independent task to every model-parallel rank.
