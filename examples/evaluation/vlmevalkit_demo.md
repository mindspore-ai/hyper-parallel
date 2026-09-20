# VLMEvalKit image-MCQ demo

Status: **integration contract; Hyper adapter not yet implemented**.

Pin a VLMEvalKit revision and one labeled image multiple-choice development split. Register a model such as
`HyperQwen3VL` whose adapter implements the framework's `generate_inner(message, dataset=None)` contract.

```bash
python run.py \
  --model HyperQwen3VL \
  --data <PINNED_IMAGE_MCQ_DATASET> \
  --mode all \
  --work-dir <output>/vlmevalkit
```

Use the pinned revision's exact CLI. The adapter must preserve interleaved text/image order, use the checkpoint-matched
processor, validate image placeholders, and return one generated string per logical request. VLMEvalKit remains responsible
for prediction persistence, option extraction, optional judge use, and metrics.

Start with a 32-sample native/reference comparison. Save media digests, processor parameters, visual token counts, generated
text, extracted answers, judge/fallback state, failures, and sample ownership. Multi-image and video require separate
validation and must be rejected clearly until supported.

See `.agent/skills/benchmark-evaluation/references/vlmevalkit-demo.md` for the full validation flow.
