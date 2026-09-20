# VLMEvalKit Mechanism Demo

This is an integration contract and runnable configuration pattern after a `HyperVLMEvalModel` adapter exists. It does not
claim that the adapter is already implemented in HyperParallel.

## First Task

Use one pinned image multiple-choice development split with accessible labels, such as a supported MMBench dev variant.
Keep it generation-based; do not replace the framework's answer extraction with a logits-choice shortcut.

```text
VLMEvalKit dataset.build_prompt
  -> interleaved text/image message
  -> HyperVLMEvalModel.generate_inner
  -> pinned processor creates tensors and visual metadata
  -> Hyper execution group performs visual + language generation
  -> one owner returns generated text
  -> VLMEvalKit prediction persistence and dataset.evaluate
```

## Adapter Surface

The proposed adapter follows the framework model base contract:

```python
class HyperVLMEvalModel:
    def generate_inner(self, message, dataset=None): ...
```

The adapter owns processor selection, message/media order, placeholder validation, conversion to tensors, and the Hyper
execution session. It must reject unsupported multi-image or video inputs explicitly instead of silently using the first
image.

## Invocation Pattern

Register a stable model name such as `HyperQwen3VL` in the pinned VLMEvalKit revision, then run:

```bash
python run.py \
  --model HyperQwen3VL \
  --data <PINNED_IMAGE_MCQ_DATASET> \
  --mode all \
  --work-dir <output>/vlmevalkit
```

Use the pinned revision's exact CLI flags. Keep checkpoint, processor, Hyper topology, precision, image resolution/cropping,
dynamic tiling, generation, and judge configuration in `run_manifest.json`.

## Validation Sequence

1. Native/reference adapter on a fixed 32-sample subset; save the framework-built messages and media digests.
2. Hyper single-device path with identical processor outputs.
3. Hyper DP, followed by individually supported TP/CP/PP/FSDP paths.
4. Multi-image and video only after separate processor, ordering, sampling, cache, and stop-condition tests.
5. Full task after raw text, extracted answers, parser/judge path, and metrics pass subset comparison.

Record image count, resolution/tiling, placeholder order, visual token count, generated token IDs/text, extraction result,
judge or fallback state, and per-sample score. Switching processor/media transform or prompt requires new inference; a
judge-only change may reuse predictions when VLMEvalKit's independent evaluation mode supports it.
