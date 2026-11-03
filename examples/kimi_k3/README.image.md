# Kimi K3 image, video and text pretraining

This entry composes the native Kimi Linear decoder, MoonViT and patch-merger
projector through `VLMTrainer` and the standard model adapter lifecycle. It uses
local model configuration and processor assets for random-initialized pretraining.
It does not load or convert the published inference checkpoint. Decoder details
and text-only training are described in [README.llm.md](README.llm.md).

The original launcher retains eager CP1/EP1 defaults. The explicit
[parallel recipe](train_media_parallel_pretrain.yaml) composes decoder CP, EP,
encoder CP, packing and model-owned recomputation with TP=PP=1. These controls
are independent and require validation of the selected combination.

## Local assets and launch

Provide a local composite `config.json` declaring `model_type: kimi_k3`, with
`text_config` and `vision_config`, a reviewed local K3 processor/tokenizer
installation, and a JSONL dataset. The default architecture is large: use an
explicit capacity-appropriate configuration. The entry never crops layer counts,
widths, expert counts or image resolution on the caller's behalf.

```bash
PYTHON_BIN=/path/to/python NPROC_PER_NODE=1 \
  bash examples/kimi_k3/run_pretrain.sh \
  /local/config.json /local/Kimi-K3-processor /local/train.jsonl
```

`NPROC_PER_NODE=2` selects FSDP2 and global batch size 2. Other trainer overrides
can follow the three paths. The launcher sets Hugging Face offline mode and
validates local paths; it performs no downloads. Install this checkout in editable
mode and install the dependencies required by the reviewed local processor.
Raw video files additionally require PyAV; image and pre-decoded frame inputs do
not import it.

Checkpoint controls use the shared Trainer schema, for example:

```bash
bash examples/kimi_k3/run_pretrain.sh /local/config.json /local/processor /local/train.jsonl \
  --checkpoint.save_ckpt=true --checkpoint.save_steps=10 --checkpoint.save_epochs=0 \
  --checkpoint.save_optimizer=true --checkpoint.save_train_state=true \
  --checkpoint.save_extra_state_per_rank=true \
  --checkpoint.checkpoint_dir=/local/k3-checkpoints
```

Resume through the same launcher using the same `checkpoint_dir` and
`--checkpoint.restore_from=LATEST --checkpoint.restore_optimizer=true` plus
`--checkpoint.restore_train_state=true --checkpoint.restore_dataloader_state=true`.
The checkpoint must match the saved architecture, topology and dataset recipe;
this example does not qualify a new FSDP topology for continuation.

[train_image_pretrain.yaml](train_image_pretrain.yaml) initializes parameters in
FP32, computes with BF16 FSDP parameters, reduces gradients in FP32, retains FP32
optimizer main parameters, and sets `cast_forward_inputs: false`. It uses the
chunked causal objective and whole-step supervised-token weighting. These
settings preserve the training contract; a smoke run does not establish
large-scale convergence or memory capacity.

## JSONL and media ownership

The standard `build_online_omni_mapping_dataset` entry accepts records containing
`input_ids`, optional `labels`, and `medias`. They use the model's tokenizer IDs
and pre-rendered prompt format. The transform expands each media placeholder into
the actual projected visual-token span before collation. It does not invent an
official video conversation template or insert timestamp text into the prompt.
Existing processor-supported `messages` records remain available; new video
message types are not inferred from a file extension.

Media entries support local images and raw local video containers, for example:

```json
{"input_ids":[163584,163605,42,43],"medias":[{"type":"image","image":"frame.png"}]}
{"input_ids":[163584,163605,42,43],"medias":[{"type":"video","video":"clip.mkv","num_frames":4}]}
```

Relative media paths resolve against the source JSONL directory. Explicit video
sampling uses either `timestamps` or `num_frames` with optional `start_time` and
`end_time`. Times are relative to the video stream's start presentation timestamp.
Automatic duration uses the stream duration, or a streaming scan of the final
frame's presentation end when that duration is absent; it does not infer the
stream length from the container's absolute end timestamp. If no reliable final
frame duration exists, provide explicit timestamps or an end time. Explicit
sampling can have `media_metadata.duration=null` when the true duration is
unknown; that does not claim that all requested bounds were known in advance.

The Python transform also accepts typed `media_frames` containing pre-decoded
PIL images/arrays, or validated patch/grid payloads. JSONL cannot directly encode
PIL objects or tensors. Frames are grouped according to the processor's temporal
limit; the prompt must contain the corresponding media placeholders. Processor
output determines the patch/grid/span counts. Labels for visual positions and
padding are excluded from the objective.

The CPU patch transport handles multiple source records per microbatch. The
baseline collator retains independent rows; `packed: true` combines documents
with explicit boundaries, reset positions and adjusted visual spans. Pre-shifted
targets are required for the parallel recipe so CP slices preserve the global
next-token alignment. Raw labels are shifted once by the loss adapter. Setting
`pre_shift_labels=true` moves that operation to the collator, which records
`labels_are_shifted=true`; the loss and token denominator then consume the aligned
targets without shifting again. Masks follow the selected target alignment.

Media patches and metadata follow one global media order across the local batch.
`media_spans` owns each item by batch row and token interval;
`media_patch_offsets` partitions the flattened patches. Text-only and zero-target
microbatches retain a zero-gradient visual path so every FSDP rank visits the same
visual units. `media_metadata` retains source identity and video sampling
provenance on the host and is not an input to the model's mathematical operations.

## Verification and remaining scope

The recipe enables `model_integration.mode: runtime` for structure,
materialization and runtime modality diagnostics. The model-owned validation
builder is
`hyper_parallel.models.kimi_k3.adapter.validation.model_builder.build_multimodal_validation_model`.
It constructs the complete locally configured architecture without downloading
weights, replacing modules or cropping the model.

CPU checks cover the full configured vision depth, visual outputs and all visual
parameter gradients, image/video/text collation, standard JSONL loading and
loader resume. These are implementation and integration checks; independent
released-model vision parity and complete distributed numerical qualification
remain separate gates. Candidate-tree device evidence must be recorded for the
actual recipe before treating a topology as accepted.

The parallel recipe runs through the standard VLM entry:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module scripts.train_vl examples/kimi_k3/train_media_parallel_pretrain.yaml
```

Set local model/processor/data paths and explicitly select compatible text kernels
as described in the LLM example. Decoder CP, encoder CP and EP have independent
sizes; adjust FSDP groups with the world size. Exact visual block replacements
follow the configured vision depth. Enabling `vision_config.recompute` adds those
blocks to the Trainer recomputation provider rather than nesting a second vision
checkpoint implementation.

Fused visual kernels, swap combinations, official video message rendering,
real-corpus convergence and released-width capacity still require separate work.
Runtime validation records include expensive diagnostic observations; their step
times must not be presented as steady-state training performance.
