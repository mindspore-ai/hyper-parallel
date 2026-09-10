# Cropped Qwen3-MoE training demos

These examples build a layer-cropped Qwen3-30B-A3B model with
`HyperAutoModelForCausalLM.from_config`. They read the complete Hugging Face
configuration, keep all original hidden, attention, vocabulary, and expert
dimensions, and change only `num_hidden_layers` (four by default). No model
checkpoint tensor is loaded, so model parameters are randomly initialized.

## Placement validation from YAML

YAML can directly control placement validation:

```yaml
model:
  _target_: examples.training_demo.cropped_qwen3_moe.build_cropped_qwen3_moe
  validate_placement: true
```

The value is retained by `Target`, passed through `BaseTrainer._build_model`,
forwarded by `build_cropped_qwen3_moe` to `HyperAutoModelForCausalLM.from_config`,
and finally used as the infrastructure `validate_mode` value. The typed CLI
override `--model.validate_placement=true` follows the same path. The launch
scripts do not override this field, so editing the YAML remains effective.

## Data modes

- Offline uses the Indexed Dataset format. Its targets are already shifted,
  so `labels_are_shifted: true`; the implicit-mask CP wrapper owns the causal
  mask.
- Online tokenizes and packs JSONL text at runtime. Its labels are unshifted,
  and packed document boundaries use a global block mask accepted by
  `qwen3_moe_flash_attention_cp_mask_wrapper`.

Both launchers automatically generate small deterministic local datasets when
the expected files are absent. This process does not access or download an
external dataset.

## Run

Pass a local Qwen3-30B-A3B Hugging Face directory containing `config.json` and
tokenizer assets as the first argument. The examples deliberately set
`local_files_only: true`: missing assets cause an explicit error instead of an
implicit model or weight download. Prepare the runtime according to the
project installation guide before launching the example.

```bash
bash examples/training_demo/run_parallel_offline.sh /path/to/Qwen3-30B-A3B
bash examples/training_demo/run_parallel_online.sh /path/to/Qwen3-30B-A3B
```

The default topology uses eight devices with TP=2, CP=2, EP=2, and FSDP. To
enable placement validation without editing YAML:

```bash
bash examples/training_demo/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --model.validate_placement=true
```

Additional typed overrides are forwarded to the Trainer. For example, a
one-step smoke test is:

```bash
bash examples/training_demo/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --training.train_iters=1
```

Logs and generated data are stored under `output/training_demo`.

## Kimi-K2.5/K2.6 multimodal demo

`cropped_kimi_vlm.py` builds the multimodal `KimiK25ForConditionalGeneration`
with a layer-cropped text tower and the full 27-layer vision tower. It reads the
top-level native configuration (`trust_remote_code: false`, so the processor
speaks the `image_grid_thw` protocol instead of the repo's remote-code
`grid_thws`), crops only the nested text tower, and parallelizes the whole model
through `HyperAutoModelForImageTextToText.from_config`. The family's TP layout
(MLA down-projections and the vision path replicated, up-projections colwise) is
registered by `hyper_parallel/models/kimi_k25/adapter/registration.py`.

The demo consumes a content-list conversation dataset: every record is
`{"messages": [...], "images": [...]}` where media appear as explicit
`{"type": "image", "url": ...}` items inside the turn, so their position between
text spans is preserved. `prepare_kimi_vlm_data.py` writes such a dataset with
deterministic synthetic images and never downloads anything:

```bash
python -m examples.training_demo.prepare_kimi_vlm_data \
    --output-dir ./output/training_demo/kimi_vlm_data --num-samples 8
```

Pass a local Kimi-K2.5/K2.6 Hugging Face directory containing `config.json` as
the first argument; the launcher generates the demo data when it is absent:

```bash
bash examples/training_demo/run_kimi_vlm.sh /path/to/Kimi-K2.6
```

The default topology is eight devices with FSDP and no TP/CP/PP. The temporary
VLM batch path replicates the batch across TP ranks, so `tp_size > 1` is usable
for the text tower, but CP and PP remain unsupported. With `tp_size > 1` the
template-less vision leaves also need explicit `plan_overrides` parameter specs
to satisfy FSDP owner coverage. Generated data and logs are stored under
`output/training_demo`.

## Full pretrained model

Both full-model launchers load all 48 layers and the complete Hugging Face
checkpoint through `HyperAutoModelForCausalLM.from_pretrained`. Online uses the
packaged `hyper_parallel/models/qwen3_moe/recipes/train.yaml`;
Offline uses `examples/training_demo/train_parallel_full_offline.yaml` because
the two data paths instantiate different Dataset, DataLoader, and collate
targets. Ordinary values can be overridden on the command line, but the typed
configuration interface intentionally does not replace `_target_` values.

The Online launcher tokenizes and packs a deterministic local JSONL file at
runtime. It generates that file under `output/training_demo/data` when needed
and uses a 128-token smoke-test length by default; an appended typed override
can increase the sequence length:

```bash
bash examples/training_demo/run_parallel_full_online.sh \
    /path/to/Qwen3-30B-A3B
```

The Offline launcher requires an existing Indexed Dataset and never generates
or downloads one implicitly. Pass the dataset prefix without the `.bin` or
`.idx` suffix:

```bash
bash examples/training_demo/run_parallel_full_offline.sh \
    /path/to/Qwen3-30B-A3B \
    /path/to/offline_text_document
```

Both launchers validate the local model `config.json` before starting and force
model/tokenizer loading into `local_files_only` mode. The Offline launcher also
validates both Indexed Dataset files. Missing local assets therefore fail
explicitly rather than triggering a network download. Additional typed Trainer
overrides may be appended to either command.
