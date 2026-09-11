# Cropped Hugging Face training demos

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
- Online tokenizes and packs JSONL text at runtime. Its transform emits
  pre-shifted labels, and packed document boundaries use a global block mask accepted by
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

## DeepSeek-V4.1 Engram and shared compressed attention

`train_deepseek_v41_online.yaml` is a four-layer, randomly initialized
DeepSeek-V4.1 text crop. Four layers are the minimum that execute all requested
paths: layer 1 owns Engram, layer 2 publishes compressed KV and Lightning
Indexer selections plus compact hierarchical candidate blocks, and layer 3
performs a fresh Reindex over that candidate pool. The crop also enables the
PanGu-style selected-TopK Indexer KL training path. Vision, DSpark, the second
Engram layer, later compressed-attention source groups, PP-stage shadow
indexers, runtime KV-cache decode, and FP4 QAT remain outside this validation
crop. Raw KV, shared compressed KV, and compressed index K use PanGu-style
asynchronous KV-all-gather CP.

The Engram table is scaled consistently instead of truncating a checkpoint
table. `prepare_deepseek_v41_assets.py` changes every active hash bucket to a
different prime near 4096, then recomputes offsets and the embedding row count.
For the active layer-1 table this changes 384,006,168 rows to 100,776 rows while
retaining 3 n-gram orders, 8 hash heads, and a 256-wide embedding. The tokenizer
normalization and hash multipliers remain the V4.1 values.

Online packing emits compact sample boundaries instead of a dense `[S,S]`
attention mask. Each packed sample is aligned to the encoder compression ratio,
so neither CSA2 compressor groups nor Engram n-grams cross sample boundaries.

Prepare the shell using
[`current_hf_model_environment.md`](../../docs/guide/trainer/current_hf_model_environment.md),
then run TP1 first. A successful TP1 run creates a marker required by TP2:

```bash
bash examples/training_demo/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash tp1

bash examples/training_demo/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash tp2

bash examples/training_demo/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash cp2
```

All modes use 16 processes, Online tokenization, a 4096-token sequence, EP=16,
and the mandatory FP32-main-parameter policy. TP1 uses FSDP=16; TP2 uses
FSDP=8 and enables sequence parallel so Engram's replicated fusion projections
operate on disjoint token slices, matching PanGu's `SequenceParallelLinear`
contract. CP2 keeps TP=1 and uses asynchronous Colossal/KV-all-gather CP; it
does not use Ulysses sequence-to-head exchange. The launcher reads only local
config/tokenizer files and creates the scaled Engram metadata plus deterministic
Online JSONL under `output/training_demo/deepseek_v41`.

The algorithm comparison, TP/CP/EP placement rationale, parameter inventory,
and validation results are recorded in
[`deepseek_v41_mhc_engram_migration_report.md`](../../docs/guide/trainer/deepseek_v41_mhc_engram_migration_report.md).
The DSA module and CP design, including why the legacy MLA/DSA Ulysses wrapper
does not fit CSA2, are documented in
[`deepseek_v41_dsa_cp_adapter_analysis.md`](../../docs/guide/trainer/deepseek_v41_dsa_cp_adapter_analysis.md).

## DeepSeek-V4.1 native multimodal Online smoke

The multimodal recipe adds the native V4.1 ViT, 3x3 aligner, image-boundary
embeddings, image-aware MoE routing, and OpenAI-messages image data transform.
It uses one vision block and 16 routed experts for the validation crop while
retaining the released model dimensions. The model adapter declares per-vision
block, aligner, and mixed-mesh Engram FSDP units plus the actual forward order;
the generic FSDP manager contains no DeepSeek-specific branches.

Prepare the environment and the local image JSONL, then run:

```bash
bash examples/training_demo/run_deepseek_v41_vlm_online.sh \
    /path/to/DeepSeek-V4.1-Flash \
    /path/to/train.jsonl
```

The default data path is
`output/training_demo/deepseek_v41/mm_data/deepseek_v41_messages/train.jsonl`.
The launcher removes a stale success marker before starting and recreates it
only after all 16 ranks complete. FSDP design, parameter ownership, validation
evidence, and remaining full-model work are recorded in
[`deepseek_v41_multimodal_fsdp_report.md`](../../docs/guide/trainer/deepseek_v41_multimodal_fsdp_report.md).
The 16-card, 4K, 100-step Online result and reproducible loss curve are in
[`deepseek_v41_flash_hyperparallel_100step_report.md`](../../docs/guide/trainer/deepseek_v41_flash_hyperparallel_100step_report.md).
