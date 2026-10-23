# Training demos

The root keeps the shared `train_text.py` entry point. Model-owned builders,
recipes, data preparation, launchers, and validation manifests are grouped by
family:

- `qwen3_moe/`: cropped and full-model Qwen3-MoE text demos;
- `deepseek_v41/`: DeepSeek-V4.1 text/VLM crops and validation tooling.

## Pipeline parallel training

For the first dense Qwen3 PP integration, use
`train_cropped_qwen3_pp.yaml`. It crops a local dense Qwen3 to four decoder
layers, creates two pipeline stages, and uses GPipe with four one-sample
micro-batches:

```bash
torchrun --standalone --nproc_per_node=2 \
  --module examples.training_demo.train_text \
  examples/training_demo/train_cropped_qwen3_pp.yaml \
  --model.config_path=/path/to/Qwen3 \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/Qwen3
```

The model builder also supports a self-contained tiny `Qwen3Config` when
called with `config_path=None`, which is useful for unit tests without a
checkpoint.

For a mixed-parallel launch, keep the same model recipe and set the topology
to a product of stage-local and cross-stage domains. For example, on 16
devices:

```yaml
accelerator:
  pp_size: 2
  tp_size: 2
  cp_size: 2
  ep_size: 1                 # use ep_size: 2 for a Qwen3-MoE recipe
fsdp_config:
  dp_shard_size: 2
pipeline:
  enabled: true
  schedule: 1f1b
  pp_microbatch_size: 1
  split: []
  p2p_overlap: true
  p2p_multi_stream: false
```

For Qwen3-MoE, retain the model-family CP/EP `plan_overrides` from
`qwen3_moe/train_parallel_online.yaml`; those overrides provide the explicit CP
attention wrapper and EP local compute factory required by AutoModel.

The ready-to-run eight-device entry enables all currently composable model
parallel modes (`PP=2`, `TP=2`, `CP=2`, `EP=2`, and FSDP over `DP*CP=2`):

```bash
bash examples/training_demo/run_parallel_pp_online.sh /path/to/Qwen3-30B-A3B
```

### PP and VPP configuration

Set `pipeline.enabled: true` together with `accelerator.pp_size > 1`.
When PP is disabled, use `pipeline.enabled: false` and `pp_size: 1`.
`schedule` accepts `gpipe` and `1f1b`. `split` specifies Transformer layer
counts, excluding embedding, final norm, and LM head:

- `split: []` balances layers over the physical PP ranks, with one stage per rank.
- `split: [2, 3, 3, 2]` assigns a ten-layer model to four ranks explicitly.
- `split: [[1, 1, 1, 1], [1, 2, 2, 1]]` assigns the same ten layers to eight
  logical stages on four ranks. Each row is one virtual chunk and each column
  is one PP rank. Logical stages follow row order; rank `r` owns stages `r`
  and `r + pp_size`. Nested splits automatically select Interleaved 1F1B and
  require `schedule: 1f1b`; GPipe with a nested split is rejected.

Each row must have `pp_size` positive integer counts, and all counts must sum
to the model's layer count. Empty nested rows (`[[], []]`) are invalid. The
first logical stage owns embedding; the last owns final norm and LM head.
Every local chunk receives the same stage-local TP/CP/EP/FSDP transformation.

For a ten-layer model on four PP ranks with two virtual chunks per rank:

```yaml
accelerator:
  pp_size: 4
pipeline:
  enabled: true
  schedule: 1f1b
  pp_microbatch_size: 1
  split: [[1, 1, 1, 1], [1, 2, 2, 1]]
  p2p_overlap: true
  p2p_multi_stream: false
```

Trainer derives the DP-local batch size as `training.global_batch_size / dp_size`;
the pipeline micro-batch count is that value divided by
`pipeline.pp_microbatch_size`. Both divisions must be exact. The pipeline
micro-batch size is independent of the data-loader's `training.micro_batch_size`
and is never overwritten by it. The DP-local batch must also satisfy the normal
data-loader batching constraints.

`p2p_overlap` defaults to `true` and defers P2P waits until activation/gradient
consumption. `p2p_multi_stream` defaults to `false`; when enabled, it uses the
core's experimental per-peer communication groups and requires `p2p_overlap: true`.
Both options apply to ordinary PP and VPP, and must match on all ranks.

To run the four-layer dense demo with two virtual chunks on each of two ranks:

```bash
torchrun --standalone --nproc_per_node=2 \
  --module examples.training_demo.train_text \
  examples/training_demo/train_cropped_qwen3_pp.yaml \
  --model.config_path=/path/to/Qwen3 \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/Qwen3 \
  --pipeline.schedule=1f1b \
  '--pipeline.split=[[1, 1], [1, 1]]'
```

## Qwen3-MoE cropped Hugging Face demos

These examples build a layer-cropped Qwen3-30B-A3B model with
`HyperAutoModelForCausalLM.from_config`. They read the complete Hugging Face
configuration, keep all original hidden, attention, vocabulary, and expert
dimensions, and change only `num_hidden_layers` (four by default). No model
checkpoint tensor is loaded, so model parameters are randomly initialized.

## Placement validation from YAML

YAML can directly control placement validation:

```yaml
model:
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
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
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh /path/to/Qwen3-30B-A3B
bash examples/training_demo/qwen3_moe/run_parallel_online.sh /path/to/Qwen3-30B-A3B
```

The default topology uses eight devices with TP=2, CP=2, EP=2, and FSDP. To
enable placement validation without editing YAML:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --model.validate_placement=true
```

Additional typed overrides are forwarded to the Trainer. For example, a
one-step smoke test is:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --training.train_iters=1
```

Logs and generated data are stored under `output/training_demo`.

## Full pretrained model

Both full-model launchers load all 48 layers and the complete Hugging Face
checkpoint through `HyperAutoModelForCausalLM.from_pretrained`. Online uses the
packaged `hyper_parallel/models/qwen3_moe/recipes/train.yaml`;
Offline uses `examples/training_demo/qwen3_moe/train_parallel_full_offline.yaml` because
the two data paths instantiate different Dataset, DataLoader, and collate
targets. Ordinary values can be overridden on the command line, but the typed
configuration interface intentionally does not replace `_target_` values.

The Online launcher tokenizes and packs a deterministic local JSONL file at
runtime. It generates that file under `output/training_demo/data` when needed
and uses a 128-token smoke-test length by default; an appended typed override
can increase the sequence length:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_full_online.sh \
    /path/to/Qwen3-30B-A3B
```

The Offline launcher requires an existing Indexed Dataset and never generates
or downloads one implicitly. Pass the dataset prefix without the `.bin` or
`.idx` suffix:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_full_offline.sh \
    /path/to/Qwen3-30B-A3B \
    /path/to/offline_text_document
```

Both launchers validate the local model `config.json` before starting and force
model/tokenizer loading into `local_files_only` mode. The Offline launcher also
validates both Indexed Dataset files. Missing local assets therefore fail
explicitly rather than triggering a network download. Additional typed Trainer
overrides may be appended to either command.

## DeepSeek-V4.1 Engram and shared compressed attention

`deepseek_v41/train_deepseek_v41_online.yaml` is a four-layer, randomly initialized
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
table. `deepseek_v41/prepare_deepseek_v41_assets.py` changes every active hash bucket to a
different prime near 4096, then recomputes offsets and the embedding row count.
For the active layer-1 table this changes 384,006,168 rows to 100,776 rows while
retaining 3 n-gram orders, 8 hash heads, and a 256-wide embedding. The tokenizer
normalization and hash multipliers remain the V4.1 values.

Online packing emits compact sample boundaries instead of a dense `[S,S]`
attention mask. Each packed sample is aligned to the encoder compression ratio,
so neither CSA2 compressor groups nor Engram n-grams cross sample boundaries.

Prepare the shell using the project
[`installation guide`](../../docs/installation.md), then run TP1 first. A
successful TP1 run creates a marker required by TP2:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash tp1

bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash tp2

bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
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

The model-owned Engram, CSA2, TP/CP/EP, checkpoint, and validation declarations
live together in the
[`deepseek_v41` adapter](../../hyper_parallel/models/deepseek_v41/adapter/).
The generic workflow is documented in the
[`model-integration validation guide`](../../docs/guide/trainer/model_integration_validation.md).

## DeepSeek-V4.1 multimodal Online smoke

The custom validation crop adds the V4.1 ViT, 3x3 aligner, image-boundary
embeddings, image-aware MoE routing, and OpenAI-messages image data transform.
It uses one vision block and 16 routed experts for the validation crop while
retaining the released model dimensions. The model adapter declares per-vision
block, aligner, and mixed-mesh Engram FSDP units plus the actual forward order;
the generic FSDP manager contains no DeepSeek-specific branches.

Prepare the environment and the local image JSONL, then run:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
    /path/to/DeepSeek-V4.1-Flash \
    /path/to/train.jsonl
```

The default data path is
`output/training_demo/deepseek_v41/mm_data/deepseek_v41_messages/train.jsonl`.
The launcher removes a stale success marker before starting and recreates it
only after all 16 ranks complete. Use
[`deepseek_v41_validation.yaml`](deepseek_v41/deepseek_v41_validation.yaml) to
generate reproducible structure, module-parity, precision, checkpoint, and
performance evidence for the selected local environment.
