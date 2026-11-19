# DeepSeek V4.1 Graph Training

This example runs native `TextTrainer` / `GraphCompiler` training on eight Ascend
NPUs with **FSDP4 + TP2/SP + EP2** (expert EDP4), FP32, sequence length 2304,
micro batch size 1, global batch size 8, and 30 optimizer steps. Dataset packing,
DSA Compressor, and Indexer are enabled; CP and PP are disabled.

The recipe was validated with a four-layer, 16-expert, text-only model with reduced
Engram capacity. The YAML does not crop the model automatically: its architecture
comes from the supplied model configuration. By default, `load_base_model: false`
randomly initializes weights; pretrained checkpoints must be supplied separately.

## Resources and Launch

Install the project and its compatible PyTorch, torch_npu, and CANN/HCCL dependencies.
Run the following commands from the repository root and check that the import
points to this checkout:

```bash
pip install -e .
python -c 'import hyper_parallel; print(hyper_parallel.__file__)'
```

Prepare a local model configuration, its matching tokenizer and Engram assets,
and a **WikiText dataset**. Set their paths using the environment variables or
YAML fields below. These training resources are not bundled with this example.

| Resource | Shell environment variable | YAML field |
| --- | --- | --- |
| Model directory containing `config.json` | `MODEL_DIR` | `model.pretrained_model_name_or_path` |
| Matching tokenizer directory | `TOKENIZER_DIR` | `dataset.model_assets.tokenizer.pretrained_model_name_or_path` |
| WikiText dataset path | `WIKITEXT_TRAIN_FILE` | `dataset.data_path` |
| Matching Engram assets JSON | `ENGRAM_ASSETS_FILE` | `model.config_overrides.v41_engram_assets_path` |

Use absolute paths. The Engram assets must match the model's table sizes and
tokenizer mapping. `ENGRAM_ASSETS_FILE` is optional when the model configuration
already contains a valid assets path. The launcher uses local resources and does
not download or convert models, tokenizers, or datasets.

Launch with the Shell script:

```bash
MODEL_DIR=/path/to/model \
TOKENIZER_DIR=/path/to/tokenizer \
WIKITEXT_TRAIN_FILE=/path/to/wikitext-train.jsonl \
ENGRAM_ASSETS_FILE=/path/to/engram.json \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29743 \
bash hyper_parallel/compile/examples/dsv41_graph/run_deepseek_v41_graph.sh
```

`CONFIG=/path/to/job.yaml` selects another YAML; `OUTPUT_LOG` changes the default
`output/dsv41_graph/train.log`. Append CLI overrides such as
`--training.train_iters=10`. For eager mode, append
`--compile.enabled=false --compile.use_joint_graph=false`.

The script requires the model, tokenizer, and WikiText environment variables and
uses them to override the YAML paths. It defaults to `NPROC_PER_NODE=8`, `TP_SIZE=2`, and `EP_SIZE=2`, derives the FSDP/EDP
degrees, and enables SP when TP is greater than 1.
Set `TP_SIZE=1` for FSDP8+EP2, or also `EP_SIZE=1` for pure FSDP8. Explicit CLI
arguments at the end take precedence over the script's generated arguments.

To configure resources entirely in YAML, copy the supplied YAML, edit the fields
in the table, and invoke the training entry point directly:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
HCCL_OP_EXPANSION_MODE=HOST HCCL_DETERMINISTIC=true \
HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_EXEC_TIMEOUT=600 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --nproc_per_node=8 --master_port=29743 \
  scripts/train_lm.py /path/to/job.yaml
```

Keep `model.torch_dtype` and `model_init_dtype` consistent when changing precision.
The default FP32 recipe completed a ten-step functional run on the cropped model;
this does not establish full-model memory requirements or numerical parity.

## Muon with Graph Training

The supplied `train_deepseek_v41_graph_muon.yaml` enables Muon with full
recomputation and the same FSDP4 + TP2/SP + EP2 topology. Use the resource
variables above and select it with:

```bash
CONFIG=hyper_parallel/compile/examples/dsv41_graph/train_deepseek_v41_graph_muon.yaml \
bash hyper_parallel/compile/examples/dsv41_graph/run_deepseek_v41_graph.sh
```

Its optimizer section is:

```yaml
optimizer:
  _target_: hyper_parallel.components.optim.builders.Muon
  fp32_main_params: false
  muon_config:
    muon_lr: 0.0001
    muon_weight_decay: 0.0
    muon_momentum: 0.95
    muon_nesterov: true
    muon_ns_steps: 5
    muon_ns_variant: asym5
  adamw_config:
    adamw_lr: 0.0001
    adamw_weight_decay: 0.0
```

Matrix parameters use Muon; embeddings, output heads,
and other fallback parameters continue to use AdamW. Forward and backward run
in the joint graph; optimizer updates remain eager.

For BF16 training, set `model.torch_dtype` and `model_init_dtype` to `bfloat16`,
enable `optimizer.fp32_main_params`, and add the precision policy to the existing
`fsdp_config` while retaining its shard degrees:

```yaml
fsdp_config:
  dp_shard_size: 4
  edp_shard_size: 4
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    output_dtype: bfloat16
    cast_forward_inputs: true
```

The runtime binds Muon layouts and creates optional FP32 main shards after the
graph compiler physically shards the model. Optimizer views retain TP/EP/FSDP
metadata, share FP32 model storage, and exclude padded rows from matrix updates.
The current eight-rank cropped recipe has been checked for 30 FP32 and BF16
updates with full recomputation, including independent matrix-update checks.
This functional validation does not establish long-run convergence.

Optimizer state export and import require compiled parameter layouts. State
round trips after binding are supported; cold-start `checkpoint.restore_from`
before the first graph compilation is not yet integrated for this optimizer
path. Parameter storage or layout replacement after binding requires rebuilding
the optimizer and restoring its state.

`benchmark_training.py` records per-rank allocator peaks and synchronized
optimizer timing. `verify_muon_training.py` independently reconstructs selected
matrix updates from real gradients; run it separately because its references
add work and memory. Both accept the training YAML and dotted overrides used
by `scripts/train_lm.py`, and write rank files under `METRICS_DIR`.

See the [development and validation guide](../../../../docs/guide/deepseek_v41_graph_muon.md)
for the design, TorchTitan reference, reproduction commands and measured results.

## Compressor, Indexer, and Packing

Edit the existing YAML entries while retaining its other settings. Model-level
DSA fields can be set in `config.json` or `model.config_overrides`; layer indices
are zero-based.

| Setting | Four-layer example | Purpose |
| --- | --- | --- |
| `v41_compress_ratios` | `[0, 0, 2, 2]` | Per-layer compression; 0 disables that layer's compressed branch. |
| `v41_kv_source_layer_ids` | `[2]` | Compressor KV producer; later compressed layers reuse its bank. |
| `v41_index_source_layer_ids` | `[2]` | Indexer producer layers. |
| `index_topk` | `512` | Maximum number of selected compressed keys. |
| `v41_indexer_loss_coeff` | `0.001` | Indexer KL loss weight; 0 disables the loss, not key selection. |

Keep source layers valid when changing model depth or compression. The runtime
adapter's `compress_ratios: [1, 2]` must cover all active model compression ratios;
it prepares metadata rather than enabling the model's Compressor itself.

Packing is configured through `TokenBatchLoader`, the V4.1 text transform, and the
runtime adapter. When changing sequence length, update these fields together:

- `dataset.data_transform.max_seq_len`
- `dataloader.token_budget`
- `dataloader.collate_fn.pad_to_length`

Set `dataset.data_transform.sample_alignment_size` to the least common multiple
of active compression ratios (2 here). Packed document boundaries flow through
`cu_seq_lens` / `packed_seq_params` to isolate documents and build compressed-key
metadata. Keep `attention_mode: compressed` and `DeepseekV41Runtime`; packing is
not controlled by a single boolean flag.
