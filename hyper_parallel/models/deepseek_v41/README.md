# DeepSeek-V4.1-Flash Training with HyperParallel

**English** | [简体中文](README.zh.md)

DeepSeek-V4.1-Flash is a multimodal MoE model. Start with the 4K image-caption recipe to
exercise the vision encoder, language model, and distributed optimizer on one Ascend host.

## 1. Support and Validated Recipes

| Recipe | Model, initialization, and data | Ascend resources and parallelism | Validation status |
| --- | --- | --- | --- |
| [4K image-caption pretraining](../../../examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml) | All 40 text layers and 32 vision blocks, parameter widths scaled by 1/4, 16 routed experts. Random initialization, online packing of image-caption JSONL. | One Atlas 900 A3/A5 SuperPoD host, 16 logical NPU dies. TP1 / CP4 / EP8 / PP1, dense FSDP8, expert FSDP1, encoder CP4. | No published result for the one-step command below on this topology. |
| [4K text pretraining](../../../examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml) | Same cropped text backbone and random initialization. Online tokenization of local text or generated test data. | Same 16-die topology, with vision and encoder parallelism disabled. | No published short-run result for this exact recipe. |
| [1M image-caption benchmark](../../../examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_1M%20online_128die.yaml) | Cropped widths, 128 routed experts, random initialization, and benchmark-only fixed expert destinations. | Eight hosts × 16 logical dies. TP1 / CP128 / EP128 / PP1, dense FSDP128, expert FSDP1. | The repository provides a configuration and launcher. End-to-end metrics for this configuration are unavailable here. |

The device counts above are logical dies used by the launcher. The published 910B3
measurements use a separate eight-device configuration. See [Validation Results](#5-validation-results).

## 2. Requirements and Asset Preparation

- Prepare a Linux Ascend training environment. The 16-die reference setup uses
  HDK 25.5.5, CANN 9.1.0, Python 3.11.14, PyTorch 2.12.0 CPU wheel, torch-npu
  2.12.0, and Transformers ≥ 5.11.0. See the [HyperParallel installation guide](../../../docs/installation.md)
  and the [model environment setup (Chinese)](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#1-准备环境).
- Build Multicore and install the Omni Ops device package and Python extension. The fused
  CSA2 path also requires matching ops-transformer device operators and the
  `cann_ops_transformer` wheel. Set `OPS_TRANSFORMER_ROOT` to the directory used
  in the build instructions and load the CANN and Omni Ops environments before launching.
  See [fused operator requirements](../../components/functional/README.md#deepseek-v41-fused-operators).

Prepare the configuration, tokenizer, and vision processor assets from the
[official model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
The randomly initialized recipe requires these files locally. Weight shards are optional
for this workflow. The launcher generates the cropped Engram assets when absent.

```bash
export MODEL_DIR="$PWD/model_from_hf/DeepSeek-V4.1-Flash"
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  config.json tokenizer.json tokenizer_config.json chat_template.jinja inference/config.json \
  --local-dir "${MODEL_DIR}"
```

Prepare a local image-caption JSONL file, for example from CC12M. Put one record per line
and resolve relative image paths from the JSONL directory. The processor applies the vision
limits from `inference/config.json`, and the loader packs whole examples into 4096-token
windows at runtime. See the [VLM data guide](../../../docs/guide/data/deepseek_v41_vlm_online_data_guide.md).

```json
{"image": "images/000001.jpg", "caption": "A cat sitting on a chair."}
```

## 3. Quick Start

From the repository root, set `DATA_PATH` to your local image-caption JSONL and run
one optimizer step on the 16 logical dies.

```bash
export DATA_PATH=/path/to/image_caption/train.jsonl
RUN_NAME=quick_start NPROC_PER_NODE=16 \
  bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
  "${MODEL_DIR}" "${DATA_PATH}" \
  --training.train_iters=1 --checkpoint.restore_from=null --debug.check_dataset=warn
```

The launcher passes `MODEL_DIR` as both `model.config_path` and
`dataset.model_assets.config_path`. A successful run exits with status 0, writes
`output/training_demo/deepseek_v41/quick_start.success`, and logs the completed step,
loss, gradient norm, and timing in `output/training_demo/deepseek_v41/run_quick_start.log`.
Checkpoint saving is disabled in this run.

## 4. Full Training and Common Configuration Changes

Remove `--training.train_iters=1` to use the 100-step recipe, or supply the desired
step budget. The [text launcher](../../../examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh)
accepts the model directory followed by an optional JSONL file with a `text` field.
It generates test text if the default dataset is absent and discards partial token
windows to preserve CSA2 and CP alignment.

| Setting | 4K recipe defaults and changes |
| --- | --- |
| Data and objective | Set `dataset.data_path` to use your own data. For image-text SFT, prepare conversations with an assistant reply at the end and set `dataset.training_mode=sft` and `dataset.data_transform.sample_format=messages`. Only the final assistant reply is supervised. |
| Sequence length | Keep `dataset.data_transform.max_seq_len` and `dataloader.collate_fn.pad_to_length` aligned. The VLM recipe also sets `dataloader.token_budget` to the same value (4096). |
| Batch and sharding | `training.global_batch_size=8` and `training.micro_batch_size=1` give two accumulated microbatches with dense DP4. Coordinate the `accelerator.*_size` and `fsdp_config.*_shard_size` fields with the world size. |
| Precision and memory | FP32 initialization, optimizer main parameters, and gradient reduction, with BF16 forward parameters. Both recipes use Host Engram and `activation_checkpoint.mode: full`. |
| Checkpoint save and resume | Enable `checkpoint.save_ckpt=true` and set `checkpoint.save_steps`. Reuse the model, topology, data and checkpoint directory, and set `checkpoint.restore_from=LATEST` to resume. The recipes configure restoration of optimizer, scheduler, training progress, RNG and data position when the corresponding state is saved. |

The launchers default to separate checkpoint directories under
`output/training_demo/deepseek_v41/artifacts/` (`checkpoints` for VLM and
`text_checkpoints` for text). To initialize from official HF weights, use the released
dimensions, all 384 routed experts, released Engram assets, and the complete weight
shards. That path has separate resource requirements.
[Weight-loading instructions (Chinese)](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#加载官方预训练权重)

## 5. Validation Results

| Validation | Configuration and result |
| --- | --- |
| Historical hardware short runs | One node with eight Ascend 910B3 NPUs (64 GiB HBM each), using randomly initialized cropped models and synthetic data. Three text variants completed six steps each with finite loss and gradient norm. The multimodal run completed ten optimizer steps and saved a checkpoint. [Published results](../../../docs/deepseek_v41_article/README.md#4-results-training-metrics-and-model-correctness-validation) |
| Numerical checks | [Checkpoint mapping tests](../../../tests/ut/auto_models/models/deepseek_v41/test_checkpoint_mapping.py) and [chunked loss tests](../../../tests/ut/auto_models/test_deepseek_v41_chunk_loss.py) cover components. End-to-end numerical parity for the 16-die recipes remains unverified here. |
| Convergence and resume | Long-run convergence and continuation from a saved checkpoint remain unverified for the recipes above. |
| Performance | The published results include throughput and memory for the eight-device experiments. Performance for the 16-die and 128-die recipes is unreported here. |

## 6. Limitations and Troubleshooting

- Fused CSA2 requires TP1 and disabled candidate pools. If fused operator imports or
  device symbols are missing, check that the Python wheel and device package come from
  the same ops-transformer source. Use the documented fallback when testing without
  fused attention. [CSA2 configuration](../../components/functional/README.md#configuration)
- The MegaMoe configuration uses EP greater than 1 and PP1. To use grouped GEMM with
  ordinary EP, follow the replacement instructions in the
  [model quick start (Chinese)](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#关闭-megakernel-后使用普通-ep).
- Check the sample schema and local image paths if packing drops records. For the 1M
  benchmark, set `DATA_PATH` to an existing image-caption JSONL file and review the
  runtime paths and rank settings in its
  [multi-node launcher](../../../examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_1M_online_128die.sh).
  The current generator rejects the launcher's `--sample-format image_caption` argument.
  Use `fix_router: false` in the MoE forward rule for learned routing.

## Further Reading

- [DeepSeek-V4.1-Flash architecture and training report](../../../docs/deepseek_v41_article/README.md)
- [Trainer YAML configuration](../../../docs/guide/trainer/yaml_config.md)
- [Distributed checkpointing](../../../docs/guide/distributed_checkpoint.md)
