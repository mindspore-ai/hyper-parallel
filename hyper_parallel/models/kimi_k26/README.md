# Kimi-K2.6 Training with HyperParallel

**English** | [简体中文](README.zh.md)

Kimi-K2.6 is Moonshot AI's image-text MoE model. Use the four-SuperPoD recipe to
check the multimodal SFT training pipeline and measure throughput on Ascend.

## 1. Support and Validated Recipes

| Recipe | Model, initialization, and data | Ascend resources and parallelism | Validation status |
| --- | --- | --- | --- |
| [4-SuperPoD multimodal SFT benchmark](recipes/train.yaml) | Full 61-layer text depth and 384 routed experts, with a frozen vision tower. Random initialization, synthetic image-text conversations, fixed expert routing. | Four SuperPoDs, 512 logical NPU devices (64 hosts × 8 launcher processes). TP1 / CP1 / EP128 / PP1, dense FSDP128 and expert FSDP4. | The recipe records a historical benchmark at this scale. The one-step command below has no published result. |

The recipe sets `fix_router: true` for benchmarking. This routes a fixed share of tokens
to each destination and ignores gate scores. Use learned routing for correctness and
convergence runs. The real-routing configuration has a separate memory profile.

## 2. Requirements and Asset Preparation

- Use a Linux Ascend PyTorch and torch-npu environment with CANN/HCCL and NPU grouped
  matmul support. Use the same HyperParallel checkout, runtime and operators on every host.
  See the [installation guide](../../../docs/installation.md).
- The recorded setup used Transformers 5.13.0 with native `kimi_k25` and
  `deepseek_v3` support. Install torchvision matching PyTorch for the native
  image processor, Pillow for the demo generator, and `tiktoken` for the
  model repository's `tiktoken.model` vocabulary.

Prepare local assets from [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6).
The benchmark uses the model config, processor config, tokenizer files and chat template.
It initializes the weights randomly. Prepare the following directory on every host
or on shared storage.

```bash
export MODEL_DIR=/path/to/Kimi-K2.6
hf download moonshotai/Kimi-K2.6 \
  config.json preprocessor_config.json tokenizer_config.json tiktoken.model chat_template.jinja \
  --local-dir "${MODEL_DIR}"
```

The launcher sets `model.config_path`, `model.pretrained_model_name_or_path`, and
`dataset.model_assets.pretrained_model_name_or_path` to this directory, with local
loading enabled for both model and processor. Keep both `trust_remote_code`
settings at `false` to use native Transformers classes.

The [data generator](recipes/prepare_kimi_vlm_data.py) writes a JSON array and
synthetic PNG images. Generate the data before launch and remove the duplicated
top-level `images` field. The current loader reserves that field for records
with `<image>` placeholders and rejects it alongside explicit image paths.

```bash
export DATA_DIR="$PWD/output/kimi_k26/kimi_k26_demo_data"
python -m hyper_parallel.models.kimi_k26.recipes.prepare_kimi_vlm_data \
  --output-dir "${DATA_DIR}" --num-samples 1024
python - "${DATA_DIR}/data.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
records = json.loads(path.read_text(encoding="utf-8"))
for record in records:
    record.pop("images", None)
path.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
PY
```

For real data, use the same image-text dialogue format in a JSON array or JSONL.
Relative media paths are resolved from the data file's directory. Each sample
ends with an assistant reply. The transform caps image area at 262,144 pixels and
truncates or pads the text and image tokens to 8192 tokens.

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "images/sample_0000.png"},
        {"type": "text", "text": "Describe the following image."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A colorful geometric pattern."}]
    }
  ]
}
```

## 3. Quick Start

Run from the repository root on every host. Set `NODE_RANK` to a distinct value
from 0 to 63 and `MASTER_ADDR` to a reachable IP address for host 0. Use the
same rendezvous port, model assets and prepared data on every host. For shared
storage, prepare the demo data once and export the same `DATA_DIR` on every host.

```bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.10
NNODES=64 NPROC_PER_NODE=8 MASTER_PORT=6100 RUN_NAME=quick_start \
  bash hyper_parallel/models/kimi_k26/recipes/run_kimi_k26_4sn.sh \
  "${MODEL_DIR}" --training.train_iters=1
```

Expect one completed optimizer step and exit status 0 on every host.
Each host writes `output/kimi_k26/quick_start_rank<node_rank>.log` by default.
Checkpoint saving is disabled. With fixed routing, use completion and timing
to assess the run and reserve loss and expert-load comparisons for learned routing.

## 4. Full Training and Common Configuration Changes

Use the same launcher with real image-text data and a training budget suited to the dataset.
Edit the `plan_overrides` entry for `kimi_k26_ep_compute_fn` in the YAML and set
`fix_router: false` before starting a correctness or convergence run.

| Setting | Recipe defaults and changes |
| --- | --- |
| Data and steps | Pass `--dataset.data_path=/path/to/train.jsonl` to use prepared data. Set `dataset.data_config.filter_trainable=true` if you want the dataset builder to filter untrainable samples. Set `training.train_iters` to the desired budget. The benchmark defaults to six steps. |
| Sequence and images | Adjust `dataset.data_transform.max_seq_len` (8192) and `dataset.data_transform.image_max_pixels` (262144) together when media tokens crowd out the answer. The recipe uses fixed-length padding. |
| Batch and sharding | `training.global_batch_size=512` and `training.micro_batch_size=1` give one microbatch per optimizer step at DP512. The YAML uses `fsdp_config.dp_shard_size=128` and `fsdp_config.edp_shard_size=4`. |
| Precision and activation memory | The recipe uses BF16 model parameters, full recomputation, and no activation swap. The vision tower is frozen through `model.freeze_patterns`. |
| Pretrained initialization | Set `model.load_pretrained=true` and use a compatible, dequantized BF16 checkpoint with the released geometry. The builder loads strictly by key and shape. The upstream release uses INT4 compressed weights and requires conversion for this path. |
| Checkpoint save and resume | Set `checkpoint.save_ckpt=true`, a shared `checkpoint.checkpoint_dir`, and `checkpoint.save_steps` to write distributed checkpoints. To continue from one, use the same model, topology and data and set `checkpoint.restore_from=LATEST` or a specific checkpoint path. Trainer can restore optimizer, scheduler, progress, RNG and data position when the corresponding state is saved. |

## 5. Validation Results

| Validation | Configuration and result |
| --- | --- |
| Local data check | With Transformers 5.13.0, the native processor loaded the listed assets offline. The recipe's Omni dataset encoded a generated record after the preprocessing above and padded it to 8192 tokens. |
| Short-run check | The one-step command above has no published result on the 512-device topology. |
| Numerical checks | [Chunked-loss tests](../../../tests/ut/auto_models/test_kimi_k26_chunk_loss.py) compare loss and gradients with full logits. [CP input tests](../../../tests/ut/auto_models/distributed/test_kimi_k26_cp_input.py) cover input slicing and masking. Full-model learned-routing parity is unverified here. |
| Convergence and resume | Long-run convergence and checkpoint continuation remain unverified for this recipe. |
| Historical performance | The [recipe notes](recipes/train.yaml) report a minimum step time of 18.2617 s, padded throughput of 228,873 tokens/s, and peak allocated memory of 39.19 GB. The benchmark used 512 devices, sequence length 8192, global batch 512, microbatch 1, a frozen vision tower, synthetic data, fixed routing, and no activation swap. The warm-up policy and measured step range are unrecorded. |

## 6. Limitations and Troubleshooting

- Keep TP1 with the recipe's chunked loss, and keep `activation_checkpoint.swap_inputs: false`.
  Input swapping previously failed during backward recomputation with this loss path.
- If a smaller run exhausts device memory, check expert sharding and the sequence,
  image and batch settings. At two SuperPoDs with EP128, sequence length 8192 and global
  batch 4096, a previous run exhausted device memory during backward. The launcher
  enforces 512 devices by default.
- If processor loading fails, check the local processor and tokenizer files and verify
  the native `kimi_k25` implementation in the active Python environment. For strict
  weight-load errors, check BF16 conversion, layer and expert counts, and shard completeness.

## Further Reading

- [Kimi-K2.6 model integration](integration.md)
- [Trainer YAML configuration](../../../docs/guide/trainer/yaml_config.md)
- [Distributed checkpointing](../../../docs/guide/distributed_checkpoint.md)
