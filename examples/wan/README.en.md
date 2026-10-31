# Wan2.1 Training Example

This example runs full-parameter Wan2.1 T2V fine-tuning with the HyperParallel
training stack.

The training entry point and configuration file are:

- `scripts/train_dit.py`
- `examples/wan/wan2_1_t2v_14b_full.yaml`

The model target uses
`hyper_parallel.models._transformers.HyperAutoModelForDiT.from_pretrained`
with `model_type: wan`. HyperParallel resolves the Wan `ModelAdapterSpec` and
constructs the official Diffusers `WanTransformer3DModel` directly; there is
no HyperParallel copy of the Wan transformer or its forward method.

The official Diffusers `WanAttnProcessor` is retained. The YAML module
replacement rule selects each official `WanAttention` module and configures
its Diffusers attention backend as `_native_npu`, which dispatches to the NPU
fused-attention implementation without changing parameter names or checkpoint
layout. The transformer returns its standard `sample` output, and
`DiTTrainer.postforward` computes the float32 MSE training loss against
`training_target`.

The existing data processing, model architecture, checkpoint keys, and
training values are preserved.

## Supported Scope

The current Wan adapter supports DP/FSDP2 full-parameter training:

- TP, CP, PP, EP, sequence parallel, and loss parallel must remain disabled.
- The defaults are `global_batch_size=8`, `micro_batch_size=1`, and
  `dp_shard_size=8`.
- `micro_batch_size` must currently remain `1`; the conditioned sample lists
  are unwrapped before calling the official Diffusers forward.
- Activation checkpointing is enabled by default with
  `activation_checkpoint.mode=full`.

## Prepare the Wan2.1 Model

Prepare a complete Diffusers-format Wan2.1 T2V model. You can download it from
Hugging Face:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --local-dir /data/models/Wan2.1-T2V-14B-Diffusers
```

The model root directory should contain the following files and subdirectories:

```text
Wan2.1-T2V-14B-Diffusers/
├── model_index.json
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model*.safetensors
├── tokenizer/
├── text_encoder/
├── vae/
└── scheduler/
```

Pass the model root directory when launching training; do not pass the
`transformer/` subdirectory. The model builder loads the trainable DiT weights
from `transformer/` and loads the frozen tokenizer, text encoder, VAE, and
scheduler from the model root.

`condition_model_name_or_path=auto` makes the condition-model path resolve to
the parent directory of `transformer/` automatically.

## Prepare the Video Data

### Tom-and-Jerry Dataset

The conversion script expects the original dataset to have the following
layout:

```text
Tom-and-Jerry-VideoGeneration-Dataset/
├── captions.txt
├── videos.txt
└── videos/
    ├── sample-000.mp4
    └── ...
```

The number of non-empty lines in `captions.txt` and `videos.txt` must match.
Paths in `videos.txt` are resolved relative to the original dataset root.

Convert the original data to Parquet:

```bash
python examples/wan/convert_tom_and_jerry.py \
  --dataset_path /data/datasets/Tom-and-Jerry-VideoGeneration-Dataset \
  --output_dir /data/datasets/Tom-and-Jerry-VideoGeneration-Dataset-parquet \
  --num_shards 30 \
  --num_proc 32
```

The converted data contains `prompt`, `video_bytes`, and a dataset-source
marker field. The default YAML uses:

```text
./Tom-and-Jerry-VideoGeneration-Dataset-parquet
```

If the data is stored elsewhere, override the path with
`--dataset.data_path`.

## Launch Training

Run the command from the repository root:

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py \
  examples/wan/wan2_1_t2v_14b_full.yaml \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```

To use a local model and local Parquet data:

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py \
  examples/wan/wan2_1_t2v_14b_full.yaml \
  --model.pretrained_model_name_or_path=/data/models/Wan2.1-T2V-14B-Diffusers \
  --model.local_files_only=true \
  --dataset.data_path=/data/datasets/Tom-and-Jerry-VideoGeneration-Dataset-parquet \
  --dataset.data_config.hf_dataset_name=null \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```

## Checkpoints and Outputs

By default, a checkpoint is saved every 250 optimizer steps under:

```text
./outputs/wan2_1_t2v_1_3b_full/checkpoints
```

To resume training, pass the checkpoint directory:

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py \
  examples/wan/wan2_1_t2v_14b_full.yaml \
  --checkpoint.restore_from=/data/output/wan2_1/checkpoints/<step> \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```
