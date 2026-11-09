# LlamaFactory Training Guide

**English** | [简体中文](llama_factory_hyperparallel_sft_guide.zh.md)

## Versions and Examples

**Use the specified versions exactly. Compatibility depends closely on the torch, torch-npu, CANN, and hardware combination.**

Base environment

| **Item** | **Configuration** |
|----------|-----------|
| Compute device | Ascend 910C |
| CANN     | 8.5.2     |
| Python   | 3.11.10   |

Frameworks and dependencies

| **Framework** | **Version** | **Source** |
|--------------|-------------|---------------------------------------------------|
| LlamaFactory | master      | <https://github.com/hiyouga/LlamaFactory>         |
| HyperParallel | master      | <https://gitcode.com/mindspore/hyper-parallel.git> |
| Transformers | 4.57.1      |                                                   |
| Accelerate   | 1.11.0      |                                                   |
| torch        | 2.7.1       |                                                   |
| torch_npu | 2.7.1.post4 | Install the matching torch_npu version; this value is a reference |
| torchvision  | 0.22.1      |                                                   |
| torchaudio   | 2.7.1       |                                                   |
| trl          | 0.24.0      |                                                   |
| datasets     | 4.0.0       |                                                   |
| peft         | 0.18.0      |                                                   |

## YAML Configuration

`qwen3vlmoe_full_sft_fsdp2.yaml`

```yaml
# Start FSDP2 fine-tuning
# accelerate launch \
#     --config_file examples/accelerate/fsdp2_config.yaml \
#     src/train.py examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml
# Change `num_processes` in fsdp2_config.yaml to 16 in A3

### model
model_name_or_path: /path/to/Qwen3-VL-30B-A3B-Instruct
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true
use_v1_kernels: true
flash_attn: fa2

### enable hyper parallel FSDP
use_hyper_parallel: true
### enable hyper parallel CP
hyper_parallel_cp_size: 2

### method
stage: sft
do_train: true
finetuning_type: full
disable_gradient_checkpointing: false

### dataset
dataset: llava_1k_en, llava_1k_zh
template: qwen3_vl
cutoff_len: 16384
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/Qwen3-VL-30B-A3B-Instruct/full/sft
logging_steps: 1
save_steps: 1000
max_steps: 100
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
seed: 1234
```

## Launch Commands

Download the model weights first, then replace model_path in the YAML file with the downloaded directory path.

accelerate config:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16  # or fp16
num_machines: 1  # the number of nodes
num_processes: 16  # the number of GPUs in all nodes
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Launch HyperParallel FSDP with:

```bash
accelerate launch --config_file examples/accelerate/fsdp2_config.yaml src/train.py examples/ascend/qwen3vlmoe_full_sft_fsdp2.yaml
```

This runs the HyperParallel FSDP training workflow.

## Numerical Validation

### Torch FSDP / Hyper FSDP Comparison

**Model:** Qwen3-VL-30B-A3B-Instruct

**Configuration:**

- Baseline torch FSDP, ours hyper FSDP

- 16 cards

- BF16

- per_device_batch_size=1

- cutoff_len=16384

![Torch FSDP and Hyper FSDP numerical comparison](../images/llama_factory_hyperparallel_sft_guide_image1.png)

### Hyper FSDP + CP Numerical Results

**Model:** Qwen3-VL-30B-A3B-Instruct

**Configuration:**

- Hyper FSDP + CP

- 16 cards

- CP size 2

- BF16

- per_device_batch_size=1

- cutoff_len=16384

![Hyper FSDP + CP numerical results](../images/llama_factory_hyperparallel_sft_guide_image2.png)

### Hyper FSDP / Hyper FSDP + CP Comparison

**Model:** Qwen3-VL-30B-A3B-Instruct(num_hidden_layers=24)

**Configuration:**

- Baseline 8 NPU FSDP, ours 8NPU FSDP + CP_SIZE 2

- 16 cards

- BF16

- per_device_batch_size=1

- cutoff_len=16384

![Hyper FSDP and Hyper FSDP + CP numerical comparison](../images/llama_factory_hyperparallel_sft_guide_image3.png)

## Performance

| **Configuration** | **Performance** |
|-----------------|-------------|
| torch FSDP      | 4.4 s/iter  |
| hyper FSDP      | 3.54 s/iter |
| hyper FSDP + CP | 2.96s/iter  |
