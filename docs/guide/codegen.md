# Codegen User Guide

Codegen writes the module replacements and parallel-strategy patches from a training YAML into model source files. It also produces diffs so that you can inspect the model code selected by your training configuration.

## 1. Prepare the Environment

### 1.1 Install the Current Checkout

Run these commands at the HyperParallel repository root:

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

### 1.2 Specify the Hugging Face Model in YAML

Use your training YAML and set `model.pretrained_model_name_or_path` to a model ID or a local Hugging Face model directory:

```yaml
model:
  pretrained_model_name_or_path: Qwen/Qwen3-30B-A3B
```

For a local model, use its directory, such as `/models/Qwen3-30B-A3B`. Codegen reads the model configuration and resolves its source from this location.

- **Models built into Transformers:** install a Transformers version that supports the model.
- **Models using custom remote code:** provide the referenced `modeling_*.py`, `configuration_*.py`, and their relative-import dependencies. Set `trust_remote_code: true` in the YAML model settings as required by the model.

### 1.3 Prepare the Training YAML

Use the same YAML that you use for training. Configure the model source, parallel degrees, and any required module or forward replacements. If no patches are needed, Codegen still emits the model source, with a comment in the diff explaining that no patch is required.

## 2. Generate Model Source

When you start training through the standard `scripts/train_lm.py` entry point, global rank 0 starts generation in an independent single-process sidecar. A generation failure reports its cause without interrupting native training:

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py path/to/train.yaml
```

To generate source without starting training, run the CLI:

```bash
python -m hyper_parallel.codegen --verbose generate \
  --config examples/training_demo/qwen3_moe/train_parallel_online.yaml
```

The CLI uses one process and does not require an NPU or GPU. Your YAML can still specify multi-device parallel strategies; do not launch the generation CLI with multiple torchrun workers.

## 3. Codegen Bundle

```text
generated/
├── __init__.py
├── modeling_<Model>_gen_npu.py
└── modeling_<Model>_gen_npu.py.diff
```

- `modeling_*_gen_npu.py`: complete model source with module replacements and conditional forward paths.
- `modeling_*_gen_npu.py.diff`: differences from the original modeling source, for code review.
- `__init__.py`: allows the generated directory to be loaded as a Python package.

If the original model uses multiple modeling files, the bundle can contain multiple generated files and their diffs. For example, Kimi K3 separates its top-level and language models into different source files; they form one bundle rather than two independent models. Remote-code bundles can also include required Python dependencies such as `configuration_*.py`.

## 4. Frequently Asked Questions

### Does Generation Require an NPU or GPU

No. Codegen uses CPU/meta models to inspect structure. It does not run training computation or create HCCL/NCCL communication groups.

### Do I Need to Download All Model Weights

No. Codegen reads model configuration and source from the Hugging Face Hub or a local directory that Transformers can recognize. Custom remote-code models also require their Python source dependencies. Generation does not load model weights or training data.

### Why Does a Bundle Contain Multiple Modeling Files

Some models separate their top-level and child models into different source files. Codegen preserves these dependencies and emits the corresponding diffs. The files belong to one model bundle.

### DeepSeek-V4.1 Training Reports Multicore or MegaMoE Library Errors

These errors can also occur in native training. DeepSeek-V4.1 uses Multicore/MegaMoE native extensions. Build and train with compatible Python, PyTorch, torch_npu, and CANN versions, and perform a strict clean build:

```bash
./build.sh --multicore on --strict on --clean
source /usr/local/Ascend/cann/set_env.sh
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

If a library reports `undefined symbol`, or its Python ABI differs from the training environment, clean old build outputs and rebuild in the training environment. Do not bypass the error by disabling MegaMoE or changing model structure. See the [installation guide](../installation.md) for environment requirements.
