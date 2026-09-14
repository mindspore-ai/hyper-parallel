# Current HF Model Training Environment

This document records the execution environment used for the current
HyperParallel Hugging Face model integration and precision experiments. The
absolute paths are specific to this validation host; keep them out of committed
example launchers and replace them when moving to another machine.

Snapshot date: 2026-09-10.

## Environment Snapshot

| Item | Current value |
| --- | --- |
| Repository | `/home/ma-user/work/y00512198/hyper-parallel` |
| Git branch | `upstream_master` |
| Git commit | `08901cba5dbed4a849970c9ded8482cb974ab79a` |
| Python command | `/usr/bin/python` |
| Python version | `3.11.12` |
| Torch | `2.9.0+cpu` |
| torch-npu | `2.9.0` |
| CANN | `/usr/local/Ascend/cann-8.5.1` |
| Visible NPUs | 16 |
| Default Transformers | `4.57.6` from system site-packages |
| Required Transformers | `5.13.0` from the isolated Python dependency directory below |

The current shell imports HyperParallel from this checkout:

```text
/home/ma-user/work/y00512198/hyper-parallel/hyper_parallel/__init__.py
```

## Select Transformers 5.13.0

The system installation is Transformers 4.57.6. The experiments must prepend
the following isolated dependency directory to `PYTHONPATH`:

```text
/home/ma-user/work/y00512198/hyper-parallel-nongit-20260910/outputs/pythonpath/transformers-5.13.0
```

Prepare each new shell as follows. Source CANN first, then prepend the required
Transformers directory so that later environment setup cannot move it behind
system site-packages.

```bash
cd /home/ma-user/work/y00512198/hyper-parallel

source /usr/local/Ascend/cann-8.5.1/set_env.sh

export TRANSFORMERS_513_PATH=/home/ma-user/work/y00512198/hyper-parallel-nongit-20260910/outputs/pythonpath/transformers-5.13.0
export PYTHONPATH="${TRANSFORMERS_513_PATH}:${PYTHONPATH:-}"
```

The directory is outside the Git checkout and was preserved when non-Git
experiment artifacts were moved out of the repository. Moving or deleting it
invalidates this environment setup.

Do not append this directory to `PYTHONPATH`: it must come first. The training
shell scripts invoke `python` and `torchrun`, and both child processes inherit the
exported value.

## Mandatory Import Check

Run this check before every experiment or after changing shells:

```bash
python - <<'PY'
from pathlib import Path
import os

import hyper_parallel
import torch
import torch_npu
import transformers

transformers_root = Path(os.environ["TRANSFORMERS_513_PATH"]).resolve()
transformers_file = Path(transformers.__file__).resolve()
hyper_parallel_file = Path(hyper_parallel.__file__).resolve()
checkout = Path("/home/ma-user/work/y00512198/hyper-parallel").resolve()

print("transformers:", transformers.__version__, transformers_file)
print("hyper_parallel:", hyper_parallel_file)
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())

if transformers.__version__ != "5.13.0":
    raise RuntimeError("Transformers 5.13.0 is not active")
if not transformers_file.is_relative_to(transformers_root):
    raise RuntimeError("Transformers was not imported from TRANSFORMERS_513_PATH")
if not hyper_parallel_file.is_relative_to(checkout):
    raise RuntimeError("HyperParallel was not imported from the active checkout")
if not torch.npu.is_available():
    raise RuntimeError("torch-npu cannot access an NPU")
PY
```

Expected key values are:

```text
transformers: 5.13.0 .../transformers-5.13.0/transformers/__init__.py
hyper_parallel: /home/ma-user/work/y00512198/hyper-parallel/hyper_parallel/__init__.py
NPU available: True
NPU count: 16
```

If HyperParallel points elsewhere, reinstall this checkout in editable mode:

```bash
python -m pip install -e .
```

## Model and Offline Data Paths

At the snapshot time:

- `/cache/huggingface` does not exist;
- `/cache/idx` exists but contains no Indexed Dataset files.

Create or populate them before running full-model experiments. The following
variables are examples for this host, not repository defaults:

```bash
QWEN3_MODEL_PATH=/cache/huggingface/Qwen3-30B-A3B
QWEN3_OFFLINE_PREFIX=/cache/idx/qwen3_30b_a3b/train_text_document
```

The model directory must contain the complete Qwen3-30B-A3B checkpoint,
`config.json`, tokenizer assets, and checkpoint index. Training uses local-only
loading and must fail instead of downloading missing files implicitly.

The Offline prefix is passed without `.bin` or `.idx`; both files must exist:

```bash
test -s "${QWEN3_OFFLINE_PREFIX}.bin"
test -s "${QWEN3_OFFLINE_PREFIX}.idx"
```

### Generate Offline Data from Local JSONL

Dataset preparation is an explicit step. For a 4096-token training sequence,
the packer writes 4097-token documents so the reader can construct
`tokens[:-1]` and `tokens[1:]`:

```bash
QWEN3_SOURCE_JSONL=/path/to/train.jsonl

python -m hyper_parallel.data.tools.offline_preparation \
    --dataset-name-or-path "${QWEN3_SOURCE_JSONL}" \
    --output-prefix "${QWEN3_OFFLINE_PREFIX}" \
    --tokenizer-name-or-path "${QWEN3_MODEL_PATH}" \
    --pack-to-seq-len 4096 \
    --workers 8 \
    --partitions 1 \
    --keep-sequential-samples
```

The current Offline examples expect pre-shifted targets and therefore use
`labels_are_shifted: true`. Confirm that the input pipeline performs exactly one
causal shift before drawing precision conclusions.

## Required Precision Overrides

The current training-demo YAML files do not yet contain the complete mandatory
FP32 initialization/main-parameter policy. Pass these typed overrides to each
launcher:

```bash
QWEN3_PRECISION_OVERRIDES=(
    --model_init_dtype=float32
    --fsdp_config.mix_precision.param_dtype=bfloat16
    --fsdp_config.mix_precision.reduce_dtype=float32
    --fsdp_config.mix_precision.output_dtype=bfloat16
    --fsdp_config.mix_precision.cast_forward_inputs=false
    --optimizer.fp32_main_params=true
)
```

Keep `--optimizer.fp32_main_params=true` after the FP32 reduce-dtype override.
CLI overrides are applied one at a time and immediately validated; enabling
main parameters before setting `reduce_dtype=float32` raises a configuration
error.

This policy means:

- model loading or random initialization is finalized in FP32;
- FSDP forward parameter/output policy is BF16;
- gradients reduce into FP32 main gradients;
- FSDP does not automatically cast forward inputs;
- Muon/AdamW updates use the FP32 main-parameter route.

## Training Demo Commands

All current demo launchers use eight processes even though this host exposes 16
NPUs. They write logs and generated smoke data under `output/training_demo`.

### Cropped Model, Offline

This path loads the complete Qwen3 configuration, changes only the layer count,
does not load checkpoint tensors, and automatically creates a deterministic
128-token smoke dataset when absent:

```bash
bash examples/training_demo/run_parallel_offline.sh \
    "${QWEN3_MODEL_PATH}" \
    "${QWEN3_PRECISION_OVERRIDES[@]}" \
    --training.train_iters=10
```

For formal self-consistency, first run a deterministic baseline and save a full
training-state checkpoint; restore every candidate topology from that checkpoint.

### Cropped Model, Online Smoke

Online is a functional check only:

```bash
bash examples/training_demo/run_parallel_online.sh \
    "${QWEN3_MODEL_PATH}" \
    "${QWEN3_PRECISION_OVERRIDES[@]}" \
    --training.train_iters=1
```

### Full Pretrained Model, Offline

This is the formal full-weight precision path. It requires an existing complete
checkpoint and Offline Indexed Dataset:

```bash
bash examples/training_demo/run_parallel_full_offline.sh \
    "${QWEN3_MODEL_PATH}" \
    "${QWEN3_OFFLINE_PREFIX}" \
    "${QWEN3_PRECISION_OVERRIDES[@]}" \
    --training.train_iters=10
```

### Full Pretrained Model, Online Smoke

This uses the packaged Qwen3-MoE recipe and generates only a small local JSONL
smoke input. It is not precision-acceptance evidence:

```bash
bash examples/training_demo/run_parallel_full_online.sh \
    "${QWEN3_MODEL_PATH}" \
    "${QWEN3_PRECISION_OVERRIDES[@]}" \
    --training.train_iters=1
```

## Placement Validation

The cropped examples expose placement validation through the model target:

```bash
bash examples/training_demo/run_parallel_offline.sh \
    "${QWEN3_MODEL_PATH}" \
    "${QWEN3_PRECISION_OVERRIDES[@]}" \
    --model.validate_placement=true \
    --training.train_iters=10
```

Do not use `region_dispatch: false` as evidence that internal Attention TP
placements are valid. Keep a strict TP case with CP disabled when checking q/k
norm, RoPE, and mask placements.

## Experiment Rules

- Offline is the formal precision source; Online is a one-step functional smoke.
- Compare loss and global gradient norm together.
- Verify all ranks load the same checkpoint and consume the same global sample
  order.
- Keep learned MoE routing in formal cases.
- Checkpoint continuation must restore model, optimizer moments, FP32 main
  parameters, scheduler, train state, RNG, and dataloader cursor.
- Do not silently reduce layers, sequence length, topology, or precision after an
  OOM. Record the case as blocked or obtain authorization for a changed case.

## Common Failures

### Transformers Still Reports 4.57.6

The isolated dependency directory is absent from `PYTHONPATH` or appears after
system site-packages. Re-run the environment block and the mandatory import check.

### Main-Parameter Configuration Fails During CLI Parsing

Apply `fsdp_config.mix_precision.reduce_dtype=float32` before
`optimizer.fp32_main_params=true`. The ordering in
`QWEN3_PRECISION_OVERRIDES` is intentional.

### Model or Dataset Is Missing

The full launchers fail explicitly. Populate the user-approved cache and Offline
prefix; do not replace formal data with the generated cropped smoke dataset.

### CANN Ownership Warning

Imports currently print a warning that `/usr/local/Ascend/cann-8.5.1` is owned by
another user. The verified environment still reports 16 available NPUs, but the
warning should be investigated if initialization or operator execution later
fails.
