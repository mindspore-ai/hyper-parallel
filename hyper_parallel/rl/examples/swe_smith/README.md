# Code Agent:SWE-Smith

**English** | [简体中文](README.zh.md)

Use Codex to modify repositories and independent tests to score results, supporting inference, GRPO training, and checkpoint recovery.
The local Docker socket is used by default; specify remote sandboxes with `--docker-host`.

## Environment Setup

Prepare Docker, model weights, idle Ascend NPUs, and the [pinned RL controller image](../../docker/README.md#repository-code-agent-and-the-qwen35-family).
The host launcher requires Python and PyYAML. Run data preparation in an environment with this project and RL dependencies installed.
Run all commands from the repository root and replace paths with actual locations:

```bash
export PYTHONPATH="$PWD/hyper_parallel/rl:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

<!-- markdownlint-disable-next-line MD033 -->
<a id="模型配方"></a>

## Model Recipes

| Model | Recipe | NPUs |
| --- | --- | --- |
| Qwen3.5-9B (default) | [qwen3_5_9b.yaml](configs/qwen3_5_9b.yaml) | 2 |
| Qwen3.8-27B | [qwen3_8_27b.yaml](configs/qwen3_8_27b.yaml) | 4 |

Both use text-only GRPO and default to eager execution. Select a recipe with `--config` and adjust `--devices` accordingly.

## Image Preparation and Identity

Images must match the sandbox host's CPU architecture. The three published images below support the nine-case ARM64 validation.
Remote amd64 hosts or other tasks require matching images and manifests; see [remote execution](#remote-sandboxes).

### Local ARM64 Nine-Case Images

```bash
SWE_REGISTRY=swr.cn-east-3.myhuaweicloud.com/huawei-hyper-rl/swe-smith
docker --host unix:///var/run/docker.sock pull "$SWE_REGISTRY:schema-arm64-codex-0.152.1"
docker --host unix:///var/run/docker.sock pull "$SWE_REGISTRY:boltons-arm64-codex-0.152.1"
docker --host unix:///var/run/docker.sock pull "$SWE_REGISTRY:oauthlib-arm64-codex-0.152.1"
```

If authentication is required, run `docker login swr.cn-east-3.myhuaweicloud.com` with an authorized account first.
Images already contain source, test dependencies, and Codex. Use [images_arm64.json](configs/images_arm64.json) directly without building.
The manifest pins Image IDs and remote digests; `x86_64` in original dataset names is used only for identity mapping.

## Prepare Subset or Full Data

Download agent-lightning's [filtered data package](https://drive.google.com/file/d/1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1/view)
and use its two JSONL files. The following prepares the [fixed nine cases](configs/selected_ids.txt): six for training, three for validation.

```bash
python -m examples.swe_smith.prepare_data \
  --train-jsonl /data/train_dataset_mixed.jsonl \
  --validation-jsonl /data/val_dataset_filtered.jsonl \
  --instance-ids-file hyper_parallel/rl/examples/swe_smith/configs/selected_ids.txt \
  --output /path/to/swe_smith/data
```

The output directory must be empty; it stores Parquet, instance records, and selection manifests. Omit `--instance-ids-file` for the full dataset,
but supply images for every task; these three images do not cover the complete dataset.

## Unified Local and Remote Entry Point

Set paths and idle NPU IDs on the host. If data preparation ran in a container, supply the corresponding host paths:

```bash
SWE_DATA=/path/to/swe_smith/data
SWE_IMAGES="$PWD/hyper_parallel/rl/examples/swe_smith/configs/images_arm64.json"
SWE_MODEL=/path/to/Qwen3.5-9B
SWE_RUNS=/path/to/swe_smith/runs
SWE_CHECKPOINTS=/path/to/swe_smith/checkpoints
swe_common=(--data-dir "$SWE_DATA" --images-path "$SWE_IMAGES"
            --model-dir "$SWE_MODEL" --devices 0,1)
```

The default endpoint is `unix:///var/run/docker.sock`. The entry point fills configuration and mount paths and starts the controller.
Models and NPUs always reside on the controller host. If privileged device access is required, explicitly append
`swe_common+=(--privileged-controller)`; this option is independent of whether sandboxes run locally or remotely.

### Short-Budget Inference

```bash
python -m examples.swe_smith.launch --mode infer "${swe_common[@]}" \
  --output-dir "$SWE_RUNS/infer-short" \
  --checkpoint-dir "$SWE_CHECKPOINTS/infer-unused"
```

By default, run the nine prepared cases, with at most two calls per case and 512 output tokens per call.
Append `--instance-id ID` to select one case. Inference saves no checkpoints, but the entry point still requires the checkpoint-directory argument.

### One Training Step and New-Process Recovery

```bash
python -m examples.swe_smith.launch --mode train "${swe_common[@]}" \
  --steps 1 --output-dir "$SWE_RUNS/train-step1" \
  --checkpoint-dir "$SWE_CHECKPOINTS/train-step1"

python -m examples.swe_smith.launch --mode train "${swe_common[@]}" \
  --steps 2 --resume "$SWE_CHECKPOINTS/train-step1/step_1" \
  --output-dir "$SWE_RUNS/resume-step2" \
  --checkpoint-dir "$SWE_CHECKPOINTS/resume-step2"
```

`--steps` is the total step count, so resuming to 2 adds only one step. Recovery requires identical model, recipe, and data/image manifest identities.
Use a new results directory each time and reserve space for complete checkpoints; training and resumed runs save separately.
Logs are in `runtime.log`, inference results in `predictions.jsonl`, and call/grading records in `sessions/`.
The entry point cleans its controller and temporary directories on exit while preserving results/checkpoints. After forced interruption, check for leftover sandboxes from that run.

## Remote Sandboxes

The remote host needs Docker and task images for its architecture. The launching host needs an unlocked SSH agent and a verified host key.
ARM64 remotes can pull the images above; amd64 remotes require their own images and manifests.
With task images and a CLI runtime image for the target architecture already available, generate a remote manifest in the RL environment:

```bash
python -m examples.swe_smith.prepare_images \
  --instances /path/to/swe_smith/data/instances.jsonl \
  --output /path/to/swe_smith/images_remote.json \
  --docker-host ssh://USER@HOST:PORT --cli-image TARGET_CLI_IMAGE
```

Skip this step if a complete manifest exists; explicitly add `--pull` for missing task images.
Replace `SWE_IMAGES` above with the host path to the remote manifest, reconstruct `swe_common`, then append:

```bash
swe_common+=(--docker-host ssh://USER@HOST:PORT)
```

The remaining inference/training commands are identical. Remote connection failure raises an error without falling back to local execution.

## Graph Execution and Validation Scope

9B short-budget eager validation covers inference, training, and new-process recovery with local ARM64 and remote amd64 sandboxes.
Actual zero rewards and zero gradients prove only execution; learning acceptance still requires real mixed rewards and nonzero updates.
Reference-budget inference uses [qwen3_5_9b_infer.yaml](configs/qwen3_5_9b_infer.yaml), selected through `launch --mode infer --config`.
See the [development guide](../../docs/swe_smith.md) for grading contracts, graph configuration, resource parameters, and 27B/full-scale validation limits.

## Asynchronous training

Pass `--config hyper_parallel/rl/examples/fully_async/configs/qwen3_4b_swe_smith.yaml`
to the existing launcher, with disjoint `--devices 0,1 --rollout-devices 2,3`.
Qwen3.5/3.8 colocated recipes remain synchronous; see the
[publication and recovery contract](../../docs/architecture.md#rayless-asynchronous-grpo).
