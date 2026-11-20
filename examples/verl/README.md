# VeRL Launcher Guide

`run.sh` launches the HyperParallel VeRL example. Set up Python, VeRL, and the device runtime before running it.

## Quick Start

Run from the repository root.

```bash
source /path/to/set_env.sh

export TRAIN_DATA_PATH=/path/to/datasets/train.parquet
export VAL_DATA_PATH=/path/to/datasets/test.parquet
export MODEL_PATH=/dev/shm/Qwen3-30B-A3B-Base

bash examples/verl/run.sh trainer.total_training_steps=10
```

The script passes trailing arguments to Hydra as configuration overrides.

```bash
bash examples/verl/run.sh \
    trainer.total_training_steps=10 \
    trainer.default_local_dir=/path/to/experiment/checkpoints
```

To save console output, create the output directory and redirect the script's output.

```bash
bash examples/verl/run.sh \
    trainer.default_local_dir=/path/to/experiment/checkpoints \
    > /path/to/experiment/train.log 2>&1
```

## Configuration and Input Paths

| Environment variable | Default | Purpose |
|---|---|---|
| `HP_SOURCE_ROOT` | Repository root containing the script | Working directory and HyperParallel source tree. Hydra configurations are under `examples/verl` in this tree. |
| `VERL_CONFIG_NAME` | `hyper_parallel` | Hydra configuration name. The default loads `hyper_parallel.yaml`. |
| `TRAIN_DATA_PATH` | `/cache/data/dapo-math-17k/train.parquet` | Training data file. |
| `VAL_DATA_PATH` | `/cache/data/aime-2024/test_unique.parquet` | Validation data file. |
| `MODEL_PATH` | `/cache/data/Qwen3-30B-A3B-Base` | Model and tokenizer path. |

## Debug Mode

| Environment variable | Default | Purpose |
|---|---|---|
| `DEBUG` | `0` | Set to `1` to run with fewer model layers. |
| `DEBUG_LAYERS` | `3` | Model and rollout layer count when `DEBUG=1`. |
| `DEBUG_STEPS` | `1` | Total training steps in debug mode. It also serves as a fallback for the number of rollout cache steps. |

`DEBUG=1` also sets the maximum response length to 1024 and disables validation before and during training.

The launcher uses `DEBUG_LAYERS` for the HyperParallel actor, any enabled reference model, and vLLM rollout.

```bash
DEBUG=1 DEBUG_LAYERS=3 DEBUG_STEPS=2 bash examples/verl/run.sh
```

## Rollout Cache

| Environment variable | Default | Purpose |
|---|---|---|
| `ROLLOUT_MODE` | `off` | Use `off` for online generation, `record` to save rollouts, or `replay` to use saved rollouts. |
| `ROLLOUT_SKIP` | `0` | Set to `1` to force `ROLLOUT_MODE=replay`. |
| `ROLLOUT_CACHE_DIR` | Empty | Cache directory required in `record` and `replay` modes. |
| `ROLLOUT_SKIP_STEPS` | Unset | Number of steps to cache or replay. Use a positive integer `N` for steps `1..N`. |
| `TRAINING_STEPS` | Unset | Fallback for cache step count. Set the training step limit separately with `trainer.total_training_steps=N`. |

The launcher uses the first available value in this order, starting with `ROLLOUT_SKIP_STEPS`, then `TRAINING_STEPS`, then `DEBUG_STEPS`, and finally `1`.

Record a cache, then replay it.

```bash
ROLLOUT_MODE=record ROLLOUT_CACHE_DIR=/path/to/model/layers_48/cache \
ROLLOUT_SKIP_STEPS=2 bash examples/verl/run.sh trainer.total_training_steps=2

ROLLOUT_MODE=replay ROLLOUT_CACHE_DIR=/path/to/model/layers_48/cache \
ROLLOUT_SKIP_STEPS=2 bash examples/verl/run.sh trainer.total_training_steps=2
```

Keep `ROLLOUT_SKIP=0` while recording. The cache must match the model, layer count, input and response lengths, and required steps. Use a separate cache directory for each layer count. Replay still initializes the rollout engine.

## Determinism and Python Environment

| Environment variable | Default or launcher behavior | Purpose |
|---|---|---|
| `DETERMINISTIC` | `0` | Set to `1` to enable `full_determinism` for the actor, ref, and rollout and set the three environment variables below. Other values retain the YAML determinism settings and unset those three variables. |
| `PYTHONHASHSEED` | Defaults to `42` when determinism is enabled | Controls Python hash randomization. The training seed is configured separately. |
| `HCCL_DETERMINISTIC` | Set to `true` when determinism is enabled | Enables HCCL determinism. |
| `CLOSE_MATMUL_K_SHIFT` | Set to `1` when determinism is enabled | Configures Ascend matrix multiplication determinism. |
| `NVTE_ALLOW_NONDETERMINISTIC_ALGO` | Set to `0` when determinism is enabled | Disables nondeterministic Transformer Engine algorithms. |
| `PYTHONPATH` | Prepends `$HP_SOURCE_ROOT` to the existing value | Prioritizes the repository's HyperParallel source. Configure dependency paths, including VeRL, before launching. |
| `PYTHONPYCACHEPREFIX` | `/tmp/pycache` | Python bytecode cache directory. |
| `VERL_USE_EXTERNAL_MODULES` | Prepends `hyper_parallel.integration.verl.patch` to the existing value | Registers the HyperParallel engine and loads compatibility patches alongside the caller's modules. |

When reusing a shell, check which debug, cache, and determinism variables are still exported.
